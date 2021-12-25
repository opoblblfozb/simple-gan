import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import trange

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms

def get_device(gpu_id=-1):
    if gpu_id >= 0 and torch.cuda.is_available():
        return torch.device("cuda", gpu_id)
    else:
        return torch.device("cpu")


device = get_device(gpu_id=0)

# Transform を作成する。
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])]
)

# Dataset を作成する。
download_dir = "./data"  # ダウンロード先は適宜変更してください
dataset = datasets.MNIST(download_dir, train=True, transform=transform, download=True)

# DataLoader を作成する。
batch_size = 128  # バッチサイズ
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.main = nn.Sequential(
            # fc1
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(0.2, inplace=True),
            # fc2
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2, inplace=True),
            # fc3
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            # fc4
            nn.Linear(512, output_dim),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.main(x)

class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.main = nn.Sequential(
            # fc1
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            # fc2
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            # fc3
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            # fc4
            nn.Linear(128, 1),
            nn.Sigmoid(),
            nn.Flatten(),
        )

    def forward(self, x):
        return self.main(x)

######################################################################
latent_dim = 100  # ノイズの次元数
data_dim = 28 * 28  # データの次元数

# 学習過程で Generator が生成する画像を可視化するためのノイズ z
fixed_z = torch.randn(100, latent_dim, device=device)

# ラベル
real_label = 1
fake_label = 0

# Generator を作成する。
G = Generator(latent_dim, data_dim).to(device)
# Discriminator を作成する。
D = Discriminator(data_dim).to(device)

# 損失関数を作成する。
criterion = nn.BCELoss()

# オプティマイザを作成する。
lr = 0.0002
G_optimizer = optim.Adam(G.parameters(), lr=lr)
D_optimizer = optim.Adam(D.parameters(), lr=lr)

def D_train(x):
    D.zero_grad()
    # (N, H, W) -> (N, H * W) に形状を変換する。
    x = x.flatten(start_dim=1)

    # 損失関数を計算する。
    # 本物のデータが入力の場合の Discriminator の損失関数を計算する。
    y_pred = D(x)
    y_real = torch.full_like(y_pred, real_label)
    loss_real = criterion(y_pred, y_real)

    # 偽物のデータが入力の場合の Discriminator の損失関数を計算する。
    z = torch.randn(x.size(0), latent_dim, device=device)
    y_pred = D(G(z))
    y_fake = torch.full_like(y_pred, fake_label)
    loss_fake = criterion(y_pred, y_fake)

    loss = loss_real + loss_fake

    # 逆伝搬する。
    loss.backward()

    # パラメータを更新する。
    D_optimizer.step()

    return float(loss)

def G_train(x):
    G.zero_grad()

    # 損失関数を計算する。
    z = torch.randn(x.size(0), latent_dim, device=device)
    y_pred = D(G(z))
    y = torch.full_like(y_pred, real_label)
    loss = criterion(y_pred, y)

    # 逆伝搬する。
    loss.backward()

    # パラメータを更新する。
    G_optimizer.step()

    return float(loss)

def generate_img(G, fixed_z):
    with torch.no_grad():
        # 画像を生成する。
        x = G(fixed_z)

    # (N, C * H * W) -> (N, C, H, W) に形状を変換する。
    x = x.view(-1, 1, 28, 28).cpu()
    # 画像を格子状に並べる。
    img = torchvision.utils.make_grid(x, nrow=10, normalize=True, pad_value=1)
    # テンソルを PIL Image に変換する。
    img = transforms.functional.to_pil_image(img)

    return img

def train_gan(n_epoch):
    G.train()
    D.train()

    history = []
    for epoch in trange(n_epoch, desc="epoch"):

        D_losses, G_losses = [], []
        for x, _ in dataloader:
            x = x.to(device)
            D_losses.append(D_train(x))
            G_losses.append(G_train(x))

        # 途中経過を確認するために画像を生成する。
        img = generate_img(G, fixed_z)
        print(epoch)
        # 途中経過を記録する。
        info = {
            "epoch": epoch + 1,
            "D_loss": np.mean(D_losses),
            "G_loss": np.mean(G_losses),
            "img": img,
        }
        history.append(info)

    history = pd.DataFrame(history)

    return history


history = train_gan(n_epoch=50)

def plot_history(history):
    fig, ax = plt.subplots()

    # 損失の推移を描画する。
    ax.set_title("Loss")
    ax.plot(history["epoch"], history["D_loss"], label="Discriminator")
    ax.plot(history["epoch"], history["G_loss"], label="Generator")
    ax.set_xlabel("Epoch")
    ax.legend()
    plt.savefig("./result/loss_history.png")
    plt.show()



plot_history(history)

def create_animation(imgs):
    """gif アニメーションにして保存する。
    """
    imgs[0].save(
        "./result/history.gif", save_all=True, append_images=imgs[1:], duration=500, loop=0
    )


# 各エポックの画像で gif アニメーションを作成する。
create_animation(history["img"])