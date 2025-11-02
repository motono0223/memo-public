VADv2の学習環境をDockerで構築するための手順と設定ファイル（`Dockerfile`、`docker-compose.yml`）を作成しました。

この手順は、`priest-yang/VADv2` のGitHubリポジトリ（VADv2の参照実装の一つ）に基づき、**CUDA 11.3** と **PyTorch 1.10.1** を使用することを前提としています。

-----

## 1\. Dockerfileの作成

まず、プロジェクトのルートディレクトリ（`VADv2`リポジトリをクローンする場所）に、`Dockerfile` という名前で以下のファイルを作成します。

この`Dockerfile`は、特定のPyTorchとCUDAバージョンをベースにし、`mmcv`や`nuplan-devkit`など、VADv2に必要な依存関係をインストールします。

```dockerfile
# ベースイメージとして、PyTorch 1.10.1 と CUDA 11.3 を指定
FROM pytorch/pytorch:1.10.1-cuda11.3-cudnn8-runtime

# 作者情報（任意）
LABEL maintainer="your_email@example.com"

# 環境変数の設定
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Tokyo

# 必要なAPTパッケージのインストール
RUN apt-get update && apt-get install -y \
    git \
    wget \
    vim \
    unzip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 作業ディレクトリの作成と設定
WORKDIR /workspace

# 必要なPythonライブラリ（mmcv-full）のインストール
# VADv2はmmcv-full 1.4.0 (torch 1.10.x, cu113) を要求します
RUN pip install mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10/index.html

# mmdetection と mmsegmentation のインストール
RUN pip install mmdet==2.20.0 mmseg==0.20.0

# VADv2のrequirements.txtに記載されている他のライブラリをインストール
# Cython, nuplan-devkit などが含まれます
RUN pip install \
    Cython \
    addict \
    pyquaternion \
    nuplan-devkit \
    scipy \
    mcdict \
    shapely \
    tqdm \
    tensorboard \
    pyyaml \
    einops \
    nuscenes-devkit \
    scikit-image \
    matplotlib \
    py-quaternion \
    opencv-python \
    torchmetrics \
    pandas \
    numba \
    imageio

# コンテナ起動時に作業ディレクトリにいるように設定
WORKDIR /workspace/VADv2
```

-----

## 2\. docker-compose.ymlの作成

次に、`Dockerfile` と同じディレクトリに `docker-compose.yml` という名前で以下のファイルを作成します。

このファイルは、コンテナのビルド方法、GPUの使用、ソースコードとデータセットのボリュームマウントを定義します。

```yaml
version: '3.8'

services:
  vadv2-dev:
    # サービス名
    container_name: vadv2_container

    # ビルド設定: カレントディレクトリの Dockerfile を使用
    build:
      context: .
    
    # イメージ名（ビルド後にこの名前で保存されます）
    image: vadv2-dev:latest

    # NVIDIA GPU を使用するための設定
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all

    # ボリュームマウント
    volumes:
      # ホストのカレントディレクトリ（ソースコード）をコンテナにマウント
      - ./:/workspace/VADv2
      
      # [重要] ホストのnuPlanデータセットのパスをコンテナにマウント
      # 例: ホストの /data/nuplan をコンテナの /workspace/VADv2/data/nuplan にマウント
      # ご自身のデータセットパスに合わせて変更してください
      - /data/nuplan:/workspace/VADv2/data/nuplan:ro

    # 共有メモリサイズの設定 (PyTorchのデータローダーで必要)
    shm_size: '16g'

    # コンテナの作業ディレクトリ
    working_dir: /workspace/VADv2

    # コンテナを起動し続けるためのコマンド
    command: sleep infinity
```

**【重要】パスの変更:**
`docker-compose.yml` 内の `volumes` セクションにある `/data/nuplan` のパスを、ご自身のホストマシン上のnuPlanデータセットが保存されている実際のパスに変更してください。

-----

## 3\. 環境構築手順書

以下の手順で、VADv2の学習環境を構築・起動します。

### ⚙️ ステップ1: 前提条件の確認

以下のツールがUbuntuインスタンスにインストールされていることを確認してください。

1.  **NVIDIA GPUドライバ**
2.  **Docker**
3.  **Docker Compose** (Docker Engine v20.10.0以降は `docker compose` コマンドとして統合されています)
4.  **NVIDIA Container Toolkit** (DockerがGPUを認識するために必須)

*NVIDIA Container Toolkitのセットアップ確認:*

```bash
docker run --rm --gpus all nvidia/cuda:11.3.1-base-ubuntu20.04 nvidia-smi
```

上記コマンドでGPU情報が表示されればOKです。

### 📥 ステップ2: VADv2リポジトリのクローン

開発用ディレクトリに `priest-yang/VADv2` リポジトリをクローンします。

```bash
git clone https://github.com/priest-yang/VADv2.git
cd VADv2
```

### 📝 ステップ3: DockerfileとComposeファイルの配置

`cd VADv2` で移動したディレクトリ（`VADv2`リポジトリのルート）に、上記で作成した `Dockerfile` と `docker-compose.yml` の2つのファイルを保存します。

### 🛠️ ステップ4: Dockerイメージのビルド

`docker-compose.yml` があるディレクトリで、以下のコマンドを実行してDockerイメージをビルドします。
（依存ライブラリのダウンロードとインストールのため、時間がかかります）

```bash
docker compose build
```

### 🚀 ステップ5: Dockerコンテナの起動

ビルドが完了したら、コンテナをバックグラウンドで起動します。

```bash
docker compose up -d
```

*起動の確認:*

```bash
docker compose ps
```

`vadv2_container` が `running` (or `up`) 状態になっていれば成功です。

### 💻 ステップ6: コンテナへの接続とセットアップ

起動したコンテナ（開発環境）に接続します。

```bash
docker compose exec vadv2-dev bash
```

コンテナ内に入ったら、VADv2リポジトリ自体をPython環境にインストールします。

```bash
# (コンテナ内での操作)
# /workspace/VADv2 ディレクトリにいることを確認

# VADv2のセットアップ
pip install -v -e .
```

これで、GPUが利用可能なVADv2の学習環境が整いました。コンテナ内で学習スクリプト（`tools/train.py` など）を実行できます。

-----

この環境構築に関して、さらにサポートが必要なことはありますか？ 例えば、学習データの準備方法や特定のスクリプトの実行方法についてお手伝いしましょうか？
