AWS EC2の`g4dn.xlarge`インスタンス（Ubuntu）で、GPUを利用可能なPyTorch + Jupyter Notebook環境を`docker compose`で構築する手順書を作成します。

-----

## 🚀 EC2ホストOSのセットアップ

まず、EC2インスタンス（Ubuntu）に接続し、DockerがGPUを認識できるようにするためのコンポーネントをインストールします。

### 1\. NVIDIAドライバのインストール

`g4dn.xlarge`はNVIDIA T4 GPUを搭載しています。Ubuntuの標準リポジトリから推奨ドライバをインストールするのが簡単です。

```bash
# パッケージリストを更新
sudo apt update

# NVIDIAドライバをインストール（バージョンは適宜変更される可能性があります）
sudo apt install nvidia-driver-535 -y
# インストールが完了したら、OSを再起動します
sudo reboot
```

再起動後、再接続して`nvidia-smi`コマンドを実行し、GPUが正しく認識されていることを確認します。

```bash
nvidia-smi
```

### 2\. Docker Engineのインストール

公式のDockerリポジトリから最新のDockerをインストールします。

```bash
# 必要なパッケージをインストール
sudo apt install -y ca-certificates curl gnupg lsb-release

# Dockerの公式GPGキーを追加
sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg

# Dockerリポジトリを追加
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Docker Engineをインストール
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin

# docker compose up コマンドが使えることを確認
docker compose version
```

### 3\. NVIDIA Container Toolkitのインストール

DockerコンテナからホストのGPUを利用できるようにするための「NVIDIA Container Toolkit」をインストールします。

```bash
# GPGキーとリポジトリを追加
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# パッケージをインストール
sudo apt update
sudo apt install -y nvidia-container-toolkit

# DockerデーモンがGPUを認識できるように設定
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# （任意）sudoなしでDockerコマンドを実行できるようにする
sudo usermod -aG docker $USER
# グループ変更を反映するために、一度ログアウトして再接続してください
```

-----

## 🐳 Docker環境の定義 (docker-compose.yml)

次に、`docker compose`で起動するための設定ファイルを作成します。

### 1\. プロジェクトディレクトリの作成

作業用のディレクトリを作成し、そこに移動します。

```bash
mkdir my-gpu-jupyter
cd my-gpu-jupyter
```

### 2\. `docker-compose.yml` の作成

`my-gpu-jupyter`ディレクトリ内に、`docker-compose.yml`という名前のファイルを以下の内容で作成します。

```bash
nano docker-compose.yml
```

**▼ `docker-compose.yml` の内容**

```yaml
version: '3.8'

services:
  jupyter-gpu:
    # PyTorch公式のGPU(CUDA)対応イメージを使用
    image: pytorch/pytorch:latest
    
    container_name: pytorch_jupyter
    
    # GPUをコンテナに割り当てるための設定
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              capabilities: [gpu]
              count: all # 利用可能な全てのGPUを割り当てる
              
    # ホストのポート8888をコンテナのポート8888にマッピング
    ports:
      - "8888:8888"
      
    # ノートブックファイルを永続化するため、ホストの./notebooksディレクトリをマウント
    volumes:
      - ./notebooks:/workspace/notebooks
      
    # コンテナ起動時にJupyter Notebookを起動するコマンド
    # トークン認証を使用（コンソールに表示される）
    command: jupyter notebook --ip=0.0.0.0 --port=8888 --allow-root --notebook-dir=/workspace/notebooks

    # データを/workspace/notebooksに配置するため
    working_dir: /workspace/notebooks
```

### 3\. ノートブック保存用ディレクトリの作成

`docker-compose.yml`で定義した通り、ホスト側に`notebooks`ディレクトリを作成します。

```bash
mkdir notebooks
```

-----

## 🏃‍♂️ 起動とアクセス

### 1\. Docker Composeの起動

`docker-compose.yml`があるディレクトリで、以下のコマンドを実行します。

```bash
# -d オプションでバックグラウンド起動
docker compose up -d
```

### 2\. Jupyterへのアクセス

コンテナのログを確認し、Jupyter NotebookにアクセスするためのURL（トークン付き）を取得します。

```bash
# ログを表示
docker compose logs
```

ログの中に、以下のようなURLが表示されます。

```
...
pytorch_jupyter  |     To access the notebook, open this file in a browser:
pytorch_jupyter  |         file:///root/.local/share/jupyter/runtime/nbserver-1-open.html
pytorch_jupyter  |     Or copy and paste one of these URLs:
pytorch_jupyter  |         http://localhost:8888/?token=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
pytorch_jupyter  |      or http://127.0.0.1:8888/?token=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
...
```

### 3\. 【重要】AWSセキュリティグループの設定

EC2インスタンスにアタッチされている**セキュリティグループ**で、**TCPポート 8888** へのインバウンド通信を許可する必要があります。
（セキュリティのため、許可するIPアドレスはご自身のIPアドレスに限定することをお勧めします）

### 4\. ブラウザでアクセス

セキュリティグループを設定したら、ブラウザを開き、以下のアドレスにアクセスします。

`http://<EC2のパブリックIPアドレス>:8888/`

ログに表示されたトークン（`token=...`の部分）を入力してログインします。

-----

## ✅ GPUの動作確認

Jupyter Notebookで新しいノートブック（Python 3）を作成し、以下のコードを実行してPyTorchがGPUを認識しているか確認します。

```python
import torch

# GPUが利用可能か確認
print(f"GPU Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    # 利用可能なGPUの数
    print(f"GPU Count: {torch.cuda.device_count()}")
    
    # 現在のGPUデバイスの名前 (g4dn.xlargeなら 'Tesla T4' などと表示される)
    print(f"Device Name: {torch.cuda.get_device_name(0)}")
    
    # 簡単なテンソル演算をGPUで実行
    x = torch.rand(5, 3).to('cuda')
    print(f"Tensor on GPU: \n{x}")
else:
    print("GPU is not available. Check your setup.")
```

**`GPU Available: True`** と **`Device Name: Tesla T4`** (または同等のGPU名) が表示されれば、セットアップは成功です。

この環境で作業を続けたい場合は、`docker compose down` を実行してコンテナを停止・削除することができます。ノートブックの作業内容はホストの `notebooks` ディレクトリに保存されています。
