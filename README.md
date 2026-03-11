## X-Trainer VLA 比赛选手指南（Docker 版）

> 目标：按文档从零配置环境 → 构建镜像 → 跑通 `dobot-challenge` 分支的 X‑Trainer 仿真（带可视化）。  

本文用于指导选手如何使用docker部署x-trainer环境，并指导选手最终提交什么样的作品，建议部署前学习docker基础知识，且能自行解决网络问题（docker国内镜像拉取困难）。

竞赛仓库地址：https://github.com/embodied-dobot/x-trainer/tree/dobot-challenge

竞赛官网：https://challenge.dobot-robots.com/

钉钉群：群号170895003131，加群请备注：比赛

你可以自由发挥，但需要保证最终提供的模型推理端口兼容性。



本项目基于[![Isaac Sim 4.5](https://camo.githubusercontent.com/3390157b79322051b43bf99f9d2c616cb3fedc2483a1d98aa1b8ccdb68991eb4/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f497361616325323053696d2d342e352d3061383466663f7374796c653d666f722d7468652d6261646765266c6f676f3d6e7669646961)](https://camo.githubusercontent.com/3390157b79322051b43bf99f9d2c616cb3fedc2483a1d98aa1b8ccdb68991eb4/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f497361616325323053696d2d342e352d3061383466663f7374796c653d666f722d7468652d6261646765266c6f676f3d6e7669646961) [![Isaac Lab 0.47.1](https://camo.githubusercontent.com/acd77da0ec29610457c1cb5fe9ce99b8c65ee1ff18e2183f8ab31ae5183b6163/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f49736161632532304c61622d302e34372e312d3334633735393f7374796c653d666f722d7468652d6261646765266c6f676f3d6e7669646961)](https://camo.githubusercontent.com/acd77da0ec29610457c1cb5fe9ce99b8c65ee1ff18e2183f8ab31ae5183b6163/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f49736161632532304c61622d302e34372e312d3334633735393f7374796c653d666f722d7468652d6261646765266c6f676f3d6e7669646961) [![Python 3.10+](https://camo.githubusercontent.com/9e03d11b200b8f6dd8dfdd67ea52da634596d8991006bbcedbbfd3b48401f9e0/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f507974686f6e2d332e31302532422d6666393530303f7374796c653d666f722d7468652d6261646765266c6f676f3d707974686f6e)](https://camo.githubusercontent.com/9e03d11b200b8f6dd8dfdd67ea52da634596d8991006bbcedbbfd3b48401f9e0/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f507974686f6e2d332e31302532422d6666393530303f7374796c653d666f722d7468652d6261646765266c6f676f3d707974686f6e)

https://github.com/huggingface/lerobot

https://github.com/huggingface/transformers/tree/fix/lerobot_openpi



## 1. 宿主机环境准备（只做一次）

### 1.1 系统与硬件要求

- **系统**：Ubuntu 20.04 / 22.04（推荐 22.04）
- **GPU**：NVIDIA 显卡（如 RTX 3060 / 4080 等），驱动安装正常
- **显卡驱动**：建议 535+，能正常运行 `nvidia-smi`

在终端执行：

```bash
nvidia-smi
```

如果能看到 GPU 型号与显存信息，则说明驱动正常；否则请先安装/修复显卡驱动再继续。

### 1.2 安装 Docker（若已安装可跳过）

```bash
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
```

将当前用户加入 `docker` 组（可选但推荐），这样后面不用每次都写 `sudo`：

```bash
sudo groupadd docker 2>/dev/null || true
sudo usermod -aG docker $USER
newgrp docker   # 或重新登录终端
```

验证 Docker：

```bash
docker run hello-world
```

看到 “Hello from Docker!” 说明安装成功。

### 1.3 安装 NVIDIA Container Toolkit（GPU 给容器用,若已安装请跳过）

参考官方流程简化版：

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

测试 GPU 是否能在容器内使用：

```bash
docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi
```

若能正常打印出 GPU 信息，说明 GPU 配置无问题。

---

## 2. 获取Docker 仓库与 X‑Trainer 代码（宿主机上操作）

> 当前这个 Docker 仓库（包含 `final-dockerfile` 等文件），这里统一称为 **x-trainer-docker 仓库**。

### 2.1 克隆 Docker 仓库

在宿主机任意目录执行（示例用 `~/proj`，可自行调整路径）：

```bash
mkdir -p ~/proj
cd ~/proj

# 将本项目拉取下来
git clone <本项目地址> x-trainer-docker
cd x-trainer-docker
```

之后你的目录结构大致如下：

- `~/proj/x-trainer-docker/final-dockerfile`
- `~/proj/x-trainer-docker/xtrainer-requirements.txt`
- `~/proj/x-trainer-docker/code/lerobot`
- `~/proj/x-trainer-docker/code/transformers`

> **注意**：选手**不需要**在宿主机单独安装 Isaac Sim / Isaac Lab，这些都已经包含在 Docker 镜像里。

### 2.2 在宿主机准备“选手工作区”（挂载用，保证代码不丢）

强烈建议你在宿主机准备一个**持久化目录**，所有自己的代码、模型、资产都放在这里，然后通过 `-v` 挂载进容器。  
这样即使容器删掉 / 重启，代码也不会丢。

在宿主机执行：

```bash
mkdir -p ~/xtrainer_player
cd ~/xtrainer_player

# 拉取比赛项目（只需一次）
git clone https://github.com/embodied-dobot/x-trainer.git
cd x-trainer
git checkout dobot-challenge
```

*从**钉钉群**下载比赛资产压缩包（ `xtrainer_assets.zip`），放到 `~/xtrainer_player/x-trainer` 下，然后解压替换原有的assets文件夹。

> 之后你写的自定义脚本、训练好的模型权重等，也都可以放在 `~/xtrainer_player` 下面，保证不会因为容器重启而丢失。

---

## 3. 构建基础镜像 `xtrainer-final:latest`

### 3.1登录 NGC 拉取 Isaac Sim 基础镜像

基础镜像是 `nvcr.io/nvidia/isaac-sim:4.5.0`，首次使用可能需要登录 NVIDIA NGC 才能拉取，但是我们每次拉取都没有登陆过，所以你也可以先跳过这一步，如果构建有问题再尝试：

1. 打开 NGC：`https://ngc.nvidia.com/setup/api-key`，创建 API Key。  
2. 在宿主机执行：

```bash
docker login nvcr.io
```

按提示输入：

- 用户名：`$oauthtoken`
- 密码：刚才生成的 API Key

如果你之前已经能从 `nvcr.io` 拉镜像，这一步可以跳过。

### 3.2 在 Docker 仓库目录构建镜像

确保当前目录在 `x-trainer-docker`：

```bash
cd ~/proj/x-trainer-docker
docker build -f final-dockerfile -t xtrainer-final:latest .
```

- `-f final-dockerfile`：使用仓库里的 `final-dockerfile`  
- `-t xtrainer-final:latest`：给镜像命名为 `xtrainer-final:latest`

构建过程可能较长（几十分钟），且建议为终端配置代理，中间可能看到依赖版本报错，没有关系，只要最后看到 `Successfully tagged xtrainer-final:latest` 就说明构建成功。如果构建有问题请先使用ai工具尝试解决。

---

## 4. 启动带 GUI 的容器（X11 可视化，挂载选手工作区）

### 4.1 宿主机允许本地容器访问 X11显示

在宿主机终端执行一次（每次重启后可再执行一次）：

```bash
xhost +local:
```

### 4.2 启动容器并把显示 + 工作区都转进去

```bash
cd ~/proj/x-trainer-docker

docker run -it --gpus all \
  -e "ACCEPT_EULA=Y" \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v ~/xtrainer_player:/workspace/xtrainer_player \
  --entrypoint /bin/bash \
  xtrainer-final:latest
```

进入容器后，你会看到提示符类似：

```bash
root@xxxxxxxxxxxx:/workspace#
```

后续没有特别说明时，命令默认都是**在容器里**执行。

---

## 5. 在容器内准备 X‑Trainer 项目

### 5.1 进入挂载的 X‑Trainer 目录

由于我们已经在宿主机的 `~/xtrainer_player` 中准备好了代码和资产，并挂载到了容器的 `/workspace/xtrainer_player`，在容器里只需要：

```bash
cd /workspace/xtrainer_player/x-trainer
```

这里就是你在宿主机克隆并解压资产后的同一个目录。

### 5.2 安装 LeIsaac（leisaac 包）

在容器内（`/workspace/x-trainer`）执行：

```bash
cd /workspace/xtrainer_player/x-trainer
/isaac-sim/python.sh -m pip install -e source/leisaac
```

验证安装：

```bash
/isaac-sim/python.sh -c "import leisaac; print('安装成功！')"
```

若看到 `安装成功！` 就可以继续。

---

## 6. 跑通一个键盘遥操作 Demo（task1）

现在开始，你只需要在容器里复制下面一条命令，就可以看到仿真窗口和双臂机器人。

在容器内执行：

```bash
cd /workspace/xtrainer_player/x-trainer
/isaac-sim/python.sh scripts/environments/teleoperation/teleop_se3_agent.py \
  --task=task1 \
  --teleop_device=bi_keyboard \
  --num_envs=1 \
  --device=cuda \
  --enable_cameras \
  --multi_view
```



如果一切正常，你会在宿主机桌面看到 Isaac Sim 窗口，里面是 task1 场景。

键盘默认控制方式（简要）：

- 左臂：`Q W E A S D`
- 右臂：`U I O J K L`
- 夹爪：`G`（左）、`H`（右）
- 系统控制：`B` 开始、`R` 失败重置、`N` 成功重置

更详细说明可参考 X‑Trainer 竞赛的官方 README：[`x-trainer@dobot-challenge`](https://github.com/embodied-dobot/x-trainer/tree/dobot-challenge)。

---

## 7. 后续你可以做什么？

在赛题要求范围内，你可以：

- 在宿主机 `~/xtrainer_player` 下开发自己的代码（例如 `~/xtrainer_player/your_method`），容器内对应为 `/workspace/xtrainer_player/your_method`。  
- 使用本指南中的命令，在容器内用 `/isaac-sim/python.sh` 启动 X‑Trainer 场景、自定义推理脚本等，对你的 VLA 模型做联调。  
- 按下一节说明，把你训练好的模型权重和推理代码，最终打包成一个**自包含**的 Docker 镜像提交给我们。

---

## 8. 打包并提交参赛镜像（重点）

> 本节说明如何在开发完成后，**打包一个只用来推理评分的镜像**，方便我们在评测机上直接运行并通过端口访问你的模型。

### 8.1 打包原则

- **基底镜像**：必须基于我们提供的 `xtrainer-final:latest`。  
- **自包含**：镜像内必须包含你推理所需的**所有代码与模型权重**，不要依赖挂载卷（我们评分时不会再加 `-v`）。  
- **自动启动服务**：容器一启动，就会在 `0.0.0.0:5555` 上启动你的推理服务（gRPC server），不需要人工进容器敲命令。  
- **协议**：使用 LeRobot 的异步推理协议（`lerobot.async_inference.policy_server`），这样可以直接用 X‑Trainer 自带的客户端脚本进行评分。

### 8.2 在宿主机准备打包目录

在宿主机执行（示例）：

```bash
mkdir -p ~/xtrainer_player/submit_v1
cd ~/xtrainer_player/submit_v1
```

假设你已经在 `~/xtrainer_player` 里训练好模型，并把权重保存为：

- `~/xtrainer_player/checkpoints/pretrained_model`（可按自己习惯调整路径）

然后将需要的内容复制到打包目录（也可以在 Dockerfile 里从挂载路径 COPY，这里给一个清晰示例）：

个人建议使用vscode的Dev Containers管理docker文件，清晰明了

```bash
cp -r ~/xtrainer_player/checkpoints ./checkpoints
```

### 8.3 编写 `run_policy_server.sh`

在 `~/xtrainer_player/submit_v1` 里创建脚本：

```bash
cat > run_policy_server.sh << 'EOF'
#!/bin/bash

set -e

# 在容器内启动 LeRobot 的异步推理服务
exec /isaac-sim/python.sh -m lerobot.async_inference.policy_server \
  --host=0.0.0.0 \
  --port=5555 \
  --fps=30 \
  --policy_checkpoint_path=/workspace/checkpoints/pretrained_model \
  --policy_type=xtrainer_act
EOF

chmod +x run_policy_server.sh
```

> 注意：  
> - `--policy_checkpoint_path` 请改成你实际权重文件的路径；  
> - `--policy_type` 也可以按照你自己的策略类型调整，只要和 X‑Trainer 评分脚本中配置的一致即可。  
> - 这里使用 `/isaac-sim/python.sh`，保证与训练/联调时使用的环境一致。

### 8.4 编写 Dockerfile（选手提交镜像）

仍然在 `~/xtrainer_player/submit_v1` 下，创建 `Dockerfile`：

```bash
cat > Dockerfile << 'EOF'
FROM xtrainer-final:latest

WORKDIR /workspace

# 拷贝模型权重和推理启动脚本到镜像中
COPY checkpoints /workspace/checkpoints
COPY run_policy_server.sh /workspace/run_policy_server.sh

RUN chmod +x /workspace/run_policy_server.sh

# 容器一启动就自动运行推理服务
ENTRYPOINT ["/workspace/run_policy_server.sh"]
EOF
```

### 8.5 构建并本地测试你的推理镜像

在宿主机执行：

```bash
cd ~/xtrainer_player/submit_v1

# 构建你的参赛镜像，例子中用 your_team:v1，请按我们规范命名
docker build -t your_team:v1 .
```

本地自测（推荐在同一台机器上直接跑一遍评分流程）：

```bash
# 1）启动推理容器（注意：这里不再挂载代码卷）
docker run -d --name xtrainer_test_run \
  -e "ACCEPT_EULA=Y" \
  --gpus all \
  -p 5555:5555 \
  your_team:v1

# 2）在本机（或另一个终端）用评分脚本连接5555端口 localhost:5555
# （根据我们提供的评分脚本命令为准，这里给一个示例）
cd ~/xtrainer_player/x-trainer
/isaac-sim/python.sh scripts/evaluation/policy_scoring.py \
  --task=task1 \
  --eval_rounds=10 \
  --policy_type=xtrainer_act \
  --policy_host=localhost \
  --policy_port=5555 \
  --device=cuda \
  --enable_cameras \
  --policy_checkpoint_path="./checkpoints/last/pretrained_model"

```

> 我们在正式评分时会先加载你上传的镜像文件（`docker load -i your_team_v1.tar`），再执行 `docker run --gpus all -p 5555:5555 your_team:v1`，然后用统一的评分脚本连 `localhost:5555`，**不会再额外挂载任何卷**。

### 8.6 导出镜像文件并上传至提交官网

我们要求选手在**比赛提交官网**直接上传打包好的镜像文件，请按以下步骤操作。

**1）将镜像导出为单个文件（.tar）**

在宿主机执行（镜像名与上一步构建时一致）：

```bash
cd ~/xtrainer_player/submit_v1

# 将镜像导出为 tar 文件，请把 your_team:v1 换成你实际的镜像名
docker save -o your_team_v1.tar your_team:v1
```

生成的文件 `your_team_v1.tar` 即为完整镜像包（体积较大，通常为数 GB）。

**2）按需压缩（若官网要求或便于上传）**

若希望减小上传体积，可压缩后再上传，例如：

```bash
gzip -k your_team_v1.tar
# 得到 your_team_v1.tar.gz
```

具体是否需压缩、是否接受 `.tar.gz`，以**后续说明**为准。

**3）上传至比赛提交官网**

- 打开我们提供的**比赛提交官网**。
- 登录你的参赛账号。
- 在「提交镜像」或「上传镜像」页面，选择刚生成的 **`your_team_v1.tar`**（或压缩后的 `your_team_v1.tar.gz`）进行上传。
- 上传完成后，按页面提示确认提交即可。

> 比赛提交详见后续提交说明

---

## 9. 常见问题（FAQ）

### 9.1 进入容器后 `nvidia-smi` 报错 / 找不到 GPU

- 检查宿主机是否已安装 NVIDIA 驱动，并能正常运行 `nvidia-smi`。  
- 检查 `docker run` 是否带了 `--gpus all`。  
- 确认 NVIDIA Container Toolkit 已安装并执行了 `nvidia-ctk runtime configure --runtime=docker`。

### 9.2 一运行脚本就 Segmentation fault（core dumped）

大部分情况是**没有正确配置 X11 可视化**导致的：

- 确认宿主机已执行：`xhost +local:`  
- 确认 `docker run` 时带了：
  - `-e DISPLAY=$DISPLAY`
  - `-v /tmp/.X11-unix:/tmp/.X11-unix`
- 确认当前是真正在 Linux 桌面环境下（不是纯命令行服务器）。

如果你不需要窗口，而只想跑 headless 模式，可以让我们提供对应参数/脚本。

### 9.3 报错找不到资产 / USD 文件

- 确认已经从 **钉钉群** 下载了比赛资产压缩包，并正确解压到 `/workspace/x-trainer` 下。  
- 解压时建议加 `-o` 参数覆盖原有文件：

```bash
unzip -o xtrainer_assets.zip -d /workspace/x-trainer
```

### 9.4 `No module named 'leisaac'`

在容器内重新执行：

```bash
cd /workspace/x-trainer
/isaac-sim/python.sh -m pip install -e source/leisaac
/isaac-sim/python.sh -c "import leisaac; print('安装成功！')"
```

---

祝比赛顺利！ 🎯
