# 行为检测系统 2.0

一个基于YOLOv8和PyQt5的实时行为检测系统，支持摄像头监控和本地视频文件分析。

## 系统要求

- Python 3.11
- 依赖请参考`pyproject.toml`

## 环境配置

### 1. 安装 uv

首先需要安装 uv 包管理器：

#### macos
可以通过`brew`进行安装：
```bash
brew install uv
```


#### Windows
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

#### 验证安装
```bash
uv --version
```

### 2. 克隆项目

```bash
git clone https://github.com/Yi-ran-Nian-Hua/Detection-System2.0.git
cd "Detection System2.0"
```

### 3. 创建虚拟环境并安装依赖

使用 uv 创建虚拟环境并安装所有依赖：

```bash
# 创建虚拟环境并安装依赖
uv sync

# 激活虚拟环境
source .venv/bin/activate  # macOS/Linux
# 或
.venv\Scripts\activate     # Windows
```


## 项目结构

```
Detection System2.0/
├── yolov8pose.py              # 主程序文件
├── pyproject.toml             # 项目配置文件
├── requirements.txt           # 依赖列表
├── uv.lock                   # uv锁定文件
├── yolov8n-pose.pt           # YOLOv8姿态检测模型
├── README.md                 # 项目说明文档
├── test_video_feature.py     # 视频功能测试脚本
├── test_camera_video_separation.py  # 功能分离测试脚本
├── cap.py                    # 摄像头测试脚本
```

## 依赖说明

主要依赖包及其用途：

- **ultralytics**: YOLOv8模型加载和推理
- **PyQt5**: GUI界面框架
- **opencv-python**: 图像处理和视频处理
- **torch**: PyTorch深度学习框架
- **numpy**: 数值计算
- **Pillow**: 图像处理

### 更新依赖

```bash
# 添加新依赖
uv add package-name

# 更新现有依赖
uv update

# 查看依赖树
uv tree
``` 