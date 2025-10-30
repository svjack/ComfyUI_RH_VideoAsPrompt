# ComfyUI Video-As-Prompt 节点

ComfyUI 的自定义节点，集成 Video-As-Prompt 实现运动引导的视频生成。

## ✨ 功能特性

* 🎬 **运动引导生成**：使用参考视频控制生成视频的运动
* 🖼️ **图像转视频**：从图像生成视频，并通过参考视频引导运动
* ⚙️ **内存优化**：INT8量化 + CPU卸载，高效推理
* 🚀 **CogVideoX-5B**：基于强大的CogVideoX-5B模型

## 🔧 节点列表

* **RunningHub VideoAsPrompt Loader**：加载并初始化 Video-As-Prompt 管线
* **RunningHub VideoAsPrompt Sampler(CogVideoX)**：从图像生成带参考运动的视频

## 🚀 快速安装

### 步骤 1：安装节点

```bash
# 进入 ComfyUI 的 custom_nodes 目录
cd ComfyUI/custom_nodes/

# 克隆仓库
git clone https://github.com/HM-RunningHub/ComfyUI_RH_VideoAsPrompt.git

cd ComfyUI_RH_VideoAsPrompt

# 安装依赖
pip install -r requirements.txt
```

### 步骤 2：下载所需模型

下载 CogVideoX-5B 模型并放置在以下目录结构：

```
ComfyUI/models/Video-As-Prompt/
└── CogVideoX-5B/
    ├── vae/
    ├── transformer/
    └── ...
```

可以从 [Video-As-Prompt 数据集](https://huggingface.co/datasets/BianYx/VAP-Data) 下载，或使用预训练的 CogVideoX-5B 模型。

### 步骤 3：重启 ComfyUI

## 📖 使用说明

### 基本工作流

```
[加载图像] → [加载视频] → [RunningHub VideoAsPrompt Loader] → [RunningHub VideoAsPrompt Sampler] → [保存视频]
```

### 生成参数

* **image**：用于视频生成的输入图像
* **ref_video**：用于运动引导的参考视频
* **prompt**：输出视频的文本描述
* **prompt_mot_ref**：参考运动的文本描述
* **height/width**：输出视频尺寸（默认：480x720）
* **num_frames**：生成帧数（默认：49）
* **num_inference_steps**：去噪步数（默认：50）

## 🛠️ 技术要求

* **GPU**：12GB+ 显存（使用 INT8 量化 + CPU 卸载）
* **内存**：建议 16GB+
* **存储**：CogVideoX-5B 模型约 20GB
* **CUDA**：需要 CUDA 以获得最佳性能

## ⚠️ 重要说明

* **模型路径**：模型必须放置在 `ComfyUI/models/Video-As-Prompt/` 目录下
* **内存优化**：默认启用 INT8 量化和 CPU 卸载以提高内存效率
* 首次使用前必须下载所有模型文件

## 🔗 参考链接

* [Video-As-Prompt 项目](https://github.com/bytedance/Video-As-Prompt)
* [Video-As-Prompt 数据集](https://huggingface.co/datasets/BianYx/VAP-Data)
* [ComfyUI](https://github.com/comfyanonymous/ComfyUI)

## 📄 许可证

本项目基于 [Video-As-Prompt](https://github.com/bytedance/Video-As-Prompt) 项目开发。

## ⭐ 引用

如果您觉得本项目有用，请考虑引用原始 Video-As-Prompt 论文。

---

**开发者：[HM-RunningHub](https://github.com/HM-RunningHub)**

