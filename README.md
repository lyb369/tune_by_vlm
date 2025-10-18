# SPARSE-SIM: 基于稀疏去卷积的超分辨率显微镜图像恢复系统

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 项目简介

SPARSE-SIM 是一个基于稀疏去卷积算法的超分辨率显微镜图像恢复系统，专门用于处理SIM（Structured Illumination Microscopy，结构光照显微镜）数据。该系统集成了VLM驱动的参数优化功能，能够自动调整算法参数以获得最佳的图像恢复效果。

### 主要特性

- 🔬 **稀疏去卷积算法**：基于Nature Biotechnology论文的高精度图像恢复
- 🤖 **AI参数优化**：集成Qwen大语言模型进行智能参数调优
- 📊 **多维度评估**：支持FRC、PSNR、SSIM等多种图像质量评估指标
- 🖼️ **图像处理工具**：包含MRC文件转换、区域选择等实用工具
- ⚡ **GPU加速**：支持CUDA加速，显著提升处理速度
- 🔧 **灵活配置**：支持多种参数配置和运行模式

## 系统架构

```
SPARSE-SIM/
├── sparse_sim.py              # 核心算法实现
├── run_sparse_sim.py          # 主运行脚本
├── qwen_helper.py             # AI参数优化模块
├── image_region_selector.py   # 图像区域选择工具
├── mrc_to_png_converter.py   # MRC文件转换工具
├── run.sh                     # 批处理脚本
└── sparse-deconv-py-main/    # 稀疏去卷积算法库
    ├── sparse_recon/         # 核心算法实现
    ├── demo.py               # 示例代码
    └── README.md             # 算法库说明
```

## 安装要求

### 系统要求
- Python 3.7+
- CUDA 11.5+ (可选，用于GPU加速)
- 8GB+ RAM (推荐16GB+)

### 依赖包
```bash
pip install numpy scipy pillow opencv-python matplotlib
pip install transformers torch torchvision
pip install cupy-cuda11x  # 如果使用CUDA 11.x
pip install scikit-image
```

### AI模型要求
- Qwen2.5-7B-Instruct 或 Qwen2.5-VL-7B-Instruct
- 模型路径可通过环境变量 `QWEN_MODEL_PATH` 设置

## 快速开始

### 1. 基本使用

```bash
# 使用默认参数运行
python run_sparse_sim.py --gt_path ground_truth.png --blur_path blurred_image.png

# 使用AI参数优化
python run_sparse_sim.py \
    --gt_path ground_truth.png \
    --blur_path blurred_image.png \
    --rounds 5 \
    --model_path /path/to/qwen/model
```

## 核心功能详解

### 1. 稀疏去卷积算法 (`sparse_sim.py`)

核心算法实现了基于稀疏性的图像去模糊，主要功能包括：

- **高斯核估计**：自动估计点扩散函数(PSF)参数
- **FISTA优化**：使用快速迭代收缩阈值算法进行稀疏优化
- **双调和正则化**：结合数据保真度和稀疏性约束
- **FRC评估**：计算傅里叶环相关度评估图像质量


### 2. AI参数优化 (`qwen_helper.py`)

集成Qwen大语言模型进行智能参数调优：

- **历史学习**：分析历史参数和结果，学习最优参数组合
- **自适应调整**：根据图像质量指标动态调整参数
- **多目标优化**：平衡去噪效果和细节保持


### 3. 图像处理工具

#### MRC文件转换器 (`mrc_to_png_converter.py`)
```bash
# 转换单个目录
python mrc_to_png_converter.py /path/to/mrc/files

# 递归处理子目录
python mrc_to_png_converter.py /path/to/mrc/files -o /output/dir
```

#### 图像区域选择器 (`image_region_selector.py`)
```bash
# 交互式选择图像区域
python image_region_selector.py /path/to/image/directory
```

## 参数说明

### 核心参数

| 参数 | 说明 | 默认值 | 范围 |
|------|------|--------|------|
| `fidelity` | 数据保真度权重 | 150.0 | 10.0-500.0 |
| `sparsity` | 稀疏性权重 | 10.0 | 0.1-50.0 |

### 运行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--rounds` | 优化轮数 | 5 |
| `--fidelity_init` | 初始保真度 | 150.0 |
| `--sparsity_init` | 初始稀疏性 | 10.0 |
| `--use_demo_params` | 使用演示参数 | False |
| `--test_mode` | 测试模式 | False |

## 输出结果

### 文件输出
- `outputs_deconv/iter_XXX.png`：每轮迭代的恢复结果
- `memory_deconv/iter_XXX.json`：详细的参数和指标记录
- `kernel_*.png`：PSF核可视化
- `kernel_effect_grid.png`：核效果对比图

### 评估指标
- **FRC_AUC**：傅里叶环相关度曲线下面积
- **R-FRC_AUC**：滚动FRC平均AUC
- **halfbit_res**：半比特分辨率
- **mean_grad_mag**：平均梯度幅值


## 内存优化
- 对于盲反演问题，建议使用`--use_demo_params`模式，根据显微镜或成像系统本身的性质制定数值
- 可以通过调整`sparse_iter`参数平衡速度和质量

## 相关链接

- [原始MATLAB实现](https://github.com/WeisongZhao/Sparse-SIM)
- [Nature Biotechnology论文](https://doi.org/10.1038/s41587-021-01092-2)
- [Qwen模型](https://huggingface.co/Qwen)


