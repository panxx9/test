# Deep Learning Assignment 2

## Part A: Chinese-to-English Translation with Seq2Seq

### 1. 环境配置 (Environment Setup)
```bash
# 基于kaggle
# Create conda environment (Python 3.10)
conda create -n seq2seq python=3.10
conda activate seq2seq

# Install dependencies
pip install torch==2.5.1 transformers==4.44.2 matplotlib pandas
```


## 2. 数据准备（Data Preparation）
* Dataset Structure:
```bash
data/
├── cmn.txt          # Raw Chinese-English corpus
└── splits/
    ├── train_pairs.pkl
    └── test_pairs.pkl
```


## 3. 训练与评估 (Training & Evaluation)
```bash
# 训练模型（默认参数）
# 即运行part_a_code_on_kaggle.ipynb 此处类比说明
python train_code.py --batch_size 32 --hidden_size 256 --epochs 50
```



## Part B: LoRA Fine-tuning for Legal QA
### 1. 系统要求 (System Requirements)
GPU: ≥24GB显存（如A100/A40）

CUDA: 12.4+

### 2. 快速开始 (Quick Start)
```bash
# 基于autodl
# 配置环境
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory

pip install --no-deps -e
pip install -e '.[torch,metrics]'

# 此时可能会遇到tranformer版本冲突
pip install "transformers>=4.45.0,<=4.51.3,!=4.46.0,!=4.46.1,!=4.46.2,!=4.46.3,!=4.47.0,!=4.47.1,!=4.48.0"

# 下载模型（需提前安装git-lfs）
git clone https://huggingface.co/Qwen/Qwen2.5-7B-Instruct
# 由于在autodl访问huggingface受限，从魔塔社区下载模型
pip install modelscope
from modelscope import snapshot_download, AutoTokenizer

# 加载模型 运行partb.ipynb文件 此处类比说明
python partb.py

# 转换数据集格式
python data_transformation.py --input D/DISC-Law-SFT-Pair-QA-released.jsonl --output train_data_law.json

# 在LLAMA-FACTORY提供的可视化界面启动LoRA微调
CUDA_VISIBLE_DAVICES=0 GRADIO_SHARE=1 llamafactory-cli webui 
```

### 3. 配置示例 (Configuration)
```bash
# config/lora.yaml
lora:
  r: 8              # LoRA秩
  target_modules: ["q_proj", "v_proj"]
```

### 4. 常见问题 (FAQ)
#### Q: 出现CUDA内存不足错误？
```bash
python
model.gradient_checkpointing_enable()  # 减少显存占用
```







