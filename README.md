# Deep Learning Assignment 2

## Part A: Chinese-to-English Translation with Seq2Seq

### 1. 环境配置 (Environment Setup)
```bash
# Create conda environment (Python 3.10)
conda create -n seq2seq python=3.10
conda activate seq2seq

# Install dependencies
pip install torch==2.5.1 transformers==4.44.2 matplotlib pandas
```


## 2. Data Preparation
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
python train.py --batch_size 32 --hidden_size 256 --epochs 35

# 生成测试集翻译结果
python evaluate.py --model_checkpoint best_model.pt --output results.csv
```

## 4. 关键文件说明 (Key Files)
```bash
文件	功能
model.py	Encoder/Decoder + Attention 实现
utils.py	数据加载与分词工具
```


## Part B: LoRA Fine-tuning for Legal QA
### 1. 系统要求 (System Requirements)
GPU: ≥24GB显存（如A100/A40）

CUDA: 12.4+

### 2. 快速开始 (Quick Start)
```bash
# 下载模型（需提前安装git-lfs）
git clone https://huggingface.co/Qwen/Qwen2.5-7B-Instruct

# 转换数据集格式
python convert_to_alpaca.py --input DISC-Law-SFT.json --output data_alpaca.json

# 启动LoRA微调
CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
    --model_name_or_path Qwen2.5-7B-Instruct \
    --dataset data_alpaca.json \
    --lora_rank 8 \
    --quantization_bit 4
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







