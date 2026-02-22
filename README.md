# AKI_LLM

## 1. Clone AKI LLM model

​	1）git clone https://github.com/Dr-zouxiaoqin/AKI_LLM.git
​	2）git branch -M main

## 2. AKI model training 

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export CUDA_LAUNCH_BLOCKING=1

export TRANSFORMERS_OFFLINE=1

python -c "import torch; torch.cuda.empty_cache()"

export CUDA_VISIBLE_DEVICES=0,1,2,3

torchrun --nproc_per_node=4  --master_port=30002 llama_qlora_AKI_distributed_model.py >llama_qlora_AKI_distributed_model.log 2>&1 &

## 3. AKI model inference

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export CUDA_LAUNCH_BLOCKING=1

export TRANSFORMERS_OFFLINE=1

python -c "import torch; torch.cuda.empty_cache()"

export CUDA_VISIBLE_DEVICES=0

torchrun --nproc_per_node=1 --master_port=30002 llama_qlora_AKI_distributed_model_inference.py >llama_qlora_AKI_distributed_model_inference.log 2>&1 &