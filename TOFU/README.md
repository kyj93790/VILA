# TOFU Experiments

- The majority of the code has been adopted from original repository of the TOFU dataset (https://github.com/locuslab/tofu).

### 1. Create conda environment and install requirements
```
conda create -n tofu python=3.10
conda activate tofu
conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
```

### 2. Finetune your models
```
master_port=18765
split=full # other options are retain90, retain95, retain99
model=phi # other option is llama2-7b
lr=2e-5 # 1e-4 for llama2-7b

CUDA_VISIBLE_DEVICES=0,1 torchrun \
--nproc_per_node=2 \
--master_port=$master_port \
finetune.py \
--config-name=finetune.yaml \
split=${split} \
batch_size=4 \
gradient_accumulation_steps=4 \
model_family=${model} \
lr=${lr}
```

### 3. Extract the importance map
```
master_port=18765
split=forget01 # other options are forget05, forget10
model=phi # other option is llama2-7b
lr=2e-5 # 1e-4 for llama2-7b

CUDA_VISIBLE_DEVICES=0,1 torchrun \
--nproc_per_node=2 \
--master_port=$master_port \
vila.py \
--config-name=forget.yaml \
split=${split} \
batch_size=4 \
gradient_accumulation_steps=4 \
model_family=${model} \
lr=${lr}
```

### 4. Forget models
```
model=llama2-7b # other option is phi, llama2-7b
lr=1e-4 # 1e-4 for llama2-7b 2e-5 for phi
RANK=8
model_path=../tofu/locuslab/tofu_ft_llama2-7b
batch_size=4
gradient_accumulation_steps=4
gpu=0,1
master_port=18811

METHOD=gd # other options are ihl, KL, npo etc.
lcoef=1 # loss coefficient (lambda)
beta=0.1
i_type=fila # vila

split=forget01
save_dir=/workspace/checkpoint/${model}_${METHOD}_${split}
importance_file=/workspace/importances/importances_${model}_${split}.pt # path of importance_file
CUDA_VISIBLE_DEVICES=${gpu} torchrun \
--nproc_per_node=2 \
--master_port=$master_port \
forget.py \
--config-name=forget.yaml \
split=${split} \
LoRA.r=${RANK} \
LoRA.alpha=$(( $RANK * 2 )) \
LoRA.dropout=0 \
batch_size=${batch_size} \
gradient_accumulation_steps=${gradient_accumulation_steps} \
model_family=${model} \
forget_loss=${METHOD} \
lr=${lr} \
save_dir=${save_dir} \
model_path=${model_path} \
i_type=${i_type} \
importance_file=${importance_file} \
lcoef=${lcoef} \
beta=${beta}
```

## 5. Evaluate models
```
CUDA_VISIBLE_DEVICES=0 torchrun \
--nproc_per_node=1 \
--master_port=$master_port \
evaluate_util.py \
model_family=$model_family split=$split \
model_path=$model_path
```

### 6. Aggregate results
```
python aggregate_eval_stat.py \
retain_result=${path_to_aggregated_retain_result} \
ckpt_result=${path_to_aggregated_retain_result} \
method_name=${method_name} save_file=${save_filename}
```
