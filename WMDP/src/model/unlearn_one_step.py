import os
import sys
import json

import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainerCallback

sys.path.append("src")
import torch
from peft import  get_peft_model, LoraConfig, AdaLoraConfig, TaskType, PeftModel
from dataset import get_dataset
from metrics import (
    eval_copyright,
    eval_few_shots,
    eval_PII,
    eval_ppl,
    eval_toxic,
    eval_wmdp,
)
from unlearn import get_unlearn_method
import matplotlib.pyplot as plt
import seaborn as sns

def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


class WeightAveragingCallback(TrainerCallback):
    def __init__(self, start_step=100, interval=5):
        self.start_step = start_step
        self.interval = interval
        self.swa_state_dict = None
        self.n = 0

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step >= self.start_step and state.global_step % self.interval == 0:
            model = kwargs["model"]
            current_state_dict = {k: p.clone().detach() for k, p in model.state_dict().items()}
            
            if self.swa_state_dict is None:
                self.swa_state_dict = current_state_dict
                self.n = 1
            else:
                for key in self.swa_state_dict:
                    self.swa_state_dict[key] = (self.swa_state_dict[key] * self.n + current_state_dict[key]) / (self.n + 1)
                self.n += 1

    def on_train_end(self, args, state, control, **kwargs):
        model = kwargs["model"]
        if self.swa_state_dict is not None:
            model.load_state_dict(self.swa_state_dict, strict=True)


class Unlearn:
    def __init__(self, model_name, cache_dir, **kwargs) -> None:
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.unlearn_method = kwargs["unlearn_method"]
        self.batch_size = kwargs["batch_size"]
        self.dataset_names = kwargs["dataset_names"]
        self.dataset_seed = kwargs["dataset_seed"]
        self.forget_ratio = kwargs["forget_ratio"]
        self.self_retain = kwargs["self_retain"]
        self.num_epochs = kwargs["num_epochs"]
        self.num_devices = int(os.environ.get("WORLD_SIZE", 1))
        self.lr = kwargs["lr"]
        self.gradient_accumulation_steps = kwargs["gradient_accumulation_steps"]
        self.weight_decay = kwargs["weight_decay"]
        self.gamma = kwargs.get("gamma", None)
        self.importance_file = kwargs.get("importance_file", None)
        self.std = kwargs.get("std", 0.01)
        self.per_step = 25

        # for sam
        self.beta = kwargs.get("beta", None)
        self.gnr_rho = kwargs.get("gnr_rho", None)
        self.sam_rho= kwargs.get("sam_rho", None)
        self.task_name = kwargs.get("task_name", None)

        self.if_llama = "llama" in self.model_name
        self.resume_path = kwargs.get("resume_path", None)
        self.max_steps = kwargs.get("max_steps", -1)
        self.use_lora = kwargs.get("use_lora", False)
        self.if_wanda = False

        self.swa = kwargs.get("swa", False)

    def init_model(self):
        print(f"Loading the checkpoint from {self.model_name}")

        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            cache_dir=self.cache_dir,
            low_cpu_mem_usage=True,
            device_map="auto",
        )

        if self.use_lora:
            peft_config = LoraConfig(
                r=8, 
                lora_alpha=16, 
                target_modules=find_all_linear_names(model), 
                lora_dropout=0,
                bias="none", 
                task_type="CAUSAL_LM"
            )
            model = get_peft_model(model, peft_config)
            print(model.print_trainable_parameters())

        model.seqlen = model.config.max_position_embeddings
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=False)

        if tokenizer.pad_token_id is None:
            if self.if_llama:
                tokenizer.add_special_tokens({"pad_token": "[pad]"})

            else:
                tokenizer.pad_token = tokenizer.eos_token
                model.config.pad_token_id = model.config.eos_token_id

        self.model = model
        self.model.resize_token_embeddings(len(tokenizer))
        self.tokenizer = tokenizer
        try:
            self.device = model.hf_device_map["lm_head"]
        except:
            self.device = torch.device("cuda:0")

    def init_dataset(self):
        unlearn_dataset, test_dataset, unlearn_collator, test_collator = get_dataset(
            self.dataset_names,
            self.tokenizer,
            self.dataset_seed,
            self.forget_ratio,
            self.self_retain,
            self.if_llama,
            self.unlearn_method
        )
        self.unlearn_dataset = unlearn_dataset
        self.test_dataset = test_dataset
        self.unlearn_collator = unlearn_collator
        self.test_collator = test_collator
        if self.max_steps == -1:
            self.max_steps = int(self.num_epochs * len(unlearn_dataset)) // (
                self.batch_size * self.gradient_accumulation_steps * self.num_devices
            )
            print("#######################################################")
            print(f"max_steps: {self.max_steps}")
            print("#######################################################")
            self.steps_per_epoch = len(unlearn_dataset) // (
                self.batch_size * self.gradient_accumulation_steps * self.num_devices
            )
        else:
            self.steps_per_epoch = self.max_steps // self.num_epochs

    def init_unlearner(self, logger):
        from functools import reduce
        def get_module_by_name(module, access_string):
            names = access_string.split(sep='.')
            return reduce(getattr, names, module)

        
        root = logger.get_root()
        unlearn_checkpoint = f"{root}/unlearn_checkpoint"
        if self.unlearn_method == "origin":
            self.unlearner = None
            return
        training_args = transformers.TrainingArguments(
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            warmup_steps=max(1, self.max_steps // 10),
            max_steps=self.max_steps,
            learning_rate=self.lr,
            bf16=True,
            bf16_full_eval=False,
            logging_steps=max(1, self.max_steps // 20),
            logging_dir=f"{root}/logs",
            output_dir=unlearn_checkpoint,
            optim="adamw_torch",
            weight_decay=self.weight_decay,
            remove_unused_columns=False,
            report_to=[],
            full_determinism=True,
            save_total_limit=5,
            save_steps=self.per_step,
        )
        lora_rank = 8

        if "FILA" in self.unlearn_method and "LFILA" not in self.unlearn_method:
            print(f'FILA in unlearn method ------- loading importance file from {os.path.abspath(self.importance_file)}')
            imp_file = torch.load(os.path.abspath(self.importance_file), map_location='cpu')
            f_cnt = imp_file['f_cnt']
            r_cnt = imp_file['r_cnt']
            importance_f = imp_file['importance_f']
            importance_r = imp_file['importance_r']

            importances = {n: torch.div(importance_f[n]/f_cnt, 1e-5+(importance_r[n]/r_cnt)) for n in importance_f.keys()}

            for old_name, importance in importances.items():

                name = old_name.replace("module.", '')
                lora_A = 'base_model.model.'+name.replace(".weight", '')+'.lora_A'
                lora_B = 'base_model.model.'+name.replace(".weight", '')+'.lora_B'
                base_layer = 'base_model.model.'+name.replace(".weight", '')+'.base_layer'
                scaling = 'base_model.model.'+name.replace(".weight", '')+'.scaling'

                lora_A = get_module_by_name(self.model, lora_A)
                lora_B = get_module_by_name(self.model, lora_B)
                base_layer = get_module_by_name(self.model, base_layer)
                scaling = get_module_by_name(self.model, scaling)

                orig_shape = base_layer.weight.shape
                W = base_layer.weight.data.reshape(orig_shape)
                dtype = W.dtype
                W = W.to(torch.float32)

                # Solve row-wise weighted low-rank approximation
                row_importance = importance.sum(dim=1).sqrt().to(W.device) # row-wise sum
                U, S, V = torch.svd_lowrank(row_importance[:,None] * W, q=lora_rank)

                S = S / scaling['default']

                new_lora_A = (V * torch.sqrt(S)).t()
                new_lora_B = (1/(row_importance+1e-5))[:,None] * (U * torch.sqrt(S))
                new_residual = base_layer.weight.data.reshape(orig_shape) - scaling['default'] * new_lora_B @ new_lora_A

                lora_A['default'].weight.data = new_lora_A.contiguous().to(dtype)
                lora_B['default'].weight.data = new_lora_B.contiguous().to(dtype)
                base_layer.weight.data = new_residual.contiguous().to(dtype)
    

        if "LFILA" in self.unlearn_method:
            print(f'LFILA in unlearn method ------- loading importance file from {self.importance_file}')
            imp_file = torch.load(self.importance_file, map_location='cpu')
            f_cnt = imp_file['f_cnt']
            r_cnt = imp_file['r_cnt']
            importance_f = imp_file['importance_f']
            importance_r = imp_file['importance_r']
            importances = {}

            # 첫번째 키, 값 출력

            for n in importance_f.keys():
                if 'lora_B' in n:
                    continue
                lora_A = n
                lora_B = n.replace('.lora_A', '.lora_B')

                importance_f_BA = importance_f[lora_B].to("cuda") @ importance_f[lora_A].to("cuda")
                importance_r_BA = importance_r[lora_B].to("cuda") @ importance_r[lora_A].to("cuda")

                name = n.replace("module.", '').replace(".lora_A.default.weight", '')
                importances[name] = torch.div(importance_f_BA / f_cnt, 1e-5+(importance_r_BA / r_cnt)).cpu()
                torch.cuda.empty_cache()

            for name, importance in importances.items():

                lora_A = name+'.lora_A'
                lora_B = name+'.lora_B'
                base_layer = name+'.base_layer'
                scaling = name+'.scaling'

                lora_A = get_module_by_name(self.model, lora_A)
                lora_B = get_module_by_name(self.model, lora_B)
                base_layer = get_module_by_name(self.model, base_layer)
                scaling = get_module_by_name(self.model, scaling)

                orig_shape = base_layer.weight.shape
                W = base_layer.weight.data.reshape(orig_shape)
                dtype = W.dtype
                W = W.to(torch.float32)

                # Solve row-wise weighted low-rank approximation
                row_importance = importance.sum(dim=1).sqrt().to(W.device) # row-wise sum
                U, S, V = torch.svd_lowrank(row_importance[:,None] * W, q=lora_rank)

                S = S / scaling['default']

                new_lora_A = (V * torch.sqrt(S)).t()
                new_lora_B = (1/(row_importance+1e-5))[:,None] * (U * torch.sqrt(S))
                new_residual = base_layer.weight.data.reshape(orig_shape) - scaling['default'] * new_lora_B @ new_lora_A

                lora_A['default'].weight.data = new_lora_A.contiguous().to(dtype)
                lora_B['default'].weight.data = new_lora_B.contiguous().to(dtype)
                base_layer.weight.data = new_residual.contiguous().to(dtype)
    
        if self.swa:
            callbacks = [WeightAveragingCallback(start_step=100, interval=5)]
        else:
            callbacks = None

        self.unlearner = get_unlearn_method(
            name=self.unlearn_method,
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=self.unlearn_dataset,
            eval_dataset=None,
            compute_metrics=None,
            args=training_args,
            data_collator=self.unlearn_collator,
            eval_collector=self.test_collator,
            gamma=self.gamma,
            beta=self.beta,
            if_wanda=self.if_wanda,
            gnr_rho=self.gnr_rho,
            sam_rho=self.sam_rho,
            callbacks=callbacks,
        )


    def eval(self, logger):
        self.model = None
        torch.cuda.empty_cache()
        root = logger.get_root()
        if self.resume_path is not None:
            model_name = self.resume_path
        else:
            model_name = os.path.join(root, "checkpoints")
 
        eval_few_shots(model_name=model_name,  task_list=["mmlu"],output_path=f"{root}/mmlu.json")
        torch.cuda.empty_cache()
        eval_few_shots(model_name=model_name, task_list=["wmdp_bio"],output_path=f"{root}/wmdp.json")
        torch.cuda.empty_cache()
        eval_few_shots(model_name=model_name, task_list=["wmdp_cyber"],output_path=f"{root}/wmdp.json")


    def save_full(self, logger):

        if isinstance(self.model, PeftModel):
            root = logger.get_root()
            residual_model_path = os.path.join(root, "checkpoints")
            residual_model = self.model.unload()
            residual_model.save_pretrained(residual_model_path)
            self.tokenizer.save_pretrained(residual_model_path)

        logger.save_ckpt("model", self.model, self.use_lora)
        logger.save_ckpt("tokenizer", self.tokenizer, self.use_lora)
    
    def run(self, logger):

        if self.resume_path is None:
            self.init_model()
            print_trainable_parameters(self.model)
            self.init_dataset()  
            
            self.init_unlearner(logger)
            if self.unlearner:
                self.unlearner.train()
            
            self.save_full(logger)
            
            os.system(f"rm -rf {logger.get_root()}/unlearn_checkpoint")
            self.eval(logger)
            os.system(f"rm -rf {logger.get_root()}/checkpoints")
        else:
            self.init_model()
            self.init_dataset()
            self.eval(logger)


def get(**kwargs):
    return Unlearn(**kwargs)