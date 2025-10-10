from data_module import TextForgetDatasetQA, TextForgetDatasetDPOQA
from dataloader import CustomTrainerForgetting, custom_data_collator_forget
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, set_seed
import hydra 
import transformers
import os
from peft import LoraConfig, get_peft_model, PeftModel
from pathlib import Path
from utils import get_model_identifiers_from_yaml
from omegaconf import OmegaConf
from functools import reduce

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

@hydra.main(version_base=None, config_path="config", config_name="forget")
def main(cfg):

    num_devices = int(os.environ.get('WORLD_SIZE', 1))
    print(f"num_devices: {num_devices}")

    if os.environ.get('LOCAL_RANK') is not None:
        local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        device_map = {'': local_rank}

    set_seed(cfg.seed)

    os.environ["WANDB_DISABLED"] = "true"
    model_cfg = get_model_identifiers_from_yaml(cfg.model_family)
    model_id = model_cfg["hf_key"]
    if cfg.model_path is None:
        cfg.model_path = model_cfg["ft_model_path"]

    Path(cfg.save_dir).mkdir(parents=True, exist_ok=True)
    # save cfg in cfg.save_dir
    if local_rank == 0:
        with open(f"{cfg.save_dir}/config.yaml", "w") as file:
            OmegaConf.save(cfg, file)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    print("######################")
    print("Saving to: ", cfg.save_dir)
    print("######################")

    max_length = 500
    if cfg.forget_loss == "dpo":
        torch_format_dataset = TextForgetDatasetDPOQA(
            cfg.data_path, 
            tokenizer=tokenizer, 
            model_family = cfg.model_family, 
            max_length=max_length, 
            split=cfg.split
        )
    else:
        torch_format_dataset = TextForgetDatasetQA(
            cfg.data_path, 
            tokenizer=tokenizer, 
            model_family=cfg.model_family, 
            max_length=max_length, 
            split=cfg.split, 
            loss_type=cfg.forget_loss
        )
    
    batch_size = cfg.batch_size
    gradient_accumulation_steps = cfg.gradient_accumulation_steps
    steps_per_epoch = len(torch_format_dataset)//(batch_size*gradient_accumulation_steps*num_devices)

    max_steps = int(cfg.num_epochs*len(torch_format_dataset))//(batch_size*gradient_accumulation_steps*num_devices)
    print(f"max_steps: {max_steps}")

    training_args = transformers.TrainingArguments(
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=max(1, steps_per_epoch),
        max_steps=max_steps,
        learning_rate=cfg.lr,
        bf16=True,
        bf16_full_eval=True,
        logging_steps=max(1,max_steps//20),
        logging_dir=f'{cfg.save_dir}/logs',
        output_dir=cfg.save_dir,
        optim="paged_adamw_32bit",
        save_strategy="steps" if cfg.save_model and (not cfg.eval_only) else "no",
        save_steps=steps_per_epoch,
        save_only_model=True,
        ddp_find_unused_parameters= False,
        deepspeed='config/ds_config.json',
        weight_decay = cfg.weight_decay,
        eval_steps = steps_per_epoch,
        evaluation_strategy = "steps" if cfg.eval_while_train else "no",
        seed=cfg.seed
    )
    
    #first get the base model architectur2e
    #if there is a pytorch*.bin file in the model path, then load that. use regex there can be anythign in between pytorch and .bin
    import re
    path_found = False
    for file in os.listdir(cfg.model_path):
        if re.search("pytorch.*\.bin", file):
            path_found = True
            break
        
        if re.search("model-*\.safetensors", file):
            path_found = True
            break

    oracle_model = None

    if path_found:
        config = AutoConfig.from_pretrained(model_id)

        print("Loading from checkpoint")
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_path, 
            config=config, 
            use_flash_attention_2=model_cfg["flash_attention2"]=="true", 
            torch_dtype=torch.bfloat16, 
            trust_remote_code = True,
            #device_map="cpu" ### NEED TO REMOVE AFTER DEBUGGING
        )
        if "KL" in cfg.forget_loss or "npo" in cfg.forget_loss:
            oracle_model = AutoModelForCausalLM.from_pretrained(
                cfg.model_path, 
                config=config, 
                use_flash_attention_2=model_cfg["flash_attention2"]=="true", 
                torch_dtype=torch.bfloat16, 
                trust_remote_code = True
            )

    else:
        print("Loading after merge and unload")
        model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            use_flash_attention_2=model_cfg["flash_attention2"]=="true", 
            torch_dtype=torch.bfloat16, 
            device_map=device_map
        )
        #now use the checkpoint to add the LoRA modules
        model = PeftModel.from_pretrained(model, model_id = cfg.model_path)
        #save this as a standard model so that we can again do PEFT style finetuneing from scratch
        model = model.merge_and_unload()
        #save the model for next time
        model.save_pretrained(cfg.model_path)
    
    
    # Hot fix for https://discuss.huggingface.co/t/help-with-llama-2-finetuning-setup/50035
    model.generation_config.do_sample = True

    config = LoraConfig(
        r=cfg.LoRA.r, 
        lora_alpha=cfg.LoRA.alpha, 
        target_modules=find_all_linear_names(model),
        lora_dropout=cfg.LoRA.dropout,
        bias="none", 
        task_type="CAUSAL_LM"
    )
    if cfg.LoRA.r != 0:
        model = get_peft_model(model, config)
        print_trainable_parameters(model)

    ### MODIFY MODEL WEIGHTS BASED ON IMPORTANCES
    def get_module_by_name(module, access_string):
        names = access_string.split(sep='.')
        return reduce(getattr, names, module)

    if cfg.i_type is not None:
        print(f'loading importance file from {cfg.importance_file}')
        imp_file = torch.load(cfg.importance_file, map_location='cpu')
        f_cnt = imp_file['f_cnt']
        r_cnt = imp_file['r_cnt']
        importance_f = imp_file['importance_f']
        importance_r = imp_file['importance_r']

        if cfg.i_type == 'fila':
            importances = {n: torch.div(importance_f[n]/f_cnt, 1e-5+(importance_r[n]/r_cnt)) for n in importance_f.keys()}
        elif cfg.i_type == 'vila':
            importances = {}
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
        else:
            raise ValueError(f'Unknown i_type: {cfg.i_type}')
                  
        for old_name, importance in importances.items():
            print(old_name)
            name = old_name.replace("module.", '')
            if cfg.i_type == 'fila':
                lora_A = 'base_model.model.'+name.replace(".weight", '')+'.lora_A'
                lora_B = 'base_model.model.'+name.replace(".weight", '')+'.lora_B'
                base_layer = 'base_model.model.'+name.replace(".weight", '')+'.base_layer'
                scaling = 'base_model.model.'+name.replace(".weight", '')+'.scaling'
            elif cfg.i_type == 'vila':
                lora_A = name+'.lora_A'
                lora_B = name+'.lora_B'
                base_layer = name+'.base_layer'
                scaling = name+'.scaling'
                
            lora_A = get_module_by_name(model, lora_A)
            lora_B = get_module_by_name(model, lora_B)
            base_layer = get_module_by_name(model, base_layer)
            scaling = get_module_by_name(model, scaling)

            orig_shape = base_layer.weight.shape
            W = base_layer.weight.data.reshape(orig_shape)
            dtype = W.dtype
            W = W.to(torch.float32)

            row_importance = importance.sum(dim=1).sqrt().to(W.device) # row-wise sum
            U, S, V = torch.svd_lowrank(row_importance[:,None] * W, q=cfg.LoRA.r)

            S = S / scaling['default']

            new_lora_A = (V * torch.sqrt(S)).t()
            new_lora_B = (1/(row_importance+1e-5))[:,None] * (U * torch.sqrt(S))
            new_residual = base_layer.weight.data.reshape(orig_shape) - scaling['default'] * new_lora_B @ new_lora_A

            lora_A['default'].weight.data = new_lora_A.contiguous().to(dtype)
            lora_B['default'].weight.data = new_lora_B.contiguous().to(dtype)
            base_layer.weight.data = new_residual.contiguous().to(dtype)

    trainer = CustomTrainerForgetting(
        model=model,
        tokenizer=tokenizer,
        train_dataset=torch_format_dataset,
        eval_dataset=torch_format_dataset,
        args=training_args,
        data_collator=custom_data_collator_forget,
        oracle_model = oracle_model,
        forget_loss = cfg.forget_loss,
        eval_cfg = cfg.eval,
        lcoef=cfg.lcoef,
        beta=cfg.beta,
    )
    
    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
    if cfg.eval_only:
        trainer.evaluate()
    else:
        trainer.train()

    #save the tokenizer
    if cfg.save_model and (not cfg.eval_only):
        model = model.unload()
        print(model)
        model.save_pretrained(cfg.save_dir)
        tokenizer.save_pretrained(cfg.save_dir)

    #delete all "global_step*" files in the save_dir/checkpoint-*/ directories
    if local_rank == 0:
        for file in Path(cfg.save_dir).glob("checkpoint-*"):
            for global_step_dir in file.glob("global_step*"):
                #delete the directory
                import shutil
                shutil.rmtree(global_step_dir)

if __name__ == "__main__":
    main()