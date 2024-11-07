import torch
import datasets
from datasets import Dataset
import pandas as pd
import fire 
import os
from accelerate import Accelerator

from trl import (
    ModelConfig,
    RichProgressCallback,
    SFTConfig,
    SFTTrainer,
    get_peft_config,
    get_quantization_config,
    get_kbit_device_map,
    DataCollatorForCompletionOnlyLM
)

from trl import SFTTrainer
from peft import LoraConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments

# model_id = "meta-llama/Meta-Llama-3.1-8B"
LLAMA31_CHAT_TEMPLATE = "{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"

def create_folders_for_path(folder_path):
    # Create all intermediate directories needed to contain the specified path
    os.makedirs(folder_path, exist_ok=True)
    print(f"All necessary directories have been created for the path: {folder_path}")

def load_csv(filename):
    try:
        df = pd.read_csv(filename)
    except Exception as e:
        print('read plain csv failed, try gzip')
        df = pd.read_csv(filename, compression='gzip')
    return df

def main(model_id="meta-llama/Meta-Llama-3.1-8B", input_file="all", min_difficulty=5, 
         max_seq_length=12000): #, max_output_tokens=2048):
    
    # Initialize Accelerator
    accelerator = Accelerator()

    folder_path = "./data/synthetic-raw"
    print('accelerator main process')

    train_file = f"{folder_path}/train_{input_file}_{min_difficulty}.arrow"
    test_file = f"{folder_path}/test_{input_file}_{min_difficulty}.arrow"
        
    # We do this upstream now to avoid overwriting or running into shuffle issues - shuffle false fails due to concatenating the DF (the last X will be the same type and not representative for testing)
    # if accelerator.is_main_process:
    #     # Load CSV into a pandas DataFrame
    #     file_name = f"{input_file}_{min_difficulty}_prompts.csv"
    #     full_file_name = f"{folder_path}/{file_name}"
    #     print('load file')
    #     df = load_csv(full_file_name)
    #     print("input shape", df.shape)
        
    #     # Convert the DataFrame to a Dataset
    #     print('convert to dataset')
    #     ds = Dataset.from_pandas(df)
    #     ds_dict = ds.train_test_split(test_size=0.1, shuffle=False)
        
    #     ds_dict['train'].save_to_disk(train_file)
    #     ds_dict['test'].save_to_disk(test_file)

    # Ensure all processes wait until the dataset is saved
    accelerator.wait_for_everyone()

    # Load the split datasets
    ds_dict = datasets.DatasetDict({"train":Dataset.load_from_disk(train_file),
                                    "test": Dataset.load_from_disk(test_file)
                                    })

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # trying fix from https://github.com/unslothai/unsloth/issues/416
    tokenizer.add_special_tokens({"pad_token": "<|reserved_special_token_0|>"})

    tokenizer.chat_template = LLAMA31_CHAT_TEMPLATE
    tokenizer.model_max_length = max_seq_length
    tokenizer.truncation_side = "left"

    quantization_config = BitsAndBytesConfig(load_in_4bit=True,
                                             bnb_4bit_compute_dtype=torch.bfloat16,
                                             bnb_4bit_use_double_quant=True,
                                             bnb_4bit_quant_type= "nf4"
                                            )


    print('Load Model')
    model = AutoModelForCausalLM.from_pretrained(model_id, 
                                                 device_map=get_kbit_device_map() if quantization_config is not None else None,
                                                 torch_dtype=torch.bfloat16, 
                                                 quantization_config=quantization_config,
                                                 use_cache=False # for gradient checkpointing
    )
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.pad_token_id = tokenizer.pad_token_id

    system_instruction = f"""You are a clinical research assistant helping to accurately answer questions from clinical notes. You answer with a single valid JSON object based on the patient note. 
    All dates have been shifted to keep data de-identified, so they may be in the future. We care only about information captured in the note, for example when asking what is the highest lab a patient has, we mean the highest lab recorded in the note. 
    If you cannot find information to answer the question, answer "NA", unless specifically instructed to do something different. 
    """

    def apply_template(example, tokenizer):
        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": example['prompt']},
            {"role": "assistant", "content": example['output']}
        ]

        example['input_ids'] = tokenizer.apply_chat_template(messages, tokenize=True,
                                                             truncation=True, max_length=max_seq_length, 
                                                             add_generation_prompt=False) #, num_proc=32)
        return example

    print('Apply Chat Template')
    column_names = list(ds_dict["train"].features)
    ds_dict = ds_dict.map(apply_template,
                        fn_kwargs={"tokenizer": tokenizer},
                        remove_columns=column_names,
                        desc="Applying chat template",
                        num_proc=32
                        )
                        #   load_from_cache_file=False)

    train_dataset = ds_dict['train'] 
    test_dataset = ds_dict['test']

    response_template = "<|start_header_id|>assistant<|end_header_id|>\n\n"
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

    r_val = 8
    l_alpha = 16

    training_args = TrainingArguments(
        output_dir=f"./results/{model_id}_{input_file}_{min_difficulty}_{r_val}_{l_alpha}",
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        learning_rate= 0.0002,
        lr_scheduler_type='constant',
        logging_dir='./logs',
        logging_steps=10,
        save_strategy='steps',
        save_steps=500,
        save_total_limit=20,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        # dataloader_num_workers=16,
        bf16=True,
        ddp_find_unused_parameters=False,
        report_to='wandb'
    )

    lora_config = LoraConfig(
        r=r_val,
        lora_alpha=l_alpha,
        target_modules="all-linear",
        bias="none",
        task_type="CAUSAL_LM",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        max_seq_length=max_seq_length,
        peft_config=lora_config,
        train_dataset=train_dataset,
        data_collator=collator,
    )

    trainer.train()
    trainer.save_model('./models')

if __name__ == '__main__':
    fire.Fire(main)
