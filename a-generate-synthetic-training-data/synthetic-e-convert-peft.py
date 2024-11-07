from peft import AutoPeftModelForCausalLM
import torch
 
 # these need to be specified and could be arguments
checkpoint_num = "25000"
model_id = 'meta-llama/Meta-Llama-3.1-8B_all_1'
model_dir=f"../results/{model_id}/checkpoint-{checkpoint_num}"

output_dir = f"./models/{model_id.split('/')[1]}"
print(model_id, model_dir)
print(output_dir)

# Load PEFT model on CPU
model = AutoPeftModelForCausalLM.from_pretrained(
    model_dir,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
)

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
tokenizer.save_pretrained(output_dir)
