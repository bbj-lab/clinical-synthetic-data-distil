from vllm import LLM, SamplingParams
import torch
import fire 

import os 
import pandas as pd 
from transformers import AutoTokenizer
from tqdm.auto import tqdm

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

def save_chunk(df_chunk, folder_path, input_file, start_row, num_inference, chunk_index):
    chunk_file_name = f"{folder_path}/{input_file.split('.')[0]}_{start_row}_{num_inference}_chunk_{chunk_index}.csv"
    df_chunk.to_csv(chunk_file_name, index=False)
    print(f"Chunk {chunk_index} saved successfully at {chunk_file_name}")

# meta-llama/Meta-Llama-3.1-70B-Instruct
def main(model_id="meta-llama/Meta-Llama-3.1-70B-Instruct", dataset_name="i2b2", input_file="train", num_inference="all", 
         start_row=0, max_output_tokens=256, chunk_size=10000, temperature=0, top_p=1):
    file_name = f"../eval-data/{dataset_name}/{input_file}.csv"
    print("model", model_id)
    print("Dataset", dataset_name)
    print("input_file", input_file)
    print("num_inference", num_inference)

    df = load_csv(file_name)
    print(df.shape)

    if num_inference == "all":
        df = df.iloc[start_row:]
    else:
        df = df.iloc[start_row:start_row+num_inference]

    print(df.shape)
    print(df.columns)

    system_instruction = f"""You are a clinical research assistant helping to accurately answer questions from clinical notes. You answer with a single valid JSON object based on the patient note. 
All dates have been shifted to keep data de-identified, so they may be in the future. We care only about information captured in the note, for example when asking what is the highest lab a patient has, we mean the highest lab recorded in the note. 
If you cannot find information to answer the question asked of you in the note answer NA, unless the rest of the prompt recommends something different. 
"""

    llm = LLM(model=model_id, dtype=torch.bfloat16, trust_remote_code=True, tensor_parallel_size=8)

    tokenizer = AutoTokenizer.from_pretrained(model_id, 
                                            trust_remote_code=True, 
                                            padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token 

    # can use this if evaluating other parameters
    # folder_path = f"../outputs/{dataset_name}/params/{model_id.split('/')[-1]}/{temperature}_{top_p}"
    folder_path = f"../outputs/{dataset_name}/{model_id.split('/')[-1]}"
    create_folders_for_path(folder_path)

    for chunk_index, chunk_df in enumerate(tqdm([df[i:i + chunk_size] for i in range(0, df.shape[0], chunk_size)])):
        message_list = []
        for i, row in chunk_df.iterrows():
            messages = [
                        {"role": "system", "content": system_instruction},
                        {"role": "user", "content": row['prompt']},
                    ]
            message_chat = tokenizer.apply_chat_template(messages, add_generation_prompt=True, 
                                                        tokenize=False) 
            message_list.append(message_chat)

        chunk_df['input'] = message_list

        sampling_params = SamplingParams(temperature=temperature, 
                                         top_p=top_p, 
                                         max_tokens=max_output_tokens,
                                         stop_token_ids=[128000, 128001, 128007, 128008, 128009]
                                        )
        outputs = llm.generate(chunk_df['prompt'], sampling_params, use_tqdm=True)

        output_list = []

        for output in outputs:
            output_list.append(output.outputs[0].text)
            prompt = output.prompt
            generated_text = output.outputs[0].text

            chunk_df.loc[chunk_df['prompt'] == prompt, 'output'] = generated_text

        save_chunk(chunk_df, folder_path, input_file, start_row, num_inference, chunk_index)

if __name__ == '__main__':
    fire.Fire(main)
