# Clinical Synthetic Data Distillation for Scaleable Clinical Information Extraction

### Elizabeth Geena Woo, Michael C. Burkhart, Emily Alsentzer, Brett K Beaulieu-Jones

This code supplements the manuscript "Synthetic Data Distillation Enables the Extraction of Clinical Information at Scale" [^1]. The code prompts Llama-3.1-70B-Instruct [^2] to generate a synthetic dataset and then uses it to fine-tune an instance of Llama-3.1-8B [^2] as follows:

## Abstract
Large-language models (LLMs) have shown promising potential for extracting information from clinical notes. Deploying these models at scale can be challenging due to high computational costs, regulatory constraints, and privacy concerns. To address these challenges, we used synthetic data distillation to fine-tune smaller, open-source LLMs that achieve performance similar to that of larger models, including the teacher model. These smaller models can be run on less expensive local hardware or at a vastly reduced cost in cloud deployments. In this study, we used Llama-3.1-70B-Instruct to generate synthetic training examples in the form of question-answer pairs along with supporting information and model-assigned difficulty scores. These synthetic examples were used to fine-tune the smaller Llama-3.1-8B-Instruct model. We evaluated the performance of these models on an annotated synthetic dataset resembling clinical trial criteria, the i2b2 2018 Clinical Trial Eligibility Challenge, and clinical notes reflecting the clinical trial for apixaban. The fine-tuned models outperformed the 8B-Instruct model on all tasks and in some cases even exceeded the performance of the larger 70B-Instruct model. This work demonstrates the potential of synthetic data distillation to enable more scalable and efficient clinical information extraction, which could be applied toward improving accuracy and efficiency of patient phenotyping and clinical-trial matching.

Full pre-print: https://www.medrxiv.org/content/10.1101/2024.09.27.24314517v1

## There are 4 main components and two main python scripts used throughout 
## A.) a-generate-synthetic-training-data

Script for generating synthetic question-answer pairs from clinical notes using LLaMA models.
### A. Synthetic Data Generation (synthetic-a-create.py)
Script for generating synthetic question-answer pairs from clinical notes using LLaMA models.

**Arguments:**
- `model_id` (default: "meta-llama/Meta-Llama-3.1-70B-Instruct"): Model for generating questions
- `dataset` (default: 100000): Number of examples to generate
- `start_row` (default: 0): Starting row in dataset

Generates four types of questions:
- yes/no questions
- numeric questions 
- not-answerable numeric questions
- not-answerable boolean questions

### B. Post-Processing (synthetic-b-post_process_join_notes.ipynb)
Notebook for processing and combining generated synthetic data:
- Loads and combines CSV files from different question types
- Extracts structured data from JSON responses
- Filters by difficulty level
- Creates different dataset combinations (boolean only, numeric only, etc.)
- Saves processed datasets in multiple formats

### C. Training Preparation (synthetic-c-prep-training.ipynb) 
Notebook for preparing the final training datasets:
- Balances datasets across question types
- Formats prompts and responses for training
- Creates train/test splits
- Converts to HuggingFace datasets format

### D. Training Script (synthetic-d-run-finetune.sh)
Shell script for launching distributed training:
```bash
accelerate launch --config_file ../configs/deep_speed2.yaml \
  --multi_gpu --num_machines 1 --num_processes 8 \
  finetune.py --min_difficulty "1" --input_file all \
  --model_id meta-llama/Llama-3.2-1B
```

### E. Model Conversion (synthetic-e-convert-peft.py)
Script for converting trained PEFT models for deployment:
- Loads trained PEFT model
- Saves model and tokenizer in deployment format

### Requirements
```
torch
transformers
pandas
fire
datasets
accelerate
peft
vllm
tqdm
deepspeed
```

### Usage

1. Generate synthetic data:
```bash
python synthetic-a-create.py --dataset 100000
```

2. Process the data using notebooks B and C sequentially

3. Launch training:
```bash
bash synthetic-d-run-finetune.sh
```

4. Convert model:
```bash
python synthetic-e-convert-peft.py
```

## B.) b-synthetic-eval-task

This folder contains the preprocessing and evaluation scripts for the synthetic data evaluation task. 

### Dataset
Held-out set of 42,498 synthetic examples generated in an identical manner to the dataset used for fine-tuning. The breakdown of examples by type was as follows: 10,722 (25.2%) boolean, 10,666 (25.1%) numeric, 10,664 (25.1%) na-boolean, and 10,446 (24.6%) na-numeric. From this set, we drew a random sample containing 1,000 examples and manually annotated it as described in the “Limited Human Review” subsection of our methods, correcting questions, answers, and explanations when necessary. A description of these data are available in Supplementary Table 1.

### Usage
There are three steps:
1. Preprocessing (annotated_synthetic-a-preprocess.ipynb)
2. Running inference (with eval-inference.py)
3. Evaluating performance (annotated_synthetic-b-evaluate-performance.ipynb)

## C.) c-i2b2-eval-task

This folder contains the preprocessing and evaluation scripts for the i2b2 clinical trial eligibility criteria task.

### Dataset
The clinical trial eligibility criteria cohort selection shared task from the 2018 National NLP Clinical Challenges. Track 1 contains 288 de-identified longitudinal medical records for patients with diabetes, many of whom are at risk for heart disease. The records are manually annotated according to 13 selection criteria adapted from real clinical trials and split into a 202-patient training set and 86-patient test set. We calculated balanced accuracy and micro-F1 score on both the training and test datasets corresponding to the original challenge. At the time of the challenge, the top-performing team adopted a rule-based method to obtain a micro-F1 score of 0.91 on the test set. Other teams achieved similar results (F1 > 0.9) with hybrid approaches; for example, cTakes was used by 3 of the top 5 teams to extract knowledge from the text. Because we only use this dataset to test zero-shot extraction and do not train on it, we are able to evaluate the model performance on both the training and test sets to have a larger sample size. 

### Usage
There are three steps:
1. Preprocessing (i2b2-a-preprocess.ipynb)
2. Running inference (with eval-inference.py)
3. Evaluating performance (i2b2-b-evaluate-performance.ipynb)

## D.) d-apixaban-eval-task

This folder contains the preprocessing and evaluation scripts for the apixaban clinical trial task.

### Dataset
Clinical trial eligibility criteria resembling those of the 2011 ARISTOTLE trial comparing apixaban to warfarin. We developed 23 human-generated boolean and numeric questions assessing these criteria (Supplementary Table 4). Using these questions, we manually annotated notes for 2300 total question-answer pairs within MIMIC-IV. Notes from MIMIC-IV were taken from after 2012 to ensure no overlap with any of the notes from MIMIC-III which were used to generate synthetic data. We evaluated the models on these question-answer pairs and calculated both balanced accuracy and micro-F1 score. We are releasing the dataset and manual annotations to Physionet and will make them available under the same data use terms as MIMIC-III/IV. 

### Usage
There are three steps:
1. Preprocessing (apixaban-a-preprocess.ipynb)
2. Running inference (with eval-inference.py)
3. Evaluating performance (apixaban-b-evaluate-performance.ipynb)

## 1.) Finetuning - finetune.py

The script implements a supervised fine-tuning (SFT) pipeline using the `trl` library, with the following key features:

- 4-bit quantization for memory-efficient training
- LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning
- Support for distributed training using Accelerate
- Custom chat template for consistent input formatting
- DataCollator for completion-only language modeling

### Arguments

The script accepts the following command-line arguments:

- `model_id` (default: "meta-llama/Meta-Llama-3.1-8B"): The base model to fine-tune
- `input_file` (default: "all"): Name of the input dataset file
- `min_difficulty` (default: 5): Minimum difficulty level for training examples
- `max_seq_length` (default: 12000): Maximum sequence length for tokenization

### Requirements

```
torch
datasets
pandas
fire
accelerate
trl
transformers
peft
wandb
```

### Training Configuration

The training process uses the following settings:

- 4-bit quantization with NF4 format
- bfloat16 precision
- Gradient checkpointing enabled
- Constant learning rate of 0.0002
- Per device batch size of 1 with gradient accumulation of 2 steps
- QLoRA configuration:
  - r = 8 (rank)
  - alpha = 16
  - Target: all linear layers
  - No bias adaptation

### Usage

To train the model (with accelerate - ./a-generate-synthetic-training-data/synthetic-d-run-finetune.sh):

```bash
python finetune.py \
  --model_id "meta-llama/Meta-Llama-3.1-8B" \
  --input_file "all" \
  --min_difficulty 5 \
  --max_seq_length 12000
```

## 2.) Inference - eval-inference.py

This script uses vLLM for efficient batch inference. 

### Evaluation Arguments (eval-inference.py)
- `model_id` (default: "meta-llama/Meta-Llama-3.1-70B-Instruct"): Model to use for inference
- `dataset_name` (default: "i2b2"): Name of the evaluation dataset
- `input_file` (default: "train"): Input file to process
- `num_inference` (default: "all"): Number of examples to process
- `start_row` (default: 0): Starting row for processing
- `max_output_tokens` (default: 256): Maximum length of generated responses
- `chunk_size` (default: 10000): Number of examples to process in each batch
- `temperature` (default: 0): Sampling temperature
- `top_p` (default: 1): Top-p sampling parameter

### Inference-specific requirements
```
vllm
tqdm
```

### Evaluation Settings
- Tensor parallel size: 8
- bfloat16 precision
- Batch processing with configurable chunk size
- Additional Llama stop tokens for controlled generation

### Usage
```bash
python eval-inference.py \
  --model_id "meta-llama/Meta-Llama-3.1-70B-Instruct" \
  --dataset_name "i2b2" \
  --input_file "train" \
  --chunk_size 10000 \
  --max_output_tokens 256
```


## Computing environment

This code was tested on the Center for Research Informatics’
["Randi" cluster](https://cri.uchicago.edu/hpc/) at the University of Chicago.
At the time of publication, the cluster's GPU nodes each contained 8 Nvidia
A100 GPU's with two 16-core 3.0-GHz AMD Milan processors.


## References
[^1]:
    Synthetic Data Distillation Enables the Extraction of Clinical Information
    at Scale. 2024. Available from: https://www.medrxiv.org/content/10.1101/2024.09.27.24314517v1

[^2]:
    Dubey A, Jauhri A, Pandey A, Kadian A, Al-Dahle A, Letman A, et al. The
    Llama 3 herd of models. arXiv [cs.AI]. 2024. Available from:
    http://dx.doi.org/10.48550/arXiv.2309.03882

[^3]:
    BBJ Lab. Annotation-UI. Available from:
    https://github.com/bbj-lab/annotation-ui
