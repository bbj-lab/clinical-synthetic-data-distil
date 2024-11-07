from vllm import LLM, SamplingParams
import torch
import fire 

import os 
import pandas as pd 
from transformers import AutoTokenizer
from tqdm.auto import tqdm

def main(model_id="meta-llama/Meta-Llama-3.1-70B-Instruct", dataset=100000, 
         start_row=0): #, qtype='yes_no'):
    
    llm = LLM(model=model_id, dtype=torch.bfloat16, trust_remote_code=True, tensor_parallel_size=8)
    tokenizer = AutoTokenizer.from_pretrained(model_id, 
                                              trust_remote_code=True, 
                                              padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token 

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

    def create_prompt(note, qtype='yes'):
        num_questions = 5

        # , "Numeric", "Category", or "Phrase"
        if qtype == 'yes_no':
            type_str = "str - 'Yes/No'"
            answer_str = "str - 'Yes' or 'No'"
            type_request = f"Give a list of {num_questions*2} total, different, patient note-specific questions similar to clinical trial eligibility criteria. {num_questions} should have 'No' as the correct answer and {num_questions} should have 'Yes' as the answer" 
            json_example = f"""```json
    [
        {{
            "question" : "Does the note state that the patient is breathing normally on room air?",
            "type": "Yes/No",
            "answer": "No",
            "section": "History of Present Illness",
            "difficulty": "2",
            "source": "She currently is dependent on oxygen and wears 1.5-2 liters around the clock",
            "explanation": "The note states that she relies on oxygen and provides the amount as 1.5-2 liters so she is not breathing room air. We can assume since she is receiving o2 supplmentation and dependent on it, she cannot breathe normally on room air."
        }},
    ]```"""
        elif qtype == 'numeric':
            type_str = "str - 'Numeric'"
            answer_str = "float - a single number (e.g., 92.5)"
            type_request = f"Give a list of ten different, realistic, patient note-specific questions similar to clinical trial eligibility criteria with a numeric answer (only generate questions if numeric aswers are appropriate, otherwise end the response). All questions should be specific, many numeric values can be listed more than once so make sure to specify first, last, at admission, on discharge, highest, lowest, on a specific date, within ED / ICU etc.." 
            json_example = f"""```json
    [
        {{
            "question": "What was the patient's highest creatinine measurement recorded in the note?",
            "type": "Numeric",
            "answer": "1.4",
            "section": "Pertinent Results",
            "difficulty": "4",
            "source": "12/03/2023: CREAT: 1.4 \n 12/07/2023: CREAT: 1.1",
            "explanation": "The highest CREAT measurement was 1.4 because the only other creatinine measurement was 1.1 on 12/07/2023."
        }},
    ]```"""
        elif qtype == 'category':
            type_str = 'str - "Single Word or short phrase"'
            type_request = 'Give a list of up to 5 realistic, patient-specific questions similar to clinical trial eligibility criteria with a single word or short phrase answer' 
            json_example = f"""```json
    [
        {{
            "question": "What was the patient's primary diagnosis?",
            "type": "Category",
            "answer": "Myocardial Infarction (MI)",
            "section": "History of Present Illness",
            "source": "Admitted for MI",
            "difficulty": "2",
            "explanation": "The note states that she was admitted for MI."
        }},
    ]
    ```"""
        elif qtype == 'na-numeric':
            type_str = 'str - "a realistic clinical question which could be clinical trial eligbility criteria about the patient that cannot be answered relying on the information in the note"'
            answer_str = 'Not Available'
            type_request = f"Give a list of {num_questions} questions asking for numeric answers but where the note does not contain the answer. These should be questions which seem like they would be applicable to this patient and are similar to clinical trial eligibility criteria but cannot be answered based on the information in the note."
            json_example = f"""```json
    [
        {{
            "question" : "What was the patient's highest A1C recorded in the note during the hospitalization?",
            "type": "Yes/No",
            "answer" : "N/A",
            "section": "Not Found",
            "source" : "Not in Note",
            "difficulty": "4",
            "explanation": "The note does not include an A1C value during the hospitalization and we cannot infer a value for this patient."
        }},
    ]```"""
        elif qtype == 'na-bool':
            type_str = 'str - "a realistic clinical question which could be clinical trial eligbility criteria about the patient that cannot be answered based on the information in the note"'
            answer_str = 'Not Available'
            type_request = f"Give a list of {num_questions} Yes/No questions that are not answerable using the note. These should be questions which seem like they would be applicable to this patient and are similar to clinical trial eligibility criteria but cannot be answered based on the information in the note. These questions need to be things where the answer cannot be assumed simply because something is not mentioned (e.g., They should not be questions about whether the patient has been diagnosed with serious or chronic diseases because if they were it would be mentioned in the note, since it is not mentioned we can assume the answer is no rather than NA. Do not generate questions where Yes or No is known or can be inferred.)"
            json_example = f"""```json
    [
        {{
            "question" : "Does the note state the patient has ever taken aspirin for MI prevention?",
            "type": "Yes/No",
            "answer" : "N/A",
            "section": "Not Found",
            "source" : "Not in Note",
            "difficulty": "4",
            "explanation": "The note does not include medication history. It only includes medications prescribed during this encounter. If there were a medication history we would check the list to see if is present. If it was present we would answer Yes, if it were not present we would answer No but because there is no medication history we answer N/A"
        }},
    ]```"""
        else: 
            raise Exception('not implemented')
        
        
        prompt = f"""***PATIENT NOTE:
        {note}

        Provide the following: 
        * {type_request}

        *** Format your response as a list of JSON objects with the following keys: 
        * question: str - a question about the note describing a patient that resembles realistic clinical trial criteria
        * type: {type_str}
        * answer: {answer_str}
        * section: str - the specific section of the note which contains the answer to the question (Possible Answers: 'History of Present Illness', 'Past Medical History', 'Social History', 'Family History', 'Physical Exam', 'Pertinent Results', 'Brief Hospital Course', 'Discharge Medications', 'Discharge Disposition', 'Discharge Condition', 'Discharge Instructions')
        * source: str - exact quote of content in the note that allowed you to answer the question, this should be a quote directly taken from the "***PATIENT NOTE". Copy and pasting this string should exactly match content in the Note.
        * difficulty: int - a score from 1-10 indicating how difficult this question is to answer based on the note
        * explanation: str - explanation of why the answer is correct and how the source in the note helped to answer the question

        An example of how your JSON response should be formatted is shown below (this is only an example of one question, others should be in a list, your answer should be based on the ***Patient Note provided above):
        ***EXAMPLE RESPONSE:
        {json_example}

        Provide a response that can be directly read by the Python JSON Parser library. 
        """ 

        return prompt

    system_instruction = f"""You are a clinical research assistant helping to train a model to extract structured, discreet information about a patient's condition or history from clinical notes. 
    You answer with a single valid JSON object based on the patient note, where each field is correct and as short as possible. All dates have been shifted to keep data de-identified, so they may be in the future. We care only about information captured in the note, for example when asking what is the highest lab a patient has, we mean the highest lab recorded in the note. 
    Try to come up with questions that are specific to this patient based on their note, as opposed to general demographic questions or questions about their providers or hospitals etc.. Try not to repeat questions about similar topics. Do not have multiple questions about the same patient attribute.
    The questions should resemble clinical trial elgibility criteria. We will do this one by one for many patients, so please try to think of questions that are unique for each patient by including questions about less common, but clinically relevant information. 
    """

    filename = f"../data/mimic/notes_{dataset}.csv"
        
    original_df = load_csv(filename)

    # get splits
    def split_indices(df, splits=8, max_size=1000):
        n = len(df)
        # split_size = min(n // splits, max_size)
        split_size = max_size
        indices = [i for i in range(0, n, split_size)]
        return indices

    indices = split_indices(original_df[start_row:])
    print(indices)

    qtype_list = ['yes_no', 'numeric', 'na-numeric', 'na-bool']

    for srow in indices:
        for qtype in qtype_list:
            print(srow, qtype)
            if len(indices) > 1:
                inference_size = indices[1]
                df = original_df.iloc[srow:srow+indices[1]]
            else:
                inference_size = original_df.shape[0]
                df = original_df

            print(df.shape)
            print(df.columns)

            message_list = []
            for i, row in tqdm(df.iterrows(), total=len(df)):
                messages = [
                            {"role": "system", "content": system_instruction},
                            {"role": "user", "content": create_prompt(row['text'], qtype)},
                        ]
                message_chat = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False) 
                message_list.append(message_chat)

            df['prompt'] = message_list
            sampling_params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=4096)
            outputs = llm.generate(df['prompt'], sampling_params, use_tqdm=True)

            output_list = []
            for output in outputs:
                output_list.append(output.outputs[0].text)
                prompt = output.prompt
                generated_text = output.outputs[0].text

                df.loc[df['prompt'] == prompt, 'output'] = generated_text
            
            output_path = f"../synthetic_output/{dataset}"
            create_folders_for_path(output_path)

            df.to_csv(f"{output_path}/df_{qtype}_{srow}_{srow+inference_size}.csv")

if __name__ == '__main__':
    fire.Fire(main)