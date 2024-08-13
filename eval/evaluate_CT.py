import argparse
import json
import os
import cv2
import numpy as np

from tqdm import tqdm

from inference import *

parser = argparse.ArgumentParser()
parser.add_argument("--model-path", type=str, default=None)
parser.add_argument("--model-base", type=str, default=None)
parser.add_argument("--conv-mode", type=str, default=None)
parser.add_argument("--sep", type=str, default=",")
parser.add_argument("--temperature", type=float, default=0.2)
parser.add_argument("--top_p", type=float, default=None)
parser.add_argument("--num_beams", type=int, default=1)
parser.add_argument("--max_new_tokens", type=int, default=512)
args = parser.parse_args()


def evaluate(model_path, data_path, result_path, args):
    disable_torch_init()
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, args.model_base, model_name
    )
    with open(data_path, 'r') as f:
        test_data = json.load(f)

    results = []
    for i, data in tqdm(enumerate(test_data)):
        answer = inference(args, model, model_name, tokenizer, image_processor,
                           qs = data['conversations'][0]['value'].split('\n')[1],
                           image_files = [data['image']],)
        
        target = data['conversations'][1]['value']
        result = {
            'answer': answer,
            'target': data['conversations'][1]['value'],
        }
        results.append(result)
    with open(result_path, 'w') as f:
        json.dump(results, f, ensure_ascii=False)


date = "240811"
version = "c1.epoch6"
# base_model = "llava-med-v1.5-mistral-7b"
base_model = "llava-v1.6-vicuna-13b"
# base_model = "llava-v1.6-mistral-7b"
# grounded_image_path = f"/home/jc/workspace/MedLLMs/Eso-Llava/eval/results/images/{date}{version}-{base_model}"
# os.makedirs(grounded_image_path, exist_ok=True)

if __name__ == '__main__':
    DATA_INDEX = "CT/advanced"
    model_path = f"/date/jc/models/MedLLMs/LLaVA-Med/merged/{date}{version}-lora-{base_model}-merged"
    data_path = f"/date/jc/data/Eso-Llava/processed_data/{DATA_INDEX}/v3/test.json"
    result_path = f"/home/jc/workspace/MedLLMs/Eso-Llava/eval/results/{date}{version}-lora-{base_model}-merged-test.json"
    evaluate(model_path, data_path, result_path, args)
