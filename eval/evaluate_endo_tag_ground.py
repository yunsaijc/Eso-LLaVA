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

def draw_coordinates_on_image(image_path, tgt_coords_list, coordinates_list, output_path):
    image = cv2.imread(image_path)  # 读取图片
    original_image = image.copy()
    if image is None: raise ValueError("Image not found or unable to read")
    
    height, width = image.shape[:2]
    def convert_to_pixels(coords):
        return [(int((x / 100) * width), int((y / 100) * height)) for x, y in coords]

    for coordinates in coordinates_list:    # 绘制模型给出的方框（绿色）
        points_pixel = convert_to_pixels(coordinates)
        cv2.polylines(image, [np.array(points_pixel, dtype=np.int32)], 
                      isClosed=True, color=(0, 255, 0), thickness=2)
    for tgt_coords in tgt_coords_list:  # 绘制目标方框（蓝色）
        tgt_points_pixel = convert_to_pixels(tgt_coords)
        cv2.polylines(image, [np.array(tgt_points_pixel, dtype=np.int32)], 
                      isClosed=True, color=(255, 0, 0), thickness=2)
    concat_image = np.concatenate((original_image, image), axis=1) # 将原图和标注后的图像并排显示
    cv2.imwrite(output_path, concat_image)

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
        try: coords = eval(answer.split('\n')[0].split(":")[1].strip().strip("."))
        except: 
            print(">>> Answer: ", answer)
            coords = []
            answer = "Area: [].\nLabel: no_lesions."
            # raise ValueError(">>> Answer coordinates not found")
        tgt_coords = eval(target.split('\n')[0].split(":")[1].strip().strip("."))
        original_image_path = data['image']
        image_name = original_image_path.split('/')[-1]
        global grounded_image_path
        target_image_path = f"{grounded_image_path}/{image_name}"
        draw_coordinates_on_image(original_image_path, tgt_coords, coords, target_image_path)
        result = {
            'answer': answer,
            'target': data['conversations'][1]['value'],
            'original_image_path': original_image_path,
            'grounded_image_path': grounded_image_path,
        }
        results.append(result)
    with open(result_path, 'w') as f:
        json.dump(results, f, ensure_ascii=False)


date = "240812"
version = "e1.epoch20"
# base_model = "llava-med-v1.5-mistral-7b"
base_model = "llava-v1.6-vicuna-13b"
# base_model = "llava-v1.6-mistral-7b"
grounded_image_path = f"/home/jc/workspace/MedLLMs/Eso-Llava/eval/results/images/{date}{version}-{base_model}"
os.makedirs(grounded_image_path, exist_ok=True)

if __name__ == '__main__':
    DATA_INDEX = "endo/precancer"
    model_path = f"/date/jc/models/MedLLMs/LLaVA-Med/merged/{date}{version}-lora-{base_model}-merged"
    data_path = f"/date/jc/data/Eso-Llava/processed_data/{DATA_INDEX}/v3/test.json"
    result_path = f"/home/jc/workspace/MedLLMs/Eso-Llava/eval/results/{date}{version}-lora-{base_model}-merged-test.json"
    evaluate(model_path, data_path, result_path, args)
