import json
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from shapely.geometry import Polygon

def parse_prediction(prediction):
    lines = prediction.split('\n')
    label = None
    coordinates = None
    
    for line in lines:
        if line.startswith("Label:"):
            label = line.split(":")[1].strip().lower()
        elif line.startswith("Area:"):
            coord_str = line.split("Area:")[1].strip().strip(".")
            coordinates = eval(coord_str)
    
    return label, coordinates

def calculate_iou(boxes1, boxes2):
    total_iou = 0
    total_pairs = 0
    
    for box1 in boxes1:
        for box2 in boxes2:
            p1 = Polygon(box1)
            p2 = Polygon(box2)
            intersection = p1.intersection(p2).area
            union = p1.union(p2).area
            
            iou = intersection / union if union > 0 else 0
            total_iou += iou
            total_pairs += 1
    
    return total_iou / total_pairs if total_pairs > 0 else 0

def main():
    date = "240812"
    version = "e1.epoch20"
    # base_model = "llava-med-v1.5-mistral-7b"
    base_model = "llava-v1.6-vicuna-13b"
    # base_model = "llava-v1.6-mistral-7b"
    result_path = "/home/jc/workspace/MedLLMs/Eso-Llava/eval/results/"
    inference_result_path = result_path + f"{date}{version}-lora-{base_model}-merged-test.json"
    
    with open(inference_result_path, 'r') as f:
        results = json.load(f)
    
    true_labels = []
    pred_labels = []
    ious = []
    
    for item in results:
        pred_label, pred_coords = parse_prediction(item['answer'])
        true_label, true_coords = parse_prediction(item['target'])
        
        true_labels.append(true_label)
        pred_labels.append(pred_label)
        
        # if true_coords and pred_coords:
        #     iou = calculate_iou(true_coords, pred_coords)
        #     ious.append(iou)
    
    # 计算分类指标
    accuracy = accuracy_score(true_labels, pred_labels)
    precision = precision_score(true_labels, pred_labels, average='weighted', zero_division=0)
    recall = recall_score(true_labels, pred_labels, average='weighted', zero_division=0)
    f1 = f1_score(true_labels, pred_labels, average='weighted', zero_division=0)
    
    # 计算每个类别的指标
    unique_labels = list(set(true_labels + pred_labels))
    class_precision = precision_score(true_labels, pred_labels, average=None, labels=unique_labels, zero_division=0)
    class_recall = recall_score(true_labels, pred_labels, average=None, labels=unique_labels, zero_division=0)
    class_f1 = f1_score(true_labels, pred_labels, average=None, labels=unique_labels, zero_division=0)
    
    # 计算混淆矩阵
    cm = confusion_matrix(true_labels, pred_labels, labels=unique_labels)
    
    # 计算平均IoU
    # avg_iou = sum(ious) / len(ious) if ious else 0
    
    print(f"Overall Accuracy: {accuracy*100:.4f}")
    print(f"Overall Precision: {precision*100:.4f}")
    print(f"Overall Recall: {recall*100:.4f}")
    print(f"Overall F1 Score: {f1*100:.4f}")
    # print(f"Average IoU: {avg_iou:.4f}")
    
    print("\nPer-class metrics:")
    for label, p, r, f in zip(unique_labels, class_precision, class_recall, class_f1):
        print(f"  {label}:")
        print(f"    Precision: {p*100:.4f}")
        print(f"    Recall: {r*100:.4f}")
        print(f"    F1 Score: {f*100:.4f}")
    
    print("\nConfusion Matrix:")
    print(cm)

if __name__ == "__main__":
    main()
