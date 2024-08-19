import json
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from shapely.geometry import Polygon

def parse_prediction(prediction):
    # print("prediction:", prediction)
    try: label = prediction.split(":")[1].strip().lower()
    except: label = 'no_lesions'
    if 'pre' in label: label = 'precancerous_lesions.'
    return label


def main():
    date = "240818"
    version = "e2.epoch3"
    # base_model = "llava-med-v1.5-mistral-7b"
    base_model = "llava-v1.6-vicuna-13b"
    # base_model = "llava-v1.6-mistral-7b"
    result_path = "/home/jc/workspace/MedLLMs/Eso-Llava/eval/results/"
    inference_result_path = result_path + f"{date}{version}-lora-{base_model}-merged-test.json"
    
    with open(inference_result_path, 'r') as f:
        results = json.load(f)
    
    true_labels = []
    pred_labels = []
    
    for item in results:
        pred_label = parse_prediction(item['answer'])
        true_label = parse_prediction(item['target'])
        
        true_labels.append(true_label)
        pred_labels.append(pred_label)
    
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
    
    
    print(f"Overall Accuracy: {accuracy*100:.4f}")
    print(f"Overall Precision: {precision*100:.4f}")
    print(f"Overall Recall: {recall*100:.4f}")
    print(f"Overall F1 Score: {f1*100:.4f}")
    
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
