### 预处理图像，并构建整个数据集

import os
import json
import shutil

import cv2
import numpy as np

PICTURE_FORMATS = ['jpg', 'bmp']
PATH_INDEX = "CT/advanced"
LESSION_TYPE_LABELS = ["no_lesions", "advanced_lesions"]

# USER_PROMPT = "<image>\nThe picture is an endocopy medical image of the esophagus. Outline the potential lession area with 4 coordinates(Each represents the percentage distance between the four vertices of the area and the left and top edges of the image), and answer which is the most possible label of this image: [no lesions, precancerous lesions]"
USER_PROMPT = "<image>\nThe picture is an CT medical image of the esophagus. Answer from the perspective of esophagus, which is the most possible label of this image: [no_lesions, advanced_lesions]"
# ASSISTANT_RESPONSE = "Area: {}.\nLabel: {}."
ASSISTANT_RESPONSE = "Label: {}."


def check_laebl(filename, labeled_data_path):
    """
    Check if the image has a label, 
    i.e. if there is a corresponding label file in the labeled data path.
    """
    for dirpath, dirnames, filenames in os.walk(labeled_data_path):
        if len(dirnames) == 0:
            # for file in filenames:
            for form in PICTURE_FORMATS:
                if filename+'.'+form in filenames:
                    return os.path.join(dirpath, filename+'.'+form)
                    # return True
    return False

def make_data_item(dirpath, filename, labeled_data_path, tgt_img_path):
    label_exist = check_laebl(filename.split('.')[0], labeled_data_path)
    # print(">>> Label exist:", label_exist)
    item_dict = {}
    item_dict['id'] = filename.split('.')[0]
    item_dict['image'] = tgt_img_path + '/' + filename
    shutil.copy(dirpath + '/' + filename, tgt_img_path)
    item_dict['conversations'] = [{"from": "human", "value": USER_PROMPT}]
    item_dict['conversations'].append({
        "from": "gpt",
        "value": ASSISTANT_RESPONSE.format(LESSION_TYPE_LABELS[1]) 
        if label_exist
        else ASSISTANT_RESPONSE.format(LESSION_TYPE_LABELS[0])
    })
    return item_dict

def build_dataset(unlabel_data_path, labeled_data_path, output_path):
    os.makedirs(output_path, exist_ok=True)
    tgt_img_path = output_path + '/images'
    os.makedirs(tgt_img_path, exist_ok=True)

    data_list = []
    for dirpath, dirnames, filenames in os.walk(unlabel_data_path):
        # print("Dirpath, dirnames, filenames:", dirpath, dirnames, filenames)
        if len(dirnames) == 0:
            for filename in filenames:
                print(">>> Processing:", dirpath+'/'+filename)
                item_dict = make_data_item(dirpath, filename, labeled_data_path, tgt_img_path)
                data_list.append(item_dict)
    with open(output_path + '/full_data.json', 'w') as f:
        json.dump(data_list, f, indent=4, ensure_ascii=False)   # ensure_ascii=False: 保证中文字符不被转义


if __name__ == '__main__':
    raw_data_path = '/date/jc/data/Eso-Llava/raw_data/' + PATH_INDEX
    unlabel_data_path = raw_data_path + '/unlabeled'
    labeled_data_path = raw_data_path + '/labeled'
    output_path = '/date/jc/data/Eso-Llava/processed_data/' + PATH_INDEX
    build_dataset(unlabel_data_path, labeled_data_path, output_path)
