### 预处理图像，并构建整个数据集
# TODO: 是否将bmp格式转换为jpeg格式？

import os
import json
import shutil

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

PICTURE_FORMATS = ['jpeg', 'bmp']
PATH_INDEX = "endo/precancer"
LESSION_TYPE_LABELS = ["no_lesions", "precancerous_lesions"]

# USER_PROMPT = "<image>\nThe picture is an endocopy medical image of the esophagus. Outline the potential lession area with 4 coordinates(Each represents the percentage distance between the four vertices of the area and the left and top edges of the image), and answer which is the most possible label of this image: [no lesions, precancerous lesions]"
USER_PROMPT = "<image>\nThe picture is an endocopy medical image of the esophagus. Answer from the perspective of esophagus, which is the most possible label of this image: [no_lesions, precancerous_lesions]"
# ASSISTANT_RESPONSE = "Area: {}.\nLabel: {}."
ASSISTANT_RESPONSE = "Label: {}."


SIM_list = []
def check_laebl(file_path, filename_without_ext, labeled_data_path):
    """
    Check if the image has a label, 
    i.e. if there is a corresponding label file in the labeled data path.
    file_path: the abs path of the image file
    filename: the name of the image file without the extension
    """
    for dirpath, dirnames, filenames in os.walk(labeled_data_path):
        if len(dirnames) == 0:
            for file in filenames:
                if filename_without_ext in file:
                    print(">>>>>")
                    print("file_without_ext: ", filename_without_ext)
                    print("Original image: ", file_path)
                    print("Annotated image: ", dirpath+'/'+file)
                    original_gray = cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2GRAY)
                    annotated_gray = cv2.cvtColor(cv2.imread(dirpath+'/'+file), cv2.COLOR_BGR2GRAY)
                    try: (score, diff) = ssim(original_gray, annotated_gray, full=True)
                    except ValueError: continue
                    diff = (diff * 255).astype("uint8")

                    print("SSIM: {}".format(score))
                    SIM_list.append((score, file_path, dirpath+'/'+file))
                    if score == 1.0 or score <= 0.95:
                        return False
                    elif score > 0.95:
                        return os.path.join(dirpath, file)
    return False

def match_label(file_path, filename_without_ext, labeled_data_path):
    """
    Match the label file to the image file.
    If the label file is not found, return None.
    """
    matched_label_path = check_laebl(file_path, filename_without_ext, labeled_data_path)
    if matched_label_path:
        # print(">>> Label found.")
        return get_blue_box_coordinates(matched_label_path)
    else: return None

def get_blue_box_coordinates(image_path):
    image = cv2.imread(image_path)
    if image is None: raise ValueError("Image not found or unable to read")
    
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)    # 转换为HSV颜色空间，便于颜色检测
    lower_blue = np.array([100, 50, 50])            # 定义蓝色的HSV范围
    upper_blue = np.array([130, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue) # 创建蓝色掩码
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # 查找轮廓
    if not contours: 
        print(">>> Error: No blue contours found in this image.")
        return None

    height, width = image.shape[:2]
    all_box_coordinates = []
    for contour in contours:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        # 只处理四边形（方框）
        if len(approx) == 4:
            points_percent = []
            for point in approx:
                x, y = point[0]
                x_percent = round((x / width) * 100, 2)
                y_percent = round((y / height) * 100, 2)
                points_percent.append((float(x_percent), float(y_percent)))
            all_box_coordinates.append(points_percent)
        else:
            print(f">>> Warning: A contour with {len(approx)} vertices found. Skipping.")

    if not all_box_coordinates:
        print(">>> Error: No valid blue boxes found in this image.")
        return None
    return all_box_coordinates

def make_data_item(dirpath, filename, labeled_data_path, tgt_img_path):
    filename_without_ext = filename.split('.')[0]
    # coords = match_label(dirpath+'/'+filename, filename_without_ext, labeled_data_path)
    coords = check_laebl(dirpath+'/'+filename, filename_without_ext, labeled_data_path)
    item_dict = {}
    item_dict['id'] = filename.split('.')[0]
    item_dict['image'] = tgt_img_path + '/' + filename
    # shutil.copy(dirpath + '/' + filename, tgt_img_path)
    item_dict['conversations'] = [{"from": "human", "value": USER_PROMPT}]
    if coords:
        item_dict['conversations'].append({
                "from": "gpt",
                # "value": ASSISTANT_RESPONSE.format(coords, LESSION_TYPE_LABELS[1])
                "value": ASSISTANT_RESPONSE.format(LESSION_TYPE_LABELS[1])
        })
    else:
        item_dict['conversations'].append({
                "from": "gpt",
                # "value": ASSISTANT_RESPONSE.format([], LESSION_TYPE_LABELS[0])
                "value": ASSISTANT_RESPONSE.format(LESSION_TYPE_LABELS[0])
        })
    return item_dict

def build_dataset(unlabel_data_path, labeled_data_path, output_path):
    os.makedirs(output_path, exist_ok=True)
    tgt_img_path = output_path + '/images'
    os.makedirs(tgt_img_path, exist_ok=True)

    data_list, image_list = [], []
    for dirpath, dirnames, filenames in os.walk(unlabel_data_path):
        if len(dirnames) == 0:  # 如果已经是最底层目录
            for filename in filenames:
                print(">>> Processing:", dirpath+'/'+filename)
                item_dict = make_data_item(dirpath, filename, labeled_data_path, tgt_img_path)
                if item_dict['image'] in image_list:    # 防止重复图片
                    print(">>> Error: Duplicate image found.")
                    count = 0
                    for item in image_list:
                        if '(' in item: existed_filename_without_ext = item.split('(')[0].strip()
                        else: existed_filename_without_ext = item.split('.')[0]
                        if existed_filename_without_ext == filename.split('.')[0]:
                            count += 1
                    # count = sum(1 for item in image_list if filename.split('.')[0] == item) # 计算重复次数
                    item_dict['image'] = item_dict['image'].split('.')[0] + f'({count}).' + item_dict['image'].split('.')[1]
                image_list.append(item_dict['image'])
                data_list.append(item_dict)
                shutil.copy(dirpath+'/'+filename, item_dict['image'])
    with open(output_path + '/full_data.json', 'w') as f:
        json.dump(data_list, f, indent=4, ensure_ascii=False)   # ensure_ascii=False: 保证中文字符不被转义


if __name__ == '__main__':

    # get_blue_box_coordinates 示例
    # image_path = '/home/jc/workspace/MedLLMs/Eso-Llava/data/raw_data/endo/precancer/labeled/侯娇娇/陈秀云_女_1144134_195401120000__1144134_20240123/1144134_陈秀云_195401120000_女_1.jpeg'
    # try:
    #     coordinates = get_blue_box_coordinates(image_path)
    #     print("Blue box coordinates in percentage:", coordinates)
    # except Exception as e:
    #     print(f"Error: {e}")

    # build_dataset 示例
    raw_data_path = '/date/jc/data/Eso-Llava/raw_data/' + PATH_INDEX
    unlabel_data_path = raw_data_path + '/unlabeled'
    labeled_data_path = raw_data_path + '/labeled'
    output_path = '/date/jc/data/Eso-Llava/processed_data/' + PATH_INDEX
    build_dataset(unlabel_data_path, labeled_data_path, output_path)

    SIM_list.sort(key=lambda x: x[0], reverse=True)
    print(">>> SSIM list:", SIM_list)
    # print("Max SSIM:", max(SIM_list))
    # print("Min SSIM:", min(SIM_list))
    # # print("Mean SSIM:", sum(SIM_list)/len(SIM_list))
    with open("/home/jc/workspace/MedLLMs/Eso-Llava/data/scripts/endo/SSIM_list.txt", 'w') as f:
        f.write(str(SIM_list)+'\n')

# def get_blue_box_coordinates(image_path):
#     # 读取图片
#     image = cv2.imread(image_path)
#     if image is None: raise ValueError("Image not found or unable to read")
    
#     hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)    # 转换为HSV颜色空间，便于颜色检测
#     lower_blue = np.array([100, 50, 50])            # 定义蓝色的HSV范围
#     upper_blue = np.array([130, 255, 255])
#     mask = cv2.inRange(hsv, lower_blue, upper_blue) # 创建蓝色掩码

#     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # 查找轮廓
#     if not contours: 
#         # raise ValueError("No blue contours found.")
#         print(">>> Error: No blue contours found in this image.")
#         return None
#     largest_contour = max(contours, key=cv2.contourArea)  # 找到最大的轮廓（假设它是蓝色方框）

#     epsilon = 0.02 * cv2.arcLength(largest_contour, True) # 获取轮廓的四个顶点
#     approx = cv2.approxPolyDP(largest_contour, epsilon, True)

#     if len(approx) != 4:    # 确保轮廓有4个顶点
#         print(">>> Error: This image's contour does not have exactly 4 vertices.")
#         return None
#         # raise ValueError("The contour does not have exactly 4 vertices.")
#     # 将顶点转换为百分比坐标
#     height, width = image.shape[:2]
#     points_percent = []
#     for point in approx:
#         x, y = point[0]
#         x_percent = round((x / width) * 100, 2)
#         y_percent = round((y / height) * 100, 2)
#         points_percent.append((float(x_percent), float(y_percent)))
#     return points_percent