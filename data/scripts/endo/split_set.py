### 对构建好的整个数据集进行划分，划分为训练集、验证集和测试集
# 将所有数据划分为正常和异常两类，每个子集中正常和异常的比例尽量保持一致
# 训练、验证、测试集的比例是70%、15%、15%

## v1 News: v1训练得到的模型，只会输出no_lession. v2减少no_lession的数量，大概1:1
## v2 News: train, test 改成 7: 3，不要val
## v3: train: test = 8: 2
## v4: train: test = 9: 1

import os
import json
import random

from sklearn.model_selection import train_test_split


full_data_path = '/date/jc/data/Eso-Llava/processed_data/' + "endo/precancer"
VERSION = 4
test_size = 0.1

def classify_data(full_data):
    """Classify data into normal and abnormal."""
    normal_data = []
    abnormal_data = []
    for item in full_data:
        if 'precancerous' in item['conversations'][1]['value']:
            abnormal_data.append(item)
        else:
            normal_data.append(item)
    print(">>> Normal data count:", len(normal_data))
    print(">>> Abnormal data count:", len(abnormal_data))
    return normal_data, abnormal_data

def random_split(normal_data, abnormal_data):
    # 划分训练集、验证集和测试集
    # normal_train, normal_temp = train_test_split(normal_data, test_size=0.3, random_state=42)
    # normal_val, normal_test = train_test_split(normal_temp, test_size=0.5, random_state=42)
    normal_train, normal_test = train_test_split(normal_data, test_size=test_size, random_state=42)

    # abnormal_train, abnormal_temp = train_test_split(abnormal_data, test_size=0.3, random_state=42)
    # abnormal_val, abnormal_test = train_test_split(abnormal_temp, test_size=0.5, random_state=42)
    abnormal_train, abnormal_test = train_test_split(abnormal_data, test_size=test_size, random_state=42)

    train_data = normal_train + abnormal_train
    # val_data = normal_val + abnormal_val
    test_data = normal_test + abnormal_test
    random.shuffle(train_data)
    # random.shuffle(val_data)
    random.shuffle(test_data)

    print(">>> Train data count:", len(train_data))
    # print(">>> Validation data count:", len(val_data))
    print(">>> Test data count:", len(test_data))

    return train_data, test_data # val_data, 

if __name__ == '__main__':
    with open(full_data_path + '/full_data.json', 'r') as f:
        full_data = json.load(f)

    normal_data, abnormal_data = classify_data(full_data)
    # 保持正常和异常数据的数量一致
    normal_data = random.sample(normal_data, len(abnormal_data))    

    train_data, test_data = random_split(normal_data, abnormal_data)

    os.makedirs(full_data_path + f'/v{VERSION}', exist_ok=True)
    with open(full_data_path + f'/v{VERSION}/train.json', 'w') as f:
        json.dump(train_data, f, indent=4, ensure_ascii=False)
    # with open(full_data_path + '/val.json', 'w') as f:
    #     json.dump(val_data, f, indent=4, ensure_ascii=False)
    with open(full_data_path + f'/v{VERSION}/test.json', 'w') as f:
        json.dump(test_data, f, indent=4, ensure_ascii=False)
    


# train_norm_count = int(len(normal_data) * 0.7)
# train_abnorm_count = int(len(abnormal_data) * 0.7)
# val_norm_count = int(len(normal_data) * 0.15)
# val_abnorm_count = int(len(abnormal_data) * 0.15)
# test_norm_count = len(normal_data) - train_norm_count - val_norm_count
# test_abnorm_count = len(abnormal_data) - train_abnorm_count - val_abnorm_count

# train_data_norm = random.sample(normal_data, train_norm_count)
# train_data_abnorm = random.sample(abnormal_data, train_abnorm_count)
# normal_data = list(set(normal_data) - set(train_data_norm))
# abnormal_data = list(set(abnormal_data) - set(train_data_abnorm))

# val_data_norm = random.sample(normal_data, val_norm_count)
# val_data_abnorm = random.sample(abnormal_data, val_abnorm_count)
# normal_data = list(set(normal_data) - set(val_data_norm))
# abnormal_data = list(set(abnormal_data) - set(val_data_abnorm))

# test_data_norm = normal_data
# test_data_abnorm = abnormal_data

# train_data = train_data_norm + train_data_abnorm
# val_data = val_data_norm + val_data_abnorm
# test_data = test_data_norm + test_data_abnorm
