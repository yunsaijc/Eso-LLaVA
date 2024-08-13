# Esophagus cancer LLM

## Data Intro
胃镜类型：[NBI, 白光, 碘染, 超声胃镜]

输出类型：[无病变, 癌前病变, 中晚期病变]
["no_lesions", "precancerous_lesions", "advanced_lesions"]

```python
example_dataset_content = 
[
    {
        "id": "yuanshen-628d-4724-b370-b84de974a19f",
        "image": "yuanshen/1.jpg",
        "conversations": [
            {
                "from": "human",
                "value": "<image>\nThe picture is an endocopy medical image of the esophagus. Outline the potential lession area with 4 coordinates(Each represents the percentage distance between the four vertices of the area and the left and top edges of the image), and answer which is the most possible label of this image: [no_lesions, precancerous_lesions]"
            },
            {
                "from": "gpt",
                "value": "Area: [[(46.04, 2.75), (46.04, 39.23), (94.37, 39.04), (94.37, 2.75)]].\nLabel: precancerous lesions."
            }
        ]
    }
]
```

## Pipeline

### Data Preparation

For endoscopy images:
1. Run ./scripts/filter_endo.py to filter out the images that are not endoscopy images.
2. Run ./scripts/preprocess.py to preprocess the images, matching labels to the images and build a full dataset.
3. Run ./scripts/split_set.py to split the dataset.

### Model Training

1. Run ./sft.sh to train the model.

The checkpoints will be saved in ./checkpoints.

### Model Evaluation

1. Run ./eval/evaluate.py to generate responses for the test data.
2. Run ./eval/score.py to score the generated results.
