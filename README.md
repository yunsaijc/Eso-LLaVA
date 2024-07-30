# Esophagus cancer LLM

胃镜类型：[NBI, 白光, 碘染, 超声胃镜]

输出类型：[无病变, 癌前病变]

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
