# Object-detection-level2-cv-19
### 🔍 Overview
#### Background
![image1](./mmdetection/tools/image_/image1.png)

#### Dataset
* 전체 이미지 개수: 9754장 (Train 4883장, Test 4871장)
* 10 class: General trash, Paper, Paper pack, Metal, Glass, Plastic, Styrofoam, Plastic bag, Battery, Clothing
* 이미지 크기: (1024, 1024)
* Annotation file: coco format

#### 평가 방법
* Test set의 mAP50(Mean Average Precision)로 평가
* Pascal VOC 포맷 사용
* PredictionString = (label, score, xmin, ymin, xmax, ymax), ......


<br />

## 📝 Members
- `권혁산` &nbsp; WarpUp 리포트에 
- `김대유` &nbsp; 프로젝트 팀   
- `김찬민` &nbsp; 구성 및 역할  
- `이상진` &nbsp; 부분 그대로  
- `정효재` &nbsp; 써주시면 되요  

<br />

## 📃 Code Structure
```
.
├── yolov5
│   ├── data
│   ├── train.py
│   ├── detect.py
│   └── val.py
├── mmdetection
│   ├── train.py
│   │   └── train.py
│   └── custom_models
│       ├── cascade_swin
│       ├── cascadeB_swin
│       └── faster_rcnn_r101_fpn_1x
├── dataset
└── data_research
    ├── EDA
    │   ├── DataAnalysis.ipynb
    │   ├── fiftyone.ipynb
    │   └── dataview.py
    ├── nms.ipynb
    ├── inference_csvtojson.ipynb
    ├── inference_jsontocsv.ipynb
    └── resize_submission
        └── resize_bbox.py
```


<br />

## 💻 How to use
#### mmdetection
```
cd mmdetection
python tools/train.py custom_models/{학습시키고 싶은 모델명}/model.py
```

#### yolov5
```
./dataset/images/ 에 test 이미지 폴더와 train 이미지 폴더를 넣어준다
python train.py --entity next_level --batch 16 --epochs 30 --data trash.yaml --weights yolov5s.pt 
```

#### EfficientDet
```

```

#### Faster R-CNN
```

```

<br />

## Evaluation
#### Models
|model name|epoch|submit mAP|K-Fold|Training|
| --- | --- | --- | --- | --- |
|yolo5xl6|150|0.5739||```TTA```|
|EfficientDet|50|0.3598||```SGD``` ```D3```|
|yolov5l6|200||||
|faster rcnn r101 fpn 1x|12|0.4028||```AdamW```|
|cascade swin|32|0.5644|0.5963|```AdamW```|
|cascadeB swin|40|0.6789|||
|UniverseNet|100|0.5564|||

#### Ensemble
* soft-NMS
* Threshold 0.52

```
● Final Score
  - Public LB score: mAP 0.6824
  - Private LB score: mAP 0.6663
```



<!-- ### yolov5 데이터 관련
./dataset/imaages/ 에 test 이미지 폴더와 train 이미지 폴더를 넣어주고 학습하면됨 

---
### yolov5 wandb 활성화 하는 방법
termial에서 yolo5폴더 들어가서 wandb online 실행

python train.py --entity next_level --batch 16 --epochs 30 --data trash.yaml --weights yolov5s.pt  -->
