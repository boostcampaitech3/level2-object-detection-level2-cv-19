# Object-detection-level2-cv-19
<br />

## ๐ Overview
### Background
> ๋ฐ์ผํ๋ก ๋๋ ์์ฐ, ๋๋ ์๋น์ ์๋. ์ฐ๋ฆฌ๋ ๋ง์ ๋ฌผ๊ฑด์ด ๋๋์ผ๋ก ์์ฐ๋๊ณ , ์๋น๋๋ ์๋๋ฅผ ์ด๊ณ  ์์ต๋๋ค. ํ์ง๋ง ์ด๋ฌํ ๋ฌธํ๋ '์ฐ๋ ๊ธฐ ๋๋', '๋งค๋ฆฝ์ง ๋ถ์กฑ'๊ณผ ๊ฐ์ ์ฌ๋ฌ ์ฌํ ๋ฌธ์ ๋ฅผ ๋ณ๊ณ  ์์ต๋๋ค.

<img src='https://camo.githubusercontent.com/86c9fd66258daf9bcaee570f6024589839ca5dfc4efaeaa29f97c9fb82b819a3/68747470733a2f2f692e696d6775722e636f6d2f506e4f6451304c2e706e67' />

> ๋ถ๋ฆฌ์๊ฑฐ๋ ์ด๋ฌํ ํ๊ฒฝ ๋ถ๋ด์ ์ค์ผ ์ ์๋ ๋ฐฉ๋ฒ ์ค ํ๋์๋๋ค. ์ ๋ถ๋ฆฌ๋ฐฐ์ถ ๋ ์ฐ๋ ๊ธฐ๋ ์์์ผ๋ก์ ๊ฐ์น๋ฅผ ์ธ์ ๋ฐ์ ์ฌํ์ฉ๋์ง๋ง, ์๋ชป ๋ถ๋ฆฌ๋ฐฐ์ถ ๋๋ฉด ๊ทธ๋๋ก ํ๊ธฐ๋ฌผ๋ก ๋ถ๋ฅ๋์ด ๋งค๋ฆฝ ๋๋ ์๊ฐ๋๊ธฐ ๋๋ฌธ์๋๋ค.  
> 
> ๋ฐ๋ผ์ ์ฐ๋ฆฌ๋ ์ฌ์ง์์ ์ฐ๋ ๊ธฐ๋ฅผ Detection ํ๋ ๋ชจ๋ธ์ ๋ง๋ค์ด ์ด๋ฌํ ๋ฌธ์ ์ ์ ํด๊ฒฐํด๋ณด๊ณ ์ ํฉ๋๋ค. ๋ฌธ์  ํด๊ฒฐ์ ์ํ ๋ฐ์ดํฐ์์ผ๋ก๋ ์ผ๋ฐ ์ฐ๋ ๊ธฐ, ํ๋ผ์คํฑ, ์ข์ด, ์ ๋ฆฌ ๋ฑ 10 ์ข๋ฅ์ ์ฐ๋ ๊ธฐ๊ฐ ์ฐํ ์ฌ์ง ๋ฐ์ดํฐ์์ด ์ ๊ณต๋ฉ๋๋ค.  
> 
> ์ฌ๋ฌ๋ถ์ ์ํด ๋ง๋ค์ด์ง ์ฐ์ํ ์ฑ๋ฅ์ ๋ชจ๋ธ์ ์ฐ๋ ๊ธฐ์ฅ์ ์ค์น๋์ด ์ ํํ ๋ถ๋ฆฌ์๊ฑฐ๋ฅผ ๋๊ฑฐ๋, ์ด๋ฆฐ์์ด๋ค์ ๋ถ๋ฆฌ์๊ฑฐ ๊ต์ก ๋ฑ์ ์ฌ์ฉ๋  ์ ์์ ๊ฒ์๋๋ค. ๋ถ๋ ์ง๊ตฌ๋ฅผ ์๊ธฐ๋ก๋ถํฐ ๊ตฌํด์ฃผ์ธ์! ๐

### Dataset
* ์ ์ฒด ์ด๋ฏธ์ง ๊ฐ์: 9754์ฅ (Train 4883์ฅ, Test 4871์ฅ)
* 10 class: General trash, Paper, Paper pack, Metal, Glass, Plastic, Styrofoam, Plastic bag, Battery, Clothing
* ์ด๋ฏธ์ง ํฌ๊ธฐ: (1024, 1024)
* Annotation file: coco format

### ํ๊ฐ ๋ฐฉ๋ฒ
* Test set์ mAP50(Mean Average Precision)๋ก ํ๊ฐ
* Pascal VOC ํฌ๋งท ์ฌ์ฉ
* PredictionString = (label, score, xmin, ymin, xmax, ymax), ......


<br />

## ๐ Members
- `๊ถํ์ฐ` &nbsp; Detection์ ์ํ ๋ผ์ด๋ธ๋ฌ๋ฆฌ ์์นญ, ๋ผ์ด๋ธ๋ฌ๋ฆฌ ์ฌ์ฉ๋ฒ ๋ฐ ํ์์ ๋ํ ์๊ฒฌ ์ ์, DataCleaning, IOU Sweep 
- `๊น๋์ ` &nbsp; EDA ๋ฐ fiftyone์ ์ด์ฉํ ๋ฐ์ดํฐ ์๊ฐํ, ๋ชจ๋ธ ๊ฒฐ๊ณผ ๋ฐ์ดํฐ ํ์ ๋ณ๊ฒฝ ํด(csv to json), stratified k-fold ๋ฐ์ดํฐ ์ ์   
- `๊น์ฐฌ๋ฏผ` &nbsp; ๋ฒ ์ด์ค๋ผ์ธ์ฝ๋ ๋ชจ๋ธ ์ดํด ๋ฐ ํ์ต  
- `์ด์์ง` &nbsp; mmdetection, k-fold ๋ฐ์ดํฐ์, ensemble, multi-scale  
- `์ ํจ์ฌ` &nbsp; ๋ชจ๋ธ ํ์ต ๋ฐ Augmentation, workspace ์์ฑ  

<br />

## ๐ Code Structure
```
.
โโโ yolov5
โ   โโโ data
โ   โโโ train.py
โ   โโโ detect.py
โ   โโโ val.py
โโโ mmdetection
โ   โโโ tools
โ   โ   โโโ train.py
โ   โโโ custom_models
โ       โโโ cascade_swin
โ       โโโ cascadeB_swin
โ       โโโ faster_rcnn_r101_fpn_1x
โโโ dataset
โโโ data_research
    โโโ EDA
    โ   โโโ DataAnalysis.ipynb
    โ   โโโ fiftyone.ipynb
    โ   โโโ dataview.py
    โโโ nms.ipynb
    โโโ inference_csvtojson.ipynb
    โโโ inference_jsontocsv.ipynb
    โโโ resize_submission
        โโโ resize_bbox.py
```


<br />

## ๐ป How to use
### mmdetection
```
cd mmdetection
python tools/train.py custom_models/{ํ์ต์ํค๊ณ  ์ถ์ ๋ชจ๋ธ๋ช}/model.py
```

### yolov5
```
./dataset/images/ ์ test ์ด๋ฏธ์ง ํด๋์ train ์ด๋ฏธ์ง ํด๋๋ฅผ ๋ฃ์ด์ค๋ค
python train.py --entity next_level --batch 16 --epochs 30 --data trash.yaml --weights yolov5s.pt 
```

### EfficientDet
```

```

### Detectron2
```
cd Detectron2
python tools/train_net.py --config-file {Config ํ์ผ๋ช}
```

<br />

## Evaluation
### Models
|model name|epoch|submit mAP|K-Fold|Training|
| --- | --- | --- | --- | --- |
|yolo5xl6|150|0.5739||```TTA```|
|EfficientDet|50|0.3598||```SGD``` ```D3```|
|yolov5l6|200||||
|faster rcnn r101 fpn 1x|12|0.4028||```AdamW```|
|cascade swin|32|0.5644|0.5963|```AdamW```|
|cascadeB swin|40|0.6789|||
|UniverseNet|100|0.5564|||

### Ensemble
* soft-NMS
* Threshold 0.52

```
โ Final Score
  - Public LB score: mAP 0.6824
  - Private LB score: mAP 0.6663
```



<!-- ### yolov5 ๋ฐ์ดํฐ ๊ด๋ จ
./dataset/imaages/ ์ test ์ด๋ฏธ์ง ํด๋์ train ์ด๋ฏธ์ง ํด๋๋ฅผ ๋ฃ์ด์ฃผ๊ณ  ํ์ตํ๋ฉด๋จ 

---
### yolov5 wandb ํ์ฑํ ํ๋ ๋ฐฉ๋ฒ
termial์์ yolo5ํด๋ ๋ค์ด๊ฐ์ wandb online ์คํ

python train.py --entity next_level --batch 16 --epochs 30 --data trash.yaml --weights yolov5s.pt  -->
