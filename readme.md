### yolov5 데이터 관련
./dataset/imaages/ 에 test 이미지 폴더와 train 이미지 폴더를 넣어주고 학습하면됨 

---
### yolov5 wandb 활성화 하는 방법
termial에서 yolo5폴더 들어가서 wandb online 실행

python train.py --entity next_level --batch 16 --epochs 30 --data trash.yaml --weights yolov5s.pt 
