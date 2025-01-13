import warnings
warnings.filterwarnings('ignore')
from ultralytics import RTDETR

if __name__ == '__main__':
    model = RTDETR('runs/train/rtdetr-r18/weights/best.pt')
    # model = RTDETR('ultralytics/cfg/models/rt-detr/rtdetr-RGCSPELAN-CascadedGroupAttention-DGCST-CGAFusion.yaml')
    model.val(data='dataset/mydata/mydata.yaml',
              split='val',
              imgsz=640,
              batch=4,  #在使用val.py测试运行速度时设置batch=1
              save_json=True, # if you need to cal coco metrice
              project='runs/val',
              name='exp',
              )