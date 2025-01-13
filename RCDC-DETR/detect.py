import warnings
warnings.filterwarnings('ignore')
from ultralytics import RTDETR

if __name__ == '__main__':
    model = RTDETR('runs/train/rtdetr-r18/weights/best.pt') # select your model.pt path
    model.predict(source='dataset/mydata/images/val',
                  project='runs/detect',
                  name='exp',
                  save=True,
                  visualize=True # visualize model features maps
                  )