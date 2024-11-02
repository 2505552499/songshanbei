from ultralytics import YOLO

model = YOLO(r'runs/detect/train15/weights/best.pt')

# 对验证集进行评估
metrics = model.val(data=r'Myyaml.yaml',workers=0)