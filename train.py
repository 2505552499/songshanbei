from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("ultralytics/cfg/models/v10/yolov10s.yaml", task="detect")
    # model = YOLO("ultralytics/yolov8n.pt")

    model.train(data='Myyaml.yaml', epochs=100, device=0, workers=0,batch=32)