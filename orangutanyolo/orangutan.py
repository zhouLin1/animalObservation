from ultralytics import YOLO

if __name__ == '__main__':
    # TRAINING
    # Load a model
    model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)

    # Train the model
    results = model.train(data="C:\\Users\\Sam\\Desktop\\orangutanyolo\\datasets\\orangutan\\data.yaml", epochs=200, imgsz=1024, device=0)

    # Validate the model
    metrics = model.val()

    # DETECTION
    # Define path to the image file
    source = 'C:\\Users\\Sam\\Desktop\\orangutanyolo\\datasets\\orangutan\\test\\images'

    # Run inference on the source
    results = model(source)  # list of Results objects

    for result in results:
        boxes = result.boxes  # Boxes object for bbox outputs
        masks = result.masks  # Masks object for segmentation masks outputs
        keypoints = result.keypoints  # Keypoints object for pose outputs
        probs = result.probs  # Probs object for classification outputs