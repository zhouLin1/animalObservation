from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('trained.pt')

    source = 'C:\\Users\\Sam\\Desktop\\orangutanyolo\\testvideos\\'

    results = model.predict(source, save=True, device=0)


    for result in results:
        boxes = result.boxes  # Boxes object for bbox outputs
        masks = result.masks  # Masks object for segmentation masks outputs
        keypoints = result.keypoints  # Keypoints object for pose outputs
        probs = result.probs  # Probs object for classification outputs