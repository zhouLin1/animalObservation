# animalObservation
Animal observation using AI for topics in computer science. 

This repository contains my work for topics in computer science. trained.pt is the trained model on the annotated imgaes included in /datasets/orangutan. 
yolov8n.pt is the pretrained model from ultralytics, further trained on the annotated data.
orangutan.py is the python code to train the model. 
test.py is the code to run the trained model on new videos.

Please change the path in the .py files if you wish to train the model again or run the model on your own videos.
the runs folder has results from running the model on videos and training the model. 

To create data, we used roboflow but any data in the correct format will work.
You can use the trained model (trained.pt), and train it on new data to avoid retraining the pretrained model.
