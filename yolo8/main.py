from ultralytics import YOLO


# DO additional training

import os

model = YOLO("yolov8s-cls.pt", task="classify")  # Load a pre-trained YOLOv10s model
model.train(data="yolo8/imagewang-320", epochs=10, imgsz=320)  # Train the model by new datasets

model.save("yolov8s_trained.pt")  # Save the trained model


# # Prediction
# model = YOLO("yolov8s-cls.pt")
#
# # Perform object detection on an image
# results = model.predict(source="test-img.jpg", task="classify")
#
# # Display the results
# results[0].show()





