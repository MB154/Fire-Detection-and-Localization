# Fire-Detection-and-Localization
Fire detection and localization using machine learning models.
Fire detection using Yolov5  and Midas depth for distance estimation.
The Yolov5 will detect fire and calculate the bouding box coordinates.
The Midas depth will calculate the depth map and will use refrence points to to derive absolute depth and then by using coordinates, calculate distance to fire.

####Install
Clone this repo

Yolov5 Fire detection model is provided in models
To download the Midas Models use the following links:
For the dpt_beit_large_512: https://github.com/islorg/MiDaS/releases/download/v3_1/dpt_beit_large_512.pt
For the midas_v21_small_256:
https://github.com/isl-org/MiDaS/releases/download/v2_1/midas_v21_small_256.pt

For requirements:
Check requirements for general requirements. Addittionally check for extra requirements provided in /yolov5 folder "requirements.txt" and in /MiDaS folder "environment.yaml" if faced with a problem.
For extra help you can check the README of Yolov5 and Midas, they provide information about the code and can be helpful for modifying the code.

To run the code use the FireD&L.ipynb
To change the refrence points or the email, you can use the UI in FireD&L.ipynb, make sure you have the Refrence_Points.txt file in /yolov5 folder. If not found create an empty txt file with same name.
Run the UI code and press open configutation then select the Refrence_Points.txt
Add, modify, delete the refrence points and change the email. When done press save configuration.

Run the inference code, --source: (0 for camera, or video path),  --weights path to yolov5 fire detection model weights; --data path to fire_config.yaml; --midas: (0 for  midas_v21_small_256 and 1 for dpt_beit_large_512) by default with including this parameter the the small model is used; --conf 0.29 (fire model confidence threshold); --half for using half precision floating point for the fire detection; --vid-stride 2 .
Example: !python detect.py --source ../VidDemo.mp4 --weights ../models/best.pt --data ../fire_config.yaml --midas-model 0  --conf 0.29 --view-img  --half --vid-stride 2

