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
Check requirements for general requirements. Addittionally check yolov5 and MiDaS requirements provided.
For extra help you can check the README of Yolov5 and Midas.

To run the code use the FireD&L.ipynb
