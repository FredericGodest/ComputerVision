# Computer Vision with Yolo
This side project has been done with Python, OpenCV, Matplotlib and (of course) YOLOV3.

## What do you need ?
First you need to have Python 3.X (.7 is good).
Then you will need to "pip install" OpenCV with this command line:

* pip install opencv-python

Then you will need Numpy, Matplotlib:

* pip install numpy
* pip install matplotlib

Finally you will need the config and the weights file of the pre-trained YOLOV3 models.
These files are available here :

https://pjreddie.com/darknet/yolo/

In this project I used YOLOv3-320 because is it way better than YOLOv3-tiny in terms of 
object detection quality even though the FPS dropped down from 15-20 to barely 1 FPS (with a classic CPU).
I highly recommend to use this code with a GPU.