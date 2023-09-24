# Yolov7 + ByteTrack

Here are some instructions to set up the tracking to the Yolov7 (detection framework):

## track_direction.py
The track_direction.py script is used for only ONE folder with a sequence of frames. For example if a sequence of frames are enclosed within a folder named 'uav_xxxxxxxxxx', then the script will run for all images within that folder. Set the **source** and the pre-trained **weights** as the argumented to running the script in a terminal.

## track_direction_multiple.py
The track_direction_multiple.py script is used for multiple folders with sequences of frames. This entire script can be executed once for different folders capturing different aerial scenes. The **source** and the **weights** will need to set as before.

## track_direction_fisheye.py
The track_direction_fisheye.py script is used for only ONE folder with a sequence of fisheye applied frames. The addition of **height** and **width** will need to be set depending on the image size. The **focal_length** should also be set so that the naming convention for the output files and videos are set correctly.