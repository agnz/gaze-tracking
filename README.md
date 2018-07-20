# gaze-tracking
Gaze tracking project

Obj-det:

(1)Extract the frames with video\_to\_frames.py

cmd: python3 video\_to\_frames.py VIDEO\_FILE\_PATH OUTPUT\_IMAGE\_DIRECTORY\_IN\_data

(2)use cvat\_xml\_to\_tf.py to convert xml and images extracted from the video to tf record file 

cmd: python3 cvat\_xml\_to\_tf.py XML\_FILE\_PATH IMAGE\_DIRECTORY\_IN\_data



