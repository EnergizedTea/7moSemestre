from roboflow import Roboflow
rf = Roboflow(api_key="nBNmcxt2DLOe9i3CS5qB")
project = rf.workspace("proyects").project("faces_v2")
version = project.version(2)
dataset = version.download("yolov8")