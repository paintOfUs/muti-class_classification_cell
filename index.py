from roboflow import Roboflow
rf = Roboflow(api_key="LUnRtXbJA62BIFv8Xdah")
project = rf.workspace("minh-tun").project("yaml1")
dataset = project.version(1).download("yolov8")