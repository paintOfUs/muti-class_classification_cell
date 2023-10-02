from roboflow import Roboflow
rf = Roboflow(api_key="a50d3h0y91WAU9xjOMSh")
project = rf.workspace().project("meterial_50image")
model = project.version(1).model
url = "D:\\important_for_study_master\\DoAn\\New folder\\data\\test\\CHECK\\C0C2\\16.JPG"
predict = model.predict(url, confidence=50, overlap=50).json() 
widthSphere = int(predict["predictions"][-1]["width"])
print(widthSphere)