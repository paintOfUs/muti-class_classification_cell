from roboflow import Roboflow
from glob import glob

url = "D:\\important_for_study_master\\DoAn\\New folder\\data\\test\\CHECK\\"
url_save = "D:\\important_for_study_master\\DoAn\\New folder\\data\\test\\CHECK\\checked\\allgen\\"
list_check = ['2C0','C0','C0C2']
check = []
for folder in list_check:
    
    # Biến để tính xác xuất đoán đúng
    total = 0
    rs = 0
    for number,filename in enumerate(glob(url+folder+"\\*")):
        try:
            # rf = Roboflow(api_key="VhEfnIEn0WbyXX0aCYES")
            # project = rf.workspace().project("aginate_same_wh_140_160")
            # model = project.version(1).model
            rf = Roboflow(api_key="a50d3h0y91WAU9xjOMSh")
            project = rf.workspace().project("materialv3")
            model = project.version(7).model
            

            #model dự đoán
            print(number)
            predict = model.predict(filename, confidence=50, overlap=50).json()
            if(len(predict['predictions'])!=0):
                if(predict["predictions"][-1]["class"] == folder):
                    rs +=1
            total +=1               
        except OSError as e:
            print('error: '+e)
    check.append(rs/total)

for i,item in enumerate(check):
    print('độ chính xác '+str(list_check[i])+' '+str(item))
