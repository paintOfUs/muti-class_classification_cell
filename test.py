from roboflow import Roboflow
from glob import glob

#Đường dẫn đến thư mục kiểm tra 
url = "D:\\important_for_study_master\\DoAn\\New folder\\data\\test\\CHECK\\"
url_save = "D:\\important_for_study_master\\DoAn\\New folder\\data\\test\\CHECK\\checked\\allgen\\"
list_check = ['2C0','C0','C0C2']

for folder in list_check:
    
    # Biến để tính xác xuất đoán đúng
    total = 0
    rs = 0

    sum_confidence_img = 0
    
    #list  chứa khối cầu với kích thước to nhỏ
    smallList =[]
    bigList = []
    xmallList = []
    #unpredict
    unpredict = []
    
    for number,filename in enumerate(glob(url+folder+"\\*")):
        try:
            rf = Roboflow(api_key="a50d3h0y91WAU9xjOMSh")
            project = rf.workspace().project("materialv3")
            model = project.version(5).model


            #model dự đoán
            predict = model.predict(filename, confidence=50, overlap=50).json() 
            #Kiem tra ket qua ma model tra ve
            if(len(predict["predictions"])==1):
                
                #Kiểm tra nồng độ nào thì sửa chuỗi sau ==
                if(predict["predictions"][0]["class"] == folder):
                    rs+=1
                    try:
                        sum_confidence_img += float(predict["predictions"][0]["confidence"])
                        widthSphere = int(predict["predictions"][0]["width"])
                        if widthSphere>140 and widthSphere<150:
                            smallList.append(str(number+1))
                        elif widthSphere <140:
                            xmallList.append(str(number+1))
                        else:
                            bigList.append(str(number+1))
                    except KeyError as c:
                        print(c)
                        pass
                else:
                    print("another1 error in picture: "+str(number+1))
                        
            elif(len(predict["predictions"]) > 1):
                
                #Kiểm tra nồng độ nào thì sửa chuỗi sau ==
                if(predict["predictions"][-1]["class"] == folder):
                    rs+=1
                    try:
                        sum_confidence_img += float(predict["predictions"][-1]["confidence"])
                        widthSphere = int(predict["predictions"][-1]["width"])
                        if widthSphere>140 and widthSphere<150:
                            smallList.append(str(number+1))
                        elif widthSphere <140:
                            xmallList.append(str(number+1))
                        else:
                            bigList.append(str(number+1))
                    except KeyError as c:
                        print(c)
                        pass
                else:
                    print("another2 error in picture: "+str(number+1))
                
            # lưu ảnh sau khi dự đoán
            #model.predict(filename, confidence=50, overlap=50).save(url_save+folder+'\\'+str(number+1)+".jpg")
            total+=1
            print("{rs}/{total}: ",rs,total)
            print("============================")
        except OSError as e:
            print("Something happened:", e)

    print("==============================================================")
    print("Tong `%` chinh xac là " + str(rs/total))
    print("Trung bình độ chính xác là: " + str(sum_confidence_img/total))
    print("===============================================================")
    
    #ghi data ket qua vao file
    with open('logdataAllGen.txt',"a") as f:
        f.write("Anh {} - Tong %\ chinh xac - Trung binh model\n".format(folder))
        f.write("xxxxxx    {}    {}\n".format(rs/total, sum_confidence_img/total))
        f.write("do to cua khoi cau 140-150 gom bao nhieu anh {}\n".format(len(smallList)))
        f.write(" ,".join(smallList))
        f.write('\n')
        f.write("do to cua khoi cau nho hon 140 gom bao nhieu anh {}\n".format(len(xmallList)))
        f.write(' ,'.join(xmallList))
        f.write('\n')
        f.write("do to cua khoi cau lon hon 150 gom bao nhieu anh {}\n".format(len(bigList)))
        f.write(" ,".join(bigList))
        f.write('\n')
        f.write("Không predict duoc gom bao nhieu anh {}\n".format(len(unpredict)))
        f.write(" ,".join(unpredict))
        f.write('\n')
        f.write('\n')

with open('logdata.txt',"a") as f:
    f.write("==================================================\n")