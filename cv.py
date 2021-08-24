import cv2
from cv2 import dnn
import numpy as np
print(cv2.__version__)
class_name = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
net = dnn.readNetFromTensorflow('D:\\code\\PYTHON\\gesture_recognition\\model\\frozen_model\\frozen_graph.pb')
cap = cv2.VideoCapture(0)
i = 0
while True:
    _,frame= cap.read() 
    src_image = frame
    cv2.rectangle(src_image, (300, 100),(600, 400), (0, 255, 0), 1, 4)
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    pic = frame[100:400,300:600]
    cv2.imshow("pic1", pic)
    # print(pic.shape)
    pic = cv2.resize(pic,(100,100))
    blob = cv2.dnn.blobFromImage(pic,     
                             scalefactor=1.0/225.,
                             size=(100, 100),
                             mean=(0, 0, 0),
                             swapRB=False,
                             crop=False)
    # blob = np.transpose(blob, (0,2,3,1))                         
    net.setInput(blob)
    out = net.forward()
    out = out.flatten()

    classId = np.argmax(out)
    # print("classId",classId)
    print("预测结果为：",class_name[classId])
    src_image =	cv2.putText(src_image,str(classId),(300,100), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,0,255),2,4)
    # cv.putText(img, text, org, fontFace, fontScale, fontcolor, thickness, lineType)
    cv2.imshow("pic",src_image)
    if cv2.waitKey(10) == ord('0'):
        break

