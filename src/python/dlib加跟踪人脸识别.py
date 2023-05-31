import cv2
import dlib
import time
from config import face_landmark_path, capMode
from pprint import pprint
from MyImport.cropImage import *

csvWrite('way.csv', ['count','fps', 'rate', 'flag'], 'w')
rightFlag = 0
posIn = [0, 0, 0, 0]
positionAdd = [0, 0]
totalTime = 0
lastTime = time.time()
# 加载人脸关键点检测模型
predictor_path = face_landmark_path
predictor = dlib.shape_predictor(predictor_path)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') # haar人脸检测器，速度更快但准确性不如dlib
detector = dlib.get_frontal_face_detector() #改为全局变量，避免重复加载

# 初始化摄像头
cap = cv2.VideoCapture(capMode)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# 计算专注时间
reduceTime = 0

#计算正确率
righrRate = 0.0
rightLoopTime = 0.0
totalLoopTime = 0.0

# 初始化帧率计算
start_time = time.time()
frame_counter = 0

ret, frame = cap.read()
if not ret:
    print('error')
    exit()
else:
    posIn[2] = frame.shape[0]
    posIn[3] = frame.shape[1]

while True:

    if totalLoopTime == 100:
        rightLoopTime = 0.0
        totalLoopTime = 0.0
    totalLoopTime += 1
    timeEveryWhile = time.time()
    ret, frame = cap.read()
    pprint(frame.shape)

    if not ret:
        break

    
    frame_crop, positionAdd = cropImage(frame, posIn[0], posIn[1], posIn[2], posIn[3])
    # cv2.imshow('frame_crop', frame_crop)
    # cv2.waitKey(0)
        # 将图像转换为灰度图
    gray = cv2.cvtColor(frame_crop, cv2.COLOR_BGR2GRAY)


   

    facestart = time.time()
    # 使用haar进行人脸检测
    # faces = face_cascade.detectMultiScale(gray, 1.1, 20) # 人脸检测
    faces = detector(gray)
    # print('faces', type(faces))
    faces_remake = dlib.rectangles()

    faceend = time.time()
    print('facetime', faceend - facestart)
    
    facelist = []
    # 遍历检测到的人脸
    for face in faces:

        landmarks = predictor(gray, face)

        # 绘制人脸关键点
        for n in range(0, 68):
            x = landmarks.part(n).x + positionAdd[0]
            y = landmarks.part(n).y + positionAdd[1]
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
        
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        facelist.append([x, y, w, h])
    
    if len(facelist) != 0:
        rightFlag = 1
        rightLoopTime += 1
        posIn = findArea(facelist)
        posIn[0] += positionAdd[0]
        posIn[1] += positionAdd[1]
        posIn[2] += positionAdd[0]
        posIn[3] += positionAdd[1]
    else:
        rightFlag = 0
        reduceTime = time.time() - timeEveryWhile + reduceTime
        posIn = [0, 0, frame.shape[0]-1, frame.shape[1]-1]
    

    righrRate = rightLoopTime / totalLoopTime
    # 计算帧率
    frame_counter += 1
    end_time = time.time()
    elapsed_time = end_time - start_time
    fps = frame_counter / elapsed_time
    csvWrite('way.csv', [frame_counter, fps, righrRate * 100, rightFlag])
    totalTime = time.time() - start_time -reduceTime
    # 在图像上显示帧率
    cv2.putText(frame, f'FPS: {fps:.2f}  Time: {int(totalTime // 3600)} : {int((totalTime % 3600) // 60)} : {int(totalTime % 60)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


    # # 显示图像
    cv2.imshow('Face Detection', frame)
    # cv2.waitKey(0)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()


