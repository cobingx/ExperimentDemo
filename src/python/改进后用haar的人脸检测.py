import cv2
import dlib
import time
from config import face_landmark_path, capMode

# 加载人脸关键点检测模型
predictor_path = face_landmark_path
predictor = dlib.shape_predictor(predictor_path)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') # haar人脸检测器，速度更快但准确性不如dlib
detector = dlib.get_frontal_face_detector() #改为全局变量，避免重复加载

# 初始化摄像头
cap = cv2.VideoCapture(capMode)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# 初始化帧率计算
start_time = time.time()
frame_counter = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # 将图像转换为灰度图
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    facestart = time.time()
    # # 使用dlib进行人脸检测
    # faces = dlib.get_frontal_face_detector()(gray)

    # 使用haar进行人脸检测
    faces = face_cascade.detectMultiScale(gray, 1.2, 20) # 人脸检测
    faces_remake = dlib.rectangles()

    faceend = time.time()
    print('facetime', faceend - facestart)

    # 遍历检测到的人脸
    for face in faces:

        # 获取人脸关键点
        face_rebuild = dlib.rectangle(face[0], face[1], face[0]+face[2], face[1]+face[3])
        faces_remake.append(face_rebuild)

        landmarks = predictor(gray, face_rebuild)

        # 绘制人脸关键点
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

    # 计算帧率
    frame_counter += 1
    end_time = time.time()
    elapsed_time = end_time - start_time
    fps = frame_counter / elapsed_time

    # 在图像上显示帧率
    cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # 显示图像
    cv2.imshow('Face Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()


#物理备份