import cv2
import dlib
import time
from config import face_landmark_path, capMode
from pprint import pprint
from MyImport.cropImage import cropImage, findArea
from threading import Thread
import multiprocessing
from concurrent.futures.process import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor

def faceCulculate(queue1, queue2, detector):
    while True:
        gray = queue1.get()
        if gray is None:
            pass
        else:
            faces = detector(gray)
            queue2.put(faces)
def getFrame(cap):
    return cap.read()


if __name__ == '__main__':

    posIn = [0, 0, 0, 0]
    positionAdd = [0, 0]

    # 加载人脸关键点检测模型
    predictor_path = face_landmark_path
    predictor = dlib.shape_predictor(predictor_path)
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') # haar人脸检测器，速度更快但准确性不如dlib
    detector = dlib.get_frontal_face_detector() #改为全局变量，避免重复加载


    queue1 = multiprocessing.Queue() # 创建队列
    queue2 = multiprocessing.Queue()


    pface = multiprocessing.Process(target=faceCulculate, args=(queue1, queue2, detector))
    pface.start()

    #创建线程池
    pool = ThreadPoolExecutor(2)

    # 初始化摄像头
    cap = cv2.VideoCapture(capMode)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

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
        f = pool.submit(getFrame, cap)
        ret, frame = f.result()
        # pprint(frame.shape)

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
        queue1.put(gray)
        # faces = detector(gray)
        faces = queue2.get(block=True, timeout=1)
        # print('faces', type(faces))
        faces_remake = dlib.rectangles()

        faceend = time.time()
        # print('facetime', faceend - facestart)
        
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
            posIn = findArea(facelist)
            posIn[0] += positionAdd[0]
            posIn[1] += positionAdd[1]
            posIn[2] += positionAdd[0]
            posIn[3] += positionAdd[1]
        else:
            posIn = [0, 0, frame.shape[0]-1, frame.shape[1]-1]
        


        # 计算帧率
        frame_counter += 1
        end_time = time.time()
        elapsed_time = end_time - start_time
        fps = frame_counter / elapsed_time

        # 在图像上显示帧率
        cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # # 显示图像
        cv2.imshow('Face Detection', frame)
        # cv2.waitKey(0)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()


