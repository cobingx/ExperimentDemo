import cv2
import dlib
import time
import csv
import multiprocessing
from config import face_landmark_path

detectorTimeStart = time.time()
detector = dlib.get_frontal_face_detector() #改为全局变量，避免重复加载
predictor = dlib.shape_predictor(face_landmark_path)
detectorTimeEnd = time.time()   
print("detectorTime:", detectorTimeEnd - detectorTimeStart, '\n')

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


#运用68模型计算
def detect_faces_landmarks(imgIn, mode='normal'):

    # 读取图像
    image = dlib.load_rgb_image(imgIn)

    #将图像灰度化
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 人脸检测
    faceStart = time.time()
    
    faces = detector(image)

    faceEnd = time.time()
    totalFaceTime = faceEnd - faceStart
    print('detector人脸检测时间：', totalFaceTime)


    # 遍历检测到的人脸
    for face in faces:
        culculateStart = time.time()
        # 获取人脸关键点
        landmarks = predictor(image, face)
        culculateEnd = time.time()
        totalCulculateTime = culculateEnd - culculateStart
        print('计算时间：', totalCulculateTime)

        # 绘制人脸框
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # 绘制人脸关键点
        for i in range(68):
            x, y = landmarks.part(i).x, landmarks.part(i).y
            cv2.circle(image, (x, y), 2, (0, 0, 255), -1)

    return totalFaceTime, totalCulculateTime


def haar_face(ImgIn):
    img = cv2.imread(ImgIn)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    start = time.time()
    faces = face_cascade.detectMultiScale(gray, 1.4, 5)
    end = time.time()
    print('haar_face时间:', end - start)

    faces_change = dlib.rectangles()
    for (x, y, w, h) in faces:
        faces_change.append(dlib.rectangle(x, y, x+w, y+h))
        print('faces_change:', faces_change, type(faces_change))

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x,y), (x+w, y+h), (255, 0, 0), 2)
    
    return end - start

    
def myFaceDetect(inputData, mode='normal'):
    img = cv2.imread(inputData)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    start = time.time()
    faces = face_cascade.detectMultiScale(gray, 1.2, 10) # 人脸检测
    end = time.time()
    print('haar_face——time:', end - start)
    face_rebuilds = dlib.rectangles()
    for fece in faces:
        face_rebuild = dlib.rectangle(fece[0], fece[1], fece[0]+fece[2], fece[1]+fece[3])
        face_rebuilds.append(face_rebuild)
        cv2.rectangle(img, (fece[0], fece[1]), (fece[0]+fece[2], fece[1]+fece[3]), (255, 0, 0), 2)

        culculateStart = time.time()
        # 获取人脸关键点
        landmarks = predictor(img, face_rebuild)
        culculateEnd = time.time()
        totalCulculateTime = culculateEnd - culculateStart
        print('计算时间：', totalCulculateTime)
        # 绘制人脸关键点
        for i in range(68):
            x, y = landmarks.part(i).x, landmarks.part(i).y
            cv2.circle(img, (x, y), 2, (0, 0, 255), -1)
    






if __name__ == '__main__':
    # with open('face_landmark.csv', 'w', newline='') as csvfile:
    #     totolFaceTime = 0
    #     totolCulculateTime = 0
    #     for i in range(20):
    #         timeFace = haar_face('test.jpg')
    #         totolFaceTime += timeFace
    #         # totolCulculateTime += timeCulculate
    #         writer = csv.writer(csvfile)
    #         writer.writerow([timeFace])
    #         print('第', i + 1, '次计算完成')
    pass



