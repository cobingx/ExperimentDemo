# 输入一帧图像，从图像中截取出目标区域
# factor为放大因子，放大后的区域宽高为原区域宽高的factor倍
import numpy
import csv
def cropImage(frameIn, x1, y1, x2, y2, factor=1.3):

    delta = factor - 1
    widthDelta = (x2 - x1) * delta /2
    highDelta = (y2 - y1) * delta /2
    x1 = int(x1 - widthDelta)
    x2 = int(x2 + widthDelta)
    y1 = int(y1 - highDelta)
    y2 = int(y2 + highDelta)
    if x1 < 0:
        x1 = 0
    if y1 < 0:
        y1 = 0
    if x2 > frameIn.shape[1]:
        x2 = frameIn.shape[1]
    if y2 > frameIn.shape[0]:
        y2 = frameIn.shape[0]
        
    return frameIn[y1:y2, x1:x2], [x1, y1]

def findArea(faces:numpy.ndarray) -> list:
    xmin = faces[0][0]
    ymin = faces[0][1]
    xmax = faces[0][0]
    ymax = faces[0][1]
    for (x, y, w, h) in faces:
        if x < xmin:
            xmin = x
        if y < ymin:
            ymin = y
        if x+w > xmax:
            xmax = x+w
        if y+h > ymax:
            ymax = y+h
    return [xmin, ymin, xmax, ymax]


#添加自适应功能
def autoAdaptive(factor, RightRate) -> float:
    #todo:根据输入帧规格进行映射
    if RightRate > 0.7 and factor > 1.3:
        factor -= 0.1
    elif RightRate < 0.5 and factor < 2.0:
        factor += 0.1
    return factor

def csvWrite(csvPath, data, mode='a'):
    with open(csvPath, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(data)
    csvfile.close()
