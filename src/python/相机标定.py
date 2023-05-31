import numpy as np
import cv2


image_filenames = ''
# 定义棋盘格的行数和列数
board_size = (9, 6)

# 设置棋盘格实际坐标
square_size = 2.5  # 棋盘格方块边长（单位：厘米）
object_points = np.zeros((board_size[0] * board_size[1], 3), np.float32)
object_points[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2) * square_size

# 准备用于标定的图片和对应的实际坐标
image_points = []  # 保存所有图片中的棋盘格角点的像素坐标
object_points_list = []  # 保存所有图片中棋盘格的实际坐标
images = []  # 保存所有图片

# 遍历图片文件夹中的所有图片
for filename in image_filenames:
    # 读取图片
    image = cv2.imread(filename)
    images.append(image)

    # 找到棋盘格角点
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, board_size, None)

    # 如果找到了棋盘格角点，则将其保存下来
    if ret:
        object_points_list.append(object_points)
        image_points.append(corners)

# 进行相机标定
ret, camera_matrix, distortion_coefficients, rvecs, tvecs = cv2.calibrateCamera(
    object_points_list,
    image_points,
    gray.shape[::-1],
    None,
    None
)

# 使用cv2.undistort函数将图像去畸变
for image in images:
    undistorted = cv2.undistort(image, camera_matrix, distortion_coefficients)
    cv2.imshow('Undistorted', undistorted)
    cv2.waitKey()

# 释放窗口
cv2.destroyAllWindows()
