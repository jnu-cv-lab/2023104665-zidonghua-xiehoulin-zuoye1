import cv2
import numpy as np
import os

# 创建输出文件夹
os.makedirs("lab05", exist_ok=True)

# ========== 作业1：测试图变换 ==========
# 方法1：用原始字符串（推荐）
img_path = r"C:\cv-course\lab05\test.png"

# 或者方法2：用 / 分隔
# img_path = "C:/cv-course/lab05/test.png"

img = cv2.imread(img_path)

# 关键：判断图片是否读取成功
if img is None:
    raise FileNotFoundError(f"❌ 找不到图片文件：{img_path}\n请检查：\n1. 文件是否存在\n2. 文件名/后缀是否正确\n3. 路径是否正确")

h, w = img.shape[:2]

# ==================== 1. 相似变换 ====================
M_similar = cv2.getRotationMatrix2D((w//2, h//2), 30, 0.8)
img_similar = cv2.warpAffine(img, M_similar, (w, h))
cv2.imwrite("lab05/task1_similar.png", img_similar)

# ==================== 2. 仿射变换 ====================
pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
pts2 = np.float32([[10, 100], [200, 50], [100, 250]])
M_affine = cv2.getAffineTransform(pts1, pts2)
img_affine = cv2.warpAffine(img, M_affine, (w, h))
cv2.imwrite("lab05/task1_affine.png", img_affine)

# ==================== 3. 透视变换 ====================
pts_p1 = np.float32([[0, 0], [w-1, 0], [0, h-1], [w-1, h-1]])
pts_p2 = np.float32([[60, 50], [w-60, 70], [40, h-50], [w-40, h-60]])
M_persp = cv2.getPerspectiveTransform(pts_p1, pts_p2)
img_persp = cv2.warpPerspective(img, M_persp, (w, h))
cv2.imwrite("lab05/task1_perspective.png", img_persp)

print("✅ 任务1完成：3张变换图已保存到 lab05 文件夹！")