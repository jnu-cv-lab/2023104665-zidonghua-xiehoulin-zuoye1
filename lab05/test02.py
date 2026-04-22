import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# ----------------------
# 1. 手动实现透视变换矩阵求解
# ----------------------
def my_getPerspectiveTransform(src, dst):
    A = np.zeros((8, 8), dtype=np.float64)
    B = np.zeros((8, 1), dtype=np.float64)
    
    for i in range(4):
        x, y = src[i][0], src[i][1]
        u, v = dst[i][0], dst[i][1]
        
        A[2*i] = [x, y, 1, 0, 0, 0, -x*u, -y*u]
        B[2*i] = u

        A[2*i+1] = [0, 0, 0, x, y, 1, -x*v, -y*v]
        B[2*i+1] = v
        
    h = np.linalg.solve(A, B)
    h = np.append(h, 1.0)
    H = h.reshape((3, 3))
    return H

# ----------------------
# 2. 自动获取当前脚本路径，避免文件找不到
# ----------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
img_path = os.path.join(script_dir, "task.jpg")

# 读取图片并做异常检查
img = cv2.imread(img_path)
if img is None:
    raise FileNotFoundError(f"❌ 找不到图片文件！\n路径：{img_path}\n请确认：task.jpg 和 test02.py 必须在同一个文件夹里")

# ----------------------
# 3. 鼠标交互标记四个角点
# ----------------------
img_copy = img.copy()
points = []

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        cv2.circle(img_copy, (x, y), 8, (0, 0, 255), -1)
        cv2.putText(img_copy, str(len(points)), (x+10, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        cv2.imshow("标记四个角：左上→右上→右下→左下（按 q 结束）", img_copy)

cv2.namedWindow("标记四个角：左上→右上→右下→左下（按 q 结束）", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("标记四个角：左上→右上→右下→左下（按 q 结束）", click_event)

cv2.imshow("标记四个角：左上→右上→右下→左下（按 q 结束）", img_copy)
cv2.waitKey(0)
cv2.destroyAllWindows()

if len(points) != 4:
    raise ValueError("❌ 你必须标记 4 个点才能进行矫正，请重新运行！")

pts_src = np.array(points, dtype=np.float32)

# ----------------------
# 4. 透视矫正
# ----------------------
width, height = 600, 860
pts_dst = np.array([
    [0, 0],
    [width-1, 0],
    [width-1, height-1],
    [0, height-1]
], dtype=np.float32)

# OpenCV 官方方法矫正
M_cv = cv2.getPerspectiveTransform(pts_src, pts_dst)
img_cv = cv2.warpPerspective(img, M_cv, (width, height))

# 自己实现的方法矫正
M_my = my_getPerspectiveTransform(pts_src, pts_dst)
img_my = cv2.warpPerspective(img, M_my, (width, height))

# ----------------------
# 5. 显示结果并保存
# ----------------------
plt.figure(figsize=(12, 5))
plt.subplot(131), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), plt.title("原图")
plt.subplot(132), plt.imshow(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)), plt.title("OpenCV矫正")
plt.subplot(133), plt.imshow(cv2.cvtColor(img_my, cv2.COLOR_BGR2RGB)), plt.title("手动实现矫正")
plt.show()

os.makedirs(os.path.join(script_dir, "lab05"), exist_ok=True)
cv2.imwrite(os.path.join(script_dir, "lab05", "final_cv.jpg"), img_cv)
cv2.imwrite(os.path.join(script_dir, "lab05", "final_my.jpg"), img_my)

print("✅ 矫正完成！结果已保存到 lab05 文件夹！")