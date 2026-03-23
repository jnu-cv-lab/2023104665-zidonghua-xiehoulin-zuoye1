import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# ===================== 核心配置（不用改） =====================
# 1. 自动获取当前代码所在的文件夹路径（lab01）
CODE_DIR = os.path.dirname(os.path.abspath(__file__))
# 2. 原始图片文件名（确保lab01里有这个文件）
IMG_FILENAME = "test.jpg"
# 3. 拼接原始图片的绝对路径（永远指向lab01里的test.jpg）
IMG_PATH = os.path.join(CODE_DIR, IMG_FILENAME)

# ===================== 解决中文显示问题 =====================
plt.rcParams["font.sans-serif"] = ["SimHei"]  # 显示中文
plt.rcParams["axes.unicode_minus"] = False    # 解决负号显示问题

# ===================== 1. 读取图片 =====================
# 读取彩色图片
img = cv2.imread(IMG_PATH)
# 检查图片是否读取成功
if img is None:
    print(f"❌ 读取图片失败！请检查文件是否存在：{IMG_PATH}")
    exit()

# ===================== 2. 输出图片基本信息 =====================
print("✅ 图片读取成功！")
print(f"📏 图片宽度（列）：{img.shape[1]} 像素")
print(f"📏 图片高度（行）：{img.shape[0]} 像素")
print(f"🎨 图片通道数：{img.shape[2]}（BGR格式）")

# ===================== 3. 显示原始图片（OpenCV + Matplotlib） =====================
# OpenCV读取的是BGR格式，转成RGB格式用于Matplotlib显示
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 用Matplotlib显示原图
plt.figure(figsize=(8, 6))
plt.imshow(img_rgb)
plt.title("original（RGB）")
plt.axis("off")  # 隐藏坐标轴
plt.show()

# 用OpenCV显示原图（按任意键关闭）
cv2.imshow("original（BGR）", img)
print("\n👉 请点击OpenCV图片窗口，按任意键继续...")
cv2.waitKey(0)
cv2.destroyWindow("original（BGR）")

# ===================== 4. 转换为灰度图 =====================
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print(f"\n✅ 灰度图转换完成！灰度图形状：{gray_img.shape}")

# 显示灰度图（按任意键关闭）
cv2.imshow("gray", gray_img)
print("👉 请点击灰度图窗口，按任意键继续...")
cv2.waitKey(0)
cv2.destroyWindow("gray")

# ===================== 5. 保存处理后的图片（自动存到lab01） =====================
# 拼接灰度图保存路径（lab01/灰度图_test.jpg）
gray_save_path = os.path.join(CODE_DIR, f"gray_{IMG_FILENAME}")
cv2.imwrite(gray_save_path, gray_img)
print(f"\n✅ 灰度图已保存：{gray_save_path}")

# ===================== 6. 裁剪图片（左上角100x100区域） =====================
# 裁剪区域：行[0:100]，列[0:100]
crop_img = img[0:100, 0:100]
print(f"✅ 图片裁剪完成！裁剪区域：左上角100x100像素")

# 保存裁剪图（lab01/裁剪图_test.jpg）
crop_save_path = os.path.join(CODE_DIR, f"cut_{IMG_FILENAME}")
cv2.imwrite(crop_save_path, crop_img)
print(f"✅ 裁剪图已保存：{crop_save_path}")

# ===================== 7. 输出像素值示例 =====================
print("\n📊 像素值示例：")
print(f"原始图左上角(0,0)像素（BGR）：{img[0, 0]}")
print(f"灰度图左上角(0,0)像素：{gray_img[0, 0]}")

# ===================== 收尾 =====================
cv2.destroyAllWindows()  # 关闭所有OpenCV窗口
print("\n🎉 所有操作完成！处理后的图片已保存在lab01文件夹里～")