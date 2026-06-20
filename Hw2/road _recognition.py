import cv2
import numpy as np
import matplotlib.pyplot as plt

def get_roi(image, vertices):
    """定義感興趣區域 (Region of Interest)"""
    mask = np.zeros_like(image)
    # 如果影像是灰階的 (單通道)，忽略顏色通道
    if len(image.shape) > 2:
        channel_count = image.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

def draw_lines(image, lines, color=[255, 0, 0], thickness=5):
    """將檢測到的線段畫在空白畫布上"""
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_image, (x1, y1), (x2, y2), color, thickness)
    return line_image

def pipeline(image_path):
    # ==========================================
    # 0. 讀取圖片
    # ==========================================
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError("圖片載入失敗，請確認路徑。")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 取得圖片長寬
    height, width = img.shape[:2]

    # ==========================================
    # 1. 灰階處理 (Grayscale)
    # ==========================================
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ==========================================
    # 2. 高斯模糊 (Gaussian Blur) - 去除雜訊
    # ==========================================
    kernel_size = 5
    blur = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)

    # ==========================================
    # 3. 邊緣檢測 (Canny Edge Detection)
    # ==========================================
    low_threshold = 50
    high_threshold = 150
    edges = cv2.Canny(blur, low_threshold, high_threshold)

    # ==========================================
    # 4. 感興趣區域 (Region of Interest, ROI)
    # ==========================================
    # 建立一個梯形範圍 (假設行車記錄器視角，車道在畫面下半部)
    # 這裡的座標 (x, y) 原點在左上角
    bottom_left = (width * 0.1, height)
    bottom_right = (width * 0.9, height)
    top_left = (width * 0.45, height * 0.6)
    top_right = (width * 0.55, height * 0.6)
    
    roi_vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
    roi_edges = get_roi(edges, roi_vertices)

    # ==========================================
    # 5. 霍夫變換 (Hough Transform) - 從邊緣找直線
    # ==========================================
    rho = 1               # 距離解析度 (像素)
    theta = np.pi / 180   # 角度解析度 (弧度)
    threshold = 30        # 交點數量門檻值 (越小找到越多線)
    min_line_len = 40     # 最短線段長度
    max_line_gap = 20     # 允許線段之間的最大中斷距離

    lines = cv2.HoughLinesP(roi_edges, rho, theta, threshold, np.array([]), 
                            minLineLength=min_line_len, maxLineGap=max_line_gap)

    # 將線條畫在全黑的畫布上
    line_image = draw_lines(img_rgb, lines)

    # ==========================================
    # 6. 結果疊加 (Overlay)
    # ==========================================
    result = cv2.addWeighted(img_rgb, 0.8, line_image, 1, 0)

    # ==========================================
    # 視覺化顯示過程
    # ==========================================
    plt.figure(figsize=(20, 10))
    
    plt.subplot(2, 3, 1)
    plt.title("1. Original Image")
    plt.imshow(img_rgb)
    plt.axis("off")

    plt.subplot(2, 3, 2)
    plt.title("2. Grayscale & Blur")
    plt.imshow(blur, cmap='gray')
    plt.axis("off")

    plt.subplot(2, 3, 3)
    plt.title("3. Canny Edges")
    plt.imshow(edges, cmap='gray')
    plt.axis("off")

    plt.subplot(2, 3, 4)
    plt.title("4. ROI Masked Edges")
    plt.imshow(roi_edges, cmap='gray')
    
    # 在畫布上畫出 ROI 的紅線框，方便你檢查位置
    pts = roi_vertices.reshape((-1, 1, 2))
    roi_visual = cv2.polylines(img_rgb.copy(), [pts], isClosed=True, color=(255, 0, 0), thickness=3)
    plt.imshow(roi_visual, alpha=0.3) 
    plt.axis("off")

    plt.subplot(2, 3, 5)
    plt.title("5. Detected Lines")
    plt.imshow(line_image)
    plt.axis("off")

    plt.subplot(2, 3, 6)
    plt.title("6. Final Result")
    plt.imshow(result)
    plt.axis("off")

    plt.tight_layout()
    plt.show()

# 執行程式
if __name__ == "__main__":
    # 替換成你的馬路圖片路徑
    IMAGE_PATH = "test4.jpg" 
    pipeline(IMAGE_PATH)