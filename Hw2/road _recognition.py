import cv2
import numpy as np
import matplotlib.pyplot as plt

def pipeline_no_roi(image_path):
    # ==========================================
    # 0. 讀取圖片
    # ==========================================
    img = cv2.imread(image_path)
    if img is None:
        print(f"找不到圖片：{image_path}")
        return
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    height, width = img.shape[:2]

    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV) #轉換到HSV色彩空間
    lower_road = np.array([0, 0, 40])   
    upper_road = np.array([180, 100, 255]) 
    color_mask = cv2.inRange(hsv, lower_road, upper_road) #顏色遮罩，找出所有可能的馬路色塊

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #灰階處理
    blur = cv2.GaussianBlur(gray, (7, 7), 0) #高斯模糊
    edges = cv2.Canny(blur, 30, 100) #Canny邊緣檢測
    
    # 將邊緣線膨脹 (加粗)，變成一道無法跨越的厚牆
    kernel_wall = np.ones((5, 5), np.uint8)
    thick_edges = cv2.dilate(edges, kernel_wall, iterations=3)

    # ==========================================
    # 3. 物理切斷 (顏色 扣除 防波堤)
    # ==========================================
    # cv2.bitwise_not 會把防波堤變成黑色(0)，其他地方為白色(1)
    # 這樣交界處就會被強制切出空隙
    disconnected_mask = cv2.bitwise_and(color_mask, cv2.bitwise_not(thick_edges)) 

    # ==========================================
    # 4. 尋找輪廓與根部檢驗 (只留碰到畫面底部的)
    # ==========================================
    contours, _ = cv2.findContours(disconnected_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    final_road_mask = np.zeros_like(disconnected_mask)
    
    if contours:
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 5000:
                x, y, w, h = cv2.boundingRect(contour)
                
                # 關鍵防呆：這個形狀必須一直延伸到畫面的最下方 (大於 95%)
                if (y + h) >= height * 0.95:
                    cv2.drawContours(final_road_mask, [contour], -1, 255, thickness=cv2.FILLED)

    # 因為剛才扣除了加粗的白線防波堤，馬路邊緣會往內縮水
    # 所以最後要再把馬路膨脹回來，填滿整個車道
    final_road_mask = cv2.dilate(final_road_mask, kernel_wall, iterations=3)

    # ==========================================
    # 5. 上色疊加
    # ==========================================
    green_fill = np.zeros_like(img_rgb)
    green_fill[:] = (0, 255, 0)
    green_road_only = cv2.bitwise_and(green_fill, green_fill, mask=final_road_mask)
    final_result = cv2.addWeighted(img_rgb, 1.0, green_road_only, 0.4, 0)

    #圖表顯示
    plt.figure(figsize=(20, 10))
    
    plt.subplot(1, 4, 1)
    plt.title("1. Original Image")
    plt.imshow(img_rgb)
    plt.axis("off")

    plt.subplot(1, 4, 2)
    plt.title("2. Thick Edges (The Walls)")
    plt.imshow(thick_edges, cmap='gray')
    plt.axis("off")

    plt.subplot(1, 4, 3)
    plt.title("3. Disconnected Mask")
    plt.imshow(disconnected_mask, cmap='gray')
    plt.axis("off")

    plt.subplot(1, 4, 4)
    plt.title("4. Final No-ROI Result")
    plt.imshow(final_result)
    plt.axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    IMAGE_PATH = "test2.jpg"
    pipeline_no_roi(IMAGE_PATH)
    