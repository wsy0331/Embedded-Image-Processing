import cv2
import numpy as np

image = cv2.imread("./images.jpeg")
blurred_image = cv2.GaussianBlur(image, (7, 7), 0)
gray_image = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2GRAY)
ret, _ = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
canny_image = cv2.Canny(gray_image, ret * 0.9, ret)
hough_circles = cv2.HoughCircles(
    canny_image,
    cv2.HOUGH_GRADIENT,
    dp=1.3,
    minDist=30,
    param1=300,
    param2=30,
    minRadius=6,
    maxRadius=20,
)

if hough_circles is not None:
    # 1. 動態獲取影像高度與寬度
    height, width = gray_image.shape[:2]

    # 將座標轉為整數
    circles = np.uint16(np.around(hough_circles))

    for i in circles[0, :]:
        center_x, center_y = i[0], i[1]

        # 2. 動態檢查：確保座標在當前影像範圍內，避免 IndexError
        if 0 <= center_x < width and 0 <= center_y < height:
            # 3. 取得圓心像素的亮度值
            pixel_value = gray_image[center_y, center_x]

            # 4. 【核心過濾】只有當圓心「夠黑」才執行繪製
            if pixel_value < 50:
                # 畫出圓周
                cv2.circle(image, (center_x, center_y), i[2], (0, 255, 0), 2)
                # 畫出圓心點
                cv2.circle(image, (center_x, center_y), 2, (0, 0, 255), 3)

# display the  image
cv2.imshow("Original Image", image)
cv2.imshow("Blurred Image", blurred_image)
cv2.imshow("Gray Image", gray_image)
cv2.imshow("Canny Image", canny_image)
cv2.imshow("Hough Circles", image)

cv2.waitKey(0)
cv2.destroyAllWindows()
