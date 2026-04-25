import cv2
import numpy as np

image = cv2.imread("./Hw1/image4.jpeg")
h, w = image.shape[:2]
image = cv2.resize(image, (int(w * 0.1), int(h * 0.1)))
blurred_image = cv2.GaussianBlur(image, (7, 7), 0)
gray_image = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2GRAY)
ret, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

contour_result = cv2.drawContours(cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR), contours, -1, (0, 255, 0), 1)




# ===== 顯示結果 =====
cv2.imshow("Original Image", image)
cv2.imshow("Blurred Image", blurred_image)
cv2.imshow("Gray Image", gray_image)
cv2.imshow("Binary Image", binary_image)
cv2.imshow("Contour Detection", contour_result)



cv2.waitKey(0)
cv2.destroyAllWindows()
