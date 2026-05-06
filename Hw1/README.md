# [Feature] Implementation of Circle Detection Pipeline with Otsu-Canny Coupling and Pixel-based Filtering

## Description (問題描述)
本提案旨在實現一套強健的圓形檢測流程。目前程式碼整合了 **Otsu 自動閥值** 與 **Canny 邊緣檢測**，並透過 **Hough Circles (霍夫圓變換)** 辨識影像中的圓形實體。為了提高檢測準確度並減少誤判，系統導入了基於「圓心像素亮度」的二次過濾機制。

## Core Algorithm Workflow (核心演算法流程)

1.  **影像預處理 (Preprocessing)**：使用 $7 \times 7$ 高斯模糊（Gaussian Blur）去除雜訊，平滑影像細節，防止邊緣檢測出現過多碎點。
2.  **動態閥值結合 (Otsu-Canny Coupling)**：
    * 透過 `cv2.THRESH_OTSU` 自動計算影像的最佳全局閥值 `ret`。
    * **關鍵創新**：將該閥值作為 Canny 邊緣檢測的動態門檻（高門檻設為 `ret`，低門檻為 `0.9 * ret`），使系統能適應不同對比度的影像。
3.  **霍夫圓變換 (Hough Circle Transform)**：
    * 使用 `HOUGH_GRADIENT` 模式偵測圓形。
    * 設定 `dp=1.3` (累加器解析度) 與 `minDist=30` (圓心最小距離) 來過濾重複的偵測結果。
4.  **像素級過濾 (Pixel-level Verification)**：
    * 在繪製結果前，程式會讀取圓心座標 `(center_x, center_y)` 的灰階值。
    * **過濾標準**：僅當圓心亮度 $< 50$（即中心為深色區域）時才視為有效目標。



---

## Implementation Steps (教學實作過程)

### Step 1: 動態計算邊緣門檻
避免使用硬編碼（Hard-coded）的數值，利用影像自身的統計特性進行運算。
```python
# 取得 Otsu 閥值
ret, _ = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# 套用至 Canny
canny_image = cv2.Canny(gray_image, ret * 0.9, ret)
