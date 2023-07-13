資源:
	HW2_40843238.py: 主程式
	myModule.py: 存放 MSE,PSNR,NC 計算函式
	.\Images\barbara_512x512.bmp :原始圖像
	.\Images\WM_40843238_160x200.jpg :要被嵌入的浮水印，我的大頭照

需求:
	三個 python 套件: cv2, numpy, pywt

	pip install PyWavelets
	pip install opencv-python
	pip install numpy

執行:
	py .\HW2_40843238.py 

執行後會在 Images 目錄底下產生:
	barbara_wmed.bmp  : 在LL2嵌入浮水印後的圖像
	extracted_wm_rotated.jpg : 從旋轉兩度後的圖像 barbara_wmed_rotated.bmp 中提取出的浮水印
	extracted_wm_cutted.jpg : 從剪裁後的圖像 barbara_wmed_cutted.bmp 中提取出的浮水印
	