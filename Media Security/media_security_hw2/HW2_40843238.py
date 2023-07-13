import cv2 # 圖像操作的套件
import numpy as np # 數值操作的套建
import pywt # 用來做 DWT 的套件
import myModule # 計算 PSNR, NC 值的套件，寫在 myModule Python 檔

# ------------------------- Class, function -------------------------------------------
class Operations:
    
    #建構子 傳入原始圖像、浮水印。設定嵌入浮水印時的強度 alpha，因為嵌入跟取出時都會用到所以在這邊設定
    def __init__(self,host,wm): 
        self.host = host
        self.wm = wm
        self.alpha = 0.02
 
        
    # 將圖像從空間域做2level DWT後轉成頻率域，在LL2嵌入浮水印後，做idwt得到對圖像頻率域做浮水印嵌入的圖像
    def embed_watermark(self): 
        # 使用pywt套件做 2 層 DWT 轉換
        # cA2 代表 LL2 頻帶的係數矩陣
        #
        # cA2 LL2  cV2 HL2 
        # cH2 LH2  cD2 HH2 
        #
        #          cV1 HL1 
        # cH1 LH1 ,cD1 HH1 
        self.cA2, (self.cH2, self.cV2, self.cD2), (self.cH1, self.cV1, self.cD1) = pywt.wavedec2(self.host, 'haar', level=2)
    
        # 縮放浮水印，因為要將浮水印塞入 LL2 頻帶中，所以要先將浮水印重新調整尺寸以跟LL2一樣
        self.wm_resized = cv2.resize(self.wm, (self.cA2.shape[1], self.cA2.shape[0]), interpolation=cv2.INTER_LINEAR)

        # 嵌入浮水印，直接將要嵌入的浮水印乘上一個代表強度的alpha後加入LL2的係數 cA2
        self.cA2_wm = self.cA2 + self.alpha * self.wm_resized

        # 使用pywt套件做 idwt 得到idwt後的係數矩陣
        self.watermarked = pywt.waverec2((self.cA2_wm, (self.cH2, self.cV2, self.cD2), (self.cH1, self.cV1, self.cD1)), 'haar')
        self.watermarked_pic = self.watermarked.astype(np.uint8)

    # 從已經嵌入浮水印的圖像中取出浮水印
    def extract_watermark(self):
        # 先將嵌入浮水印後的圖片進行2階DWT轉換
        coeffs_watermarked = pywt.wavedec2(self.watermarked_pic, 'haar', level=2)
        
        # 提取LL2的部分
        cA2_watermarked = coeffs_watermarked[0]
        
        # inverse 嵌入浮水印的部分
        # 比較 self.cA2_wm = self.cA2 + self.alpha * self.wm_resized
        wm_extracted = ( self.cA2_wm-cA2_watermarked) / self.alpha
        
        # 取得隱藏的浮水印，resize回原本浮水印的大小，方便之後比較NC值
        wm_extracted = np.round(wm_extracted).astype(np.uint8)
        self.wm_extracted_pic = cv2.resize(wm_extracted, (wm.shape[1], wm.shape[0]), interpolation=cv2.INTER_LINEAR)
        
    # 取得嵌入浮水印後的圖片
    def get_watermarked_pic(self):
        return self.watermarked_pic
    
    # 儲存嵌入浮水印後的圖片
    def save_watermarked_pic(self):
        url = "./Images/barbara_wmed.bmp"
        cv2.imwrite(url,self.watermarked_pic)
        
    def save_extracted_watermark_pic(self,name):
        url = "./Images/"+name
        
        url = url+".jpg"
        cv2.imwrite(url,self.wm_extracted_pic)
        
    # 取得從嵌入浮水印後的圖片中的隱藏浮水印
    def get_extracted_watermark(self):
        return self.wm_extracted_pic
    
    # 設定 extract_watermark 時的目標圖片(rotate, cut)
    def set_watermarked_pic(self,watermarked_pic):
        self.watermarked_pic = watermarked_pic

# ------------------------- main -------------------------------------------


host = cv2.imread("./Images/barbara_512x512.bmp", cv2.IMREAD_GRAYSCALE)
wm = cv2.imread("./Images/WM_40843238_160x200.jpg", cv2.IMREAD_GRAYSCALE)
cv2.imshow('wm',wm)

# a 小題，嵌入浮水印後顯示
x = Operations(host,wm)
x.embed_watermark()
x.save_watermarked_pic()
wmed_host = x.get_watermarked_pic()
cv2.imshow('wmed_host',wmed_host)

# b 小題，比較PSNR
myModule.PSNR_calc(host,wmed_host)


# 從 沒有做任何破壞的圖中提出浮水印做比較
# x.extract_watermark()
# extracted_wm = x.get_extracted_watermark()
# cv2.imshow('extracted_wm',extracted_wm)
# myModule.NC_calc(wm,extracted_wm)

# c.1 從 rotated 2度的圖中提出浮水印做比較，以及計算NC
barbara_rotated = cv2.imread("./Images/barbara_wmed_rotated.bmp", cv2.IMREAD_GRAYSCALE)
cv2.imshow('rotated_host',barbara_rotated)
x.set_watermarked_pic(barbara_rotated)
x.extract_watermark()
extracted_wm = x.get_extracted_watermark()
x.save_extracted_watermark_pic('extracted_wm_rotated')
cv2.imshow('extracted_wm_rotated',extracted_wm)
myModule.NC_calc(wm,extracted_wm)

# c.2 從 cuted 的圖中提出浮水印做比較，以及計算NC
barbara_cutted = cv2.imread("./Images/barbara_wmed_cutted.bmp", cv2.IMREAD_GRAYSCALE)
cv2.imshow('cutted_host',barbara_cutted)
x.set_watermarked_pic(barbara_cutted)
x.extract_watermark()
extracted_wm = x.get_extracted_watermark()
x.save_extracted_watermark_pic('extracted_wm_cutted')
cv2.imshow('extracted_wm_cutted',extracted_wm)
myModule.NC_calc(wm,extracted_wm)

# --------------------------------------------------------

cv2.waitKey(0)
cv2.destroyAllWindows()