多媒體安全 HW1 
40843238 資工四乙 江知侑

-------0.使用環境------------
1.windows
2.vscode
3.python 3.11.0

-------1.環境安裝------------
1.安裝 vscode 

2.安裝 python 直譯器
    ctrl+shift+x 搜尋 python 並安裝，安裝後重開 vscode

3.安裝 opencv 用來操作圖像用
    ctrl+` 開啟終端機
    輸入: pip install opencv-python

-------2.操作方法------------
1. 將 "40843238 HW1-1.py", "40843238 HW1-2.py", 
    "elaine_512x512.bmp", "nfuwm_68x68,jpg" 放置同一資料夾底下

2. vscode 開啟該資料夾
    ctrl+k ctrl+o

3. 開啟終端機執行 python 程式
    ctrl+` 開啟終端機
    確定在正確的資料夾底下，如果不在，則用 cd 更換目錄 ls 查看當前目錄

    輸入 py '.\40843238 HW1-1.py'
        畫面顯示:
            1.原圖  
            2.用50% 個 0,1序列數目加入原圖的圖
            3.用嵌入個數為原始照片的 2 倍的0,1序列的圖
        並在下方終端機顯示:
            1.用50% 個 0,1序列數目加入原圖的圖 的 PSNR
            2.用嵌入個數為原始照片的 2 倍的0,1序列的圖 的　PSNR

        在任一圖片畫面按下任意鍵以關閉Python程式。
        
    輸入 py '.\40843238 HW1-2.py'
        畫面顯示:
            1.原圖
            2.替換第 7 位元，浮水印重複 3 次的 圖片
            3.替換第 7 位元，浮水印重複 1 次的 圖片
            4.替換第 3 位元，浮水印重複 3 次的 圖片
            5.替換第 3 位元，浮水印重複 1 次的 圖片
        並在下方終端機顯示:
            1.替換第 7 位元，浮水印重複 3 次的 PSNR
            2.替換第 7 位元，浮水印重複 1 次的 PSNR
            3.替換第 3 位元，浮水印重複 3 次的 PSNR
            4.替換第 3 位元，浮水印重複 1 次的 PSNR