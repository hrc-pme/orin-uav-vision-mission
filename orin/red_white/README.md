# Red White X OpenCV Detector

這個資料夾用 OpenCV 偵測無人機看到的「紅底白 X」80 cm x 80 cm 指認圖。

## 作法

1. 用 HSV 找紅色區塊，抓出接近正方形的候選外框。
2. 把候選外框做透視校正，轉成固定大小的正方形影像。
3. 在校正後的區域找低飽和高亮度的白色區塊。
4. 用 X 樣板重疊分數與 Hough 斜線檢查，確認白色形狀像 X。
5. 通過後畫出綠色框、中心點，並輸出 JSON。

這比只偵測紅色可靠，因為紅色車燈、布條、屋頂也可能是紅色；加入白色 X 形狀檢查後會少很多誤判。

## 啟動

```bash
cd /home/hrc/xin_uav/orin/red_white
./run_red_white_x_detector.sh
```

指定相機或影片：

```bash
SOURCE=0 ./run_red_white_x_detector.sh
SOURCE=/path/to/video.mp4 ./run_red_white_x_detector.sh
```

測試單張圖片並輸出結果：

```bash
python3 detect_red_white_x.py --source test.jpg --image --output out.jpg --print-json
```

## 無人機上建議

- 指認圖保持霧面紅底、霧面白 X，避免反光。
- 盡量讓圖案在畫面中至少有 40 x 40 px；太小時 X 形狀會不穩。
- 飛高時把相機快門調快，避免震動造成紅白邊界糊掉。
- 真正要算 WGS84 時，這個偵測器只提供影像中心點；仍要接飛控 GPS/RTK、姿態、相機內參與地面射線定位。

## 可調參數

- `MIN_AREA`：紅色候選最小面積，飛很高時可降低，例如 `MIN_AREA=250`。
- `WIDTH` / `HEIGHT`：相機解析度。
- `FOCAL_PX`：相機校正後焦距像素值；設定後會粗估距離。

例：

```bash
MIN_AREA=250 WIDTH=1920 HEIGHT=1080 ./run_red_white_x_detector.sh
```
