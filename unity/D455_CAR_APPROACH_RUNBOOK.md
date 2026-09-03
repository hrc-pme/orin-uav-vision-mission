# D455 任意汽車兩階段接近流程

這是獨立的新任務；原本的 `run_outdoor.sh` 仍維持 G3P、白車與原控制流程。

## 實際任務狀態

1. `SEARCHING`：飛機在離 HOME 70--130 m 時搜尋穩定的 `car`。
2. `DESCENDING_TO_VERIFY`：連續辨識通過後，自動切到 GUIDED，飛到粗略車位並降到離 HOME 50 m。
3. `VERIFYING`：在 50 m 重新連續辨識至少 3 秒、5 個新座標，並要求位置誤差不超過 12 m。
4. `READY_FOR_UNITY_CONFIRM`：Unity 原本的 CONFIRM 按鈕才會變成可用；太早按會被後端拒絕。
5. `FINAL_APPROACH`：按 CONFIRM 後只採用後端精修座標，飛到車輛上方離 HOME 20 m。
6. `COMPLETE`：水平距離 4 m 內且高度誤差 3 m 內，連續兩次成立。

高度都是 ArduPilot 的「相對 HOME 高度」，不是地形即時 AGL。場地若有明顯高低差，
必須先換算 50 m 與 20 m 是否仍有足夠離地/離障礙高度。

## 明天使用的命令

起飛前先做唯讀本機檢查：

```bash
cd /home/hrc/xin_uav/unity
./preflight_d455_car_approach.py
```

室內掀槳/舉機測試請只用安全模式。它會開啟 D455、TensorRT、MAVLink telemetry
與 Unity 壓縮標註畫面，但強制關閉自動下降及所有飛行命令：

```bash
cd /home/hrc/xin_uav/unity
./run_d455_car_approach_indoor.sh
```

室內要確認 Cesium 城市圖層與 WGS84 錨點，可改用下列預覽。它只送出明確標示的
假座標 `24.7998907, 121.0296557`，所有飛行命令仍關閉；不能拿來實飛：

```bash
./run_d455_map_preview_indoor.sh
```

確認 D455 鏡頭光軸垂直朝下、槳與腳架不遮鏡、遙控器可立即接管。新程式一旦在
70--130 m 看到穩定汽車，就可能自動離開 AUTO 並切到 GUIDED；不要在不希望中止
原航線時啟動。

主程式：

```bash
cd /home/hrc/xin_uav/unity
./run_d455_car_approach.sh
```

另一個終端機錄影與記錄：

```bash
cd /home/hrc/xin_uav/unity
./record_d455_car_approach.sh
```

錄影啟動後才按需發布 1280×720 JPEG 70 原始 RGB，並同時記錄 512×288 標註影像、
D455 CameraInfo、每一筆 MAVLink JSON、GPS、飛控 IMU/速度/電池、telemetry、
辨識候選、Unity 命令/回覆與 approach state。停止錄影請按 `Ctrl+C`，等待
`Recording stopped` 後才關機。

Unity 筆電是 `10.0.0.7`，Orin 是 `10.0.0.8`，Unity `RosConnector` 必須是：

```text
ws://10.0.0.8:9090
```

Windows 專案路徑：

```text
C:\Users\a5156\Desktop\SmallObjectDetect\ImageProcess_unity\Assets\Script\UAVDashboard
```

現有 Unity CONFIRM/CANCEL JSON 格式不必改。後端會讓 `ready_for_confirm` 在低空
驗證完成前保持 `false`，並在 CONFIRM 時強制改用已驗證汽車的 Track ID 與精修座標，
避免 Unity 畫面仍握有 100 m 粗座標。

低空驗證使用 5 筆有效定位，位置誤差門檻仍是 12 m。YOLO/ByteTrack 短暫漏框或
更換 Track ID 時，最多保留確認進度 6 秒；新 Track 必須在上一筆有效車位 15 m
內，且仍在高空粗座標 35 m 內，才可接續累積。這避免單張漏偵測把 `4/5` 清成零，
同時不會只靠放寬位置誤差接受不可靠座標。

達到 5 筆後 Unity 會顯示 `CAR VERIFIED - PRESS CONFIRM`，確認狀態可在短暫漏框時
保留 5 秒供操作者按鍵；後端最多只接受 6 秒內的已驗證座標，逾時會回到低空驗證，
不會拿長時間以前的車位執行 20 m 最終接近。

高空自動觸發只接受完整位於綠色 `NAV TARGET ZONE` 內、持續至少 3 秒且最後偵測
不超過 1 秒的車。貼著畫面邊緣或被裁切的車仍會留在 rosbag 原始偵測資料中，但
絕不會觸發自動下降。Unity 主畫面最多只畫一個已連續確認的車框，避免短暫低信心
框和重複 Track ID 疊在一起。

Unity 大畫面只訂閱 `/d455i/color/image_annotated/compressed`：512×288、JPEG quality
48、5 Hz、QoS depth 1。`/d455i/color/image_raw/compressed` 平時不發布資料，只有
錄影器訂閱時才按需發布。右側確認窗不再複製即時畫面；候選車成立時傳送最多
256×160、JPEG quality 55 的車輛截圖，Unity 會快取沿用。為讓晚連線/重連的 Unity
也能取得圖片，後端每 5 秒才補傳一次，不會隨 5 Hz 座標訊息重複傳輸/解碼。

Orin 上的 Unity 程式修改後，在 Windows PowerShell 同步右側目標截圖版本：

```powershell
scp hrc@10.0.0.8:/home/hrc/unity_windows_review/UAVDashboard/TargetSelectionUIController.cs "C:\Users\a5156\Desktop\SmallObjectDetect\ImageProcess_unity\Assets\Script\UAVDashboard\TargetSelectionUIController.cs"
```

模型短暫漏偵測時，黃色 `TRACK` 框會以影像模板在目前畫面附近繼續追蹤車輛；
連續兩張影像找不到相同外觀就移除，不會把已離開畫面的框留在原地。這只是 Unity
畫面功能；自動導航仍要求 1.5--3 秒內的新鮮真實偵測，不會使用 `TRACK` 框當座標。

## 現場監看

```bash
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=42
ros2 topic echo /unity/approach_state
```

關鍵狀態：

- `SEARCHING`：等待 100 m 穩定汽車。
- `DESCENDING_TO_VERIFY`：正在往 50 m 驗證點。
- `VERIFYING`：已到點、正在累積低空座標。
- `READY_FOR_UNITY_CONFIRM`：現在才按 Unity CONFIRM。
- `FAILED`：不要繼續按 CONFIRM，先看 message 並由遙控器接管。

取消會在目前 GPS 點送出 hold：

```bash
ROS_DOMAIN_ID=42 ros2 topic pub --once /unity/target_command std_msgs/msg/String \
  "{data: '{\"action\":\"CANCEL\",\"command_id\":\"pilot-cancel\"}'}"
```

完成或取消後重新搜尋：

```bash
ROS_DOMAIN_ID=42 ros2 topic pub --once /unity/target_command std_msgs/msg/String \
  "{data: '{\"action\":\"RESET_SEARCH\",\"command_id\":\"new-search\"}'}"
```

## 不能省略的實飛前確認

- D455 安裝角必須與 `camera_mount_rpy_deg: [0,-90,0]` 一致。100 m 高度每 1 度
  安裝誤差約造成 1.75 m 地面偏移。
- `GPS satellites >= 8`、`EPH <= 5 m`、飛控已解鎖，模式必須是 GUIDED 或設定中
  允許自動切換的模式。
- 第一次不要直接飛 100 m：先在空曠區以約 30--50 m、地面無人且只有一台靜止車，
  驗證影像框、目標座標方向、CANCEL 與遙控器接管。此時因搜尋高度門檻為 70 m，
  不會自動觸發；確認投影方向後才進行完整流程。
- 汽車若在移動，影像投影與飛行時間會讓座標過期；目前流程以靜止車為目標。
