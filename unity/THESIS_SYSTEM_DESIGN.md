# 碩士論文系統設計建議

## 建議題目

**基於邊緣運算與人機協同確認之無人機白色車輛偵測、地理定位與自主航行系統**

英文可用：**An Edge-AI and Human-in-the-Loop UAV System for White-Vehicle Detection, Geolocation, and Autonomous Navigation**

## 兩篇核心參考文獻的定位

1. [aerial-autonomy-stack (2026)](https://arxiv.org/abs/2602.07264)：作為系統工程架構依據。重點是 ROS2 模組化、GPU perception 到 autopilot action 的端到端整合、Jetson-in-the-loop，以及模擬與實機使用相同介面。
2. [Robust Autonomous Drone Testing Pipeline (2025)](https://arxiv.org/abs/2506.11400)：作為驗證方法依據。採用 SIL、HIL、受控實機與戶外場域四階段測試，並以 rosbag 與飛控 log 量測延遲、追蹤、定位、控制與失效恢復。

## 本研究的系統流程

```text
D455 compressed RGB
  → TensorRT car detector
  → BoT-SORT + camera-motion compensation
  → HSV white-vehicle verification
  → consecutive-frame confirmation
  → monocular ground-plane geolocation
  → robust multi-frame position filter
  → Unity candidate selection
  → candidate/track/GPS/AP6 safety supervisor
  → MAV_CMD_DO_REPOSITION
  → live distance and mission status feedback
```

## ROS2 邏輯模組

- Perception：D455、TensorRT、BoT-SORT、白色比例與連續幀確認。
- Localization：相機內外參、飛控姿態、相對高度、WGS84/TWD97 與誤差傳播。
- Mission Supervisor：候選快取、過期檢查、人機確認與安全 interlock。
- Autopilot Adapter：pymavlink telemetry、REPOSITION、ACK 與取消/hold。
- Ground Interface：rosbridge、Unity candidates、影像、telemetry 與 command status。
- Experiment Logger：rosbag 與飛控 log，確保所有實驗可重播、可比較。

## 可主張的研究貢獻

1. 在 Orin NX 上完成由感知、追蹤、語意顏色確認、地理定位到 AP6 導航的端到端系統。
2. 以 Unity 人工確認和 Orin 二次驗證結合，降低錯誤辨識直接觸發飛行的風險。
3. 在不使用深度感測器的高空情境，以單目幾何與飛控狀態估算白車 WGS84/TWD97。
4. 建立同一 ROS2 介面貫穿 mock、HIL、室內與戶外測試的可重現驗證流程。

## 實驗階段

### Stage 1：SIL / Mock

- 測試 Unity Previous/Next/Confirm、候選 timeout、CANCEL 與狀態機。
- 注入相機中斷、模型失敗、GPS loss、過期 Track ID 與錯誤 candidate_id。

### Stage 2：HIL / 室內

- 使用真 D455、Orin NX、AP6，但 `enable_flight_control=false`。
- 驗證影像延遲、TensorRT FPS、Track ID 穩定度、白車 precision/recall、MAVLink telemetry 與安全拒絕。

### Stage 3：受控戶外

- 低速、低高度、空曠區域、人工 GUIDED 與隨時可接管。
- 先只比較估算座標與 RTK/手持 GPS ground truth，再啟用 REPOSITION。

### Stage 4：場域測試

- 改變高度、視角、光照、車輛數量、遮擋與通訊品質。
- 重複每個條件並報告平均值、標準差與失敗案例。

## 建議量化指標

- Detection：precision、recall、mAP50、false positives per minute。
- Tracking：HOTA、IDF1、MOTA、ID switches、track fragmentation。
- White verification：precision、recall、陽光/反光誤判率。
- Geolocation：horizontal RMSE、median error、CEP50、CEP95。
- System：camera-to-Unity latency、CONFIRM-to-MAVLink latency、topic rate、GPU/CPU 使用率。
- Mission：accept/reject correctness、arrival error、completion time、cancel success rate。
- Robustness：camera/GPS/MAVLink loss detection time與安全拒絕覆蓋率。

## 每次實驗記錄

先啟動系統，再於另一終端機執行：

```bash
./record_experiment.sh
```

同時保存 AP6 flight log、測試條件、相機安裝角、飛行高度、天候、ground truth 與使用的 commit/config。
