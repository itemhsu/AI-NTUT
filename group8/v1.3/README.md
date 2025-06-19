## 🔁 第四版模型（v1.3）

本次為第三次 fine-tune，根據 v1.2 基礎進行進一步優化，特別強化彎道速度表現與方向控制靈敏度。

> ⚠️ **相比 v1.2 的調整：**
> - **動作空間增強**：將 ±15° 轉向角速度為2.0 m/s 提升至 2.5 m/s，進一步提高彎道最小速度。
> - **reward 函數更新**：額外新增轉向獎勵設計，針對轉向角< 5 / < 15 / > 15 對獎勵進行 1.4、1.0、0.8 調整。

---

### ✅ 訓練結果圖（穩定提升）

![訓練圖 v1.3](images/training_v1.3.png)

- 🟢 **獎勵分數**：波動下降，穩定上升，reward 高於前一版本。
- 🔵 **訓練完成率**：中後段穩定維持在 50–60%，代表策略逐漸成熟。
- 🔴 **評估完成率**：多數疊代達到 60~90%，整體穩定度提升。

---

### 🎥 評估影片截圖

![評估影片 v1.3](images/eval_v1.3.png)

- **Best lap**：`08.617 秒`（目前最佳）
- **Progress**：`100%`

---

### ⚙️ 模型設定摘要（v1.3）

- 🛣 賽道：`re:Invent 2018（逆時針）`
- 🕓 訓練時長：60 分鐘
- 🤖 強化演算法：PPO
- 🎥 感測器：相機

#### 🎮 動作空間（共 10 組）

- 將原先 v1.2 ±15° 轉向角速度從 2.0 m/s 提升至 2.5 m/s，以強化彎道衝刺效率

| 序號 | 轉向角 (°) | 速度 (m/s) |
|------|------------|------------|
| 0    | -30        | 1.5        |
| 1    | -30        | 2.0        |
| 2    | -15        | 2.5        |
| 3    | -15        | 3.0        |
| 4    | 0          | 3.0        |
| 5    | 0          | 4.0        |
| 6    | 15         | 2.5        |
| 7    | 15         | 3.0        |
| 8    | 30         | 1.5        |
| 9    | 30         | 2.0        |

---

#### 🔧 超參數設定（v1.3）

| 參數名稱                          | 數值      |
|----------------------------------|-----------|
| Batch Size                       | 64        |
| Number of Epochs                | 10        |
| Learning Rate                   | 0.00005   |
| Entropy                         | 0.001     |
| Discount Factor                 | 0.99      |
| Loss Type                       | Huber     |
| Policy Update Frequency (exp)  | 20        |

---

### 🧠 獎勵函數程式碼（v1.3）

| 比較項目       | v1.2 設定                                                    | v1.3 設定                                                   | 差異重點說明                      |
| ---------- | ---------------------------------------------------------- | --------------------------------------------------------- | --------------------------- |
| **中心獎勵結構** | 使用 `marker_1`, `marker_2`, `marker_3` 計算距離中心的分級獎勵          | 相同邏輯，但更簡化實作，去除不必要變數                                       | ✅ 保留核心機制、簡化程式碼結構            |
| **出界懲罰**   | `if not all_wheels_on_track: return 1e-3`                  | 相同邏輯                                                      | ➖ 無異動                       |
| **轉向獎勵設計** | 僅對彎道與速度配對做判斷，不對 steering 單獨分級                              | 🆕 額外新增 `steering < 5 / < 15 / > 15` 對獎勵進行 1.4、1.0、0.8 調整 | ✅ 新增**方向穩定性強化**機制           |
| **速度條件獎勵** | 若 `steering <10 且 speed ≥3`：1.5 倍，若大角轉速慢則 1.2 倍，其餘懲罰 0.8 倍 | 類似邏輯，但將「其餘懲罰」調整為 0.9 倍（略降低罰則）                             | ⚠️ 更加保守控制獎勵懲罰比例，避免過早削弱策略探索性 |

```python
def reward_function(params):
    import math

    all_wheels_on_track = params['all_wheels_on_track']
    distance_from_center = params['distance_from_center']
    track_width = params['track_width']
    speed = params['speed']
    steering = abs(params['steering_angle'])

    reward = 1e-3

    # 中心獎勵
    if distance_from_center <= 0.1 * track_width:
        reward = 1.0
    elif distance_from_center <= 0.25 * track_width:
        reward = 0.5
    elif distance_from_center <= 0.5 * track_width:
        reward = 0.1
    else:
        return 1e-3

    # 轉向角控制：鼓勵穩定方向，小角度高獎勵
    if steering < 5:
        reward *= 1.4
    elif steering < 15:
        reward *= 1.0
    else:
        reward *= 0.8  # 過度轉向懲罰

    # 速度控制：在合適角度下鼓勵加速
    if speed >= 3.0 and steering < 10:
        reward *= 1.5
    elif speed <= 2.0 and steering > 15:
        reward *= 1.2
    else:
        reward *= 0.9

    return float(reward)


