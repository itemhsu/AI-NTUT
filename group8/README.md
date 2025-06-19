# 2025_AWS-DeepRacer_NTUT_第8組

## 📘Project Overview / 專案簡介

This project applies reinforcement learning strategies to AWS DeepRacer — starting from a stable baseline model and progressively achieving sub-9 second lap times on the re:Invent 2018 track, with detailed records of each training stage.

本專案應用於 AWS DeepRacer 的強化學習策略 — 從穩定的基準模型出發，最後逐步達成 re:Invent 2018 賽道的單圈低於 9 秒表現，並詳細記錄了逐步訓練過程。

---



## 🧪 訓練版本一覽

| 版本       | 動作空間變化                                  | reward 函數邏輯                          | 超參數調整                         | 圈速最佳    | 說明重點                                      |
| -------- | --------------------------------------- | ------------------------------------ | ----------------------------- | ------- | ----------------------------------------- |
| **v1.0** | 基礎 10 組：±30° / ±15° / 0° 對應 1.0–4.0 m/s | 極簡中心線判斷，只要沒出界就給 1.0                  | 預設參數（learning rate 0.0003、entropy 0.01）  | 11.062s | ✅ **穩定完賽的 baseline 模型**，適合作為 fine-tune 起點 |
| **v1.1** | ±30° 與 ±15° 各增 0.5 m/s                  | 增加速度獎勵與出界懲罰，強調速度與穩定性                 | 預設參數不變     | 9.011s  | 🔧 初步進行速度優化與懲罰設計，模型可穩定加速，圈速明顯進步           |
| **v1.2** | ±15°  2.5 m/s 增加到 3.0 m/s ，測試極限速度           | 引入多層 steering-speed 聯動懲罰             | learning rate 與 entropy 分別調降至 **0.00005**  和  **0.001**    | 8.817s  | 🧠 精細化轉向 + 速度配對，降低亂動作機率，進一步提升彎道效率與穩定性     |
| **v1.3** | ±15°  2.0 m/s 增加到 2.5 m/s ，進一步提升最小速度    | steering 分級處理（<5°, <15°, >15°）與調整懲罰比 | 繼續使用調整後的參數 | 8.617s  | 🪄 對轉向獎懲進行精調，讓模型在彎道決策更穩定，學習速度更快           |




📄 詳細內容請見：`v1.0/README.md`、`v1.1/README.md` 等

