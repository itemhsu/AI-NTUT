import math
import csv
import numpy as np

import math

 

def reward_function(params):  


    # Read input variables

    waypoints = params['waypoints']

    closest_waypoints = params['closest_waypoints']

    heading = params['heading']

 

    # Initialize the reward with typical value

    reward = 1.0

    # Calculate the direction of the center line based 

    next_point = waypoints[closest_waypoints[1]]

    prev_point = waypoints[closest_waypoints[0]]

    # Calculate the direction in radius, arctan2(dy, dx), 

    track_direction = math.atan2(next_point[1] - prev_point[1], next_point[0] - prev_point[0])

    # Convert to degree

    track_direction = math.degrees(track_direction)

    # Calculate the difference between the track direction and the heading direction of the car

    direction_diff = abs(track_direction - heading)

    if direction_diff > 180:

        direction_diff = 360 - direction_diff

    # Penalize the reward if the difference is too large

    DIRECTION_THRESHOLD = 10.0

    if direction_diff > DIRECTION_THRESHOLD:

        reward *= 0.5

    return float(reward)

def count_waypoints_from_array(array):
    if array.shape[1] >= 6:
        centerline = array[:, 4:6]
    else:
        centerline = array[:, 0:2]
    print(f"該賽道共有 {len(centerline)} 個中心線 waypoints。")
    return len(centerline)

def check_waypoint_consistency(npy_array, csv_path, closest_wp_col_index=12):
    """
    比較 npy 中 waypoint 點數與 CSV 中使用到的最大 closest_waypoint index。
    """
    wp_len = npy_array.shape[0]
    max_csv_wp = -1

    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)  # 跳過標題列
        for row in reader:
            try:
                idx = int(row[closest_wp_col_index])
                if idx > max_csv_wp:
                    max_csv_wp = idx
            except:
                continue

    print(f"✅ waypoint 總數（.npy 檔）: {wp_len}")
    print(f"📄 CSV 中最大 closest_waypoint: {max_csv_wp}")

    if max_csv_wp >= wp_len:
        print("⚠️ 警告：CSV 中的 closest_waypoint 超過 npy 中的 waypoint 點數！")
        print("➡️ 你可能會錯過部分 reward 計算或發生 IndexError。")
    else:
        print("✅ 檢查通過：waypoint 數量與 CSV 對應一致。")


# 載入完整的 waypoints（中心線座標）
waypoints_np = np.load(r'C:\Users\盧詠林\Documents\機器學習\tracks\reinvent_base.npy')
csv_path = r'C:\Users\盧詠林\Documents\機器學習\ntut07test2clone1_traininglog\traininglog\sim-trace\training\training-simtrace\0-iteration.csv'
waypoints = waypoints_np[:, 0:2].tolist()  # 只取中心線


# 初始化變數用於統計
total_reward = 0.0
valid_rows = 0
reward_list = []

# 讀取 sim-trace csv
with open(r'C:\Users\盧詠林\Documents\機器學習\ntut07test2clone1_traininglog\traininglog\sim-trace\training\training-simtrace\0-iteration.csv', newline='') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)  # 跳過標題列
    for row in reader:
        if not row or row[0].startswith('//') or row[0] == '':
            continue
        try:
            closest_waypoint_idx = int(row[12])  # 根據你的 log 格式
            next_waypoint_idx = closest_waypoint_idx + 1 if closest_waypoint_idx + 1 < len(waypoints) else 0
            heading = float(row[4])  # 根據你的 log 格式

            params = {
                'waypoints': waypoints,
                'closest_waypoints': [closest_waypoint_idx, next_waypoint_idx],
                'heading': heading
            }
            reward = reward_function(params)
            
            # 加總reward值
            total_reward += reward
            valid_rows += 1
            reward_list.append(reward)            
            print(f'row={row}, reward={reward}')
        except Exception as e:
            print(f'row={row}, error={e}')

# 顯示統計結果
print("\n=== Reward 統計結果 ===")
print(f"總 Reward 值: {total_reward:.4f}")
print(f"有效資料筆數: {valid_rows}")
print(f"平均 Reward 值: {total_reward/valid_rows:.4f}" if valid_rows > 0 else "平均 Reward 值: 0")
print(f"最大 Reward 值: {max(reward_list):.4f}" if reward_list else "最大 Reward 值: 0")
print(f"最小 Reward 值: {min(reward_list):.4f}" if reward_list else "最小 Reward 值: 0")
print(f"Reward 標準差: {np.std(reward_list):.4f}" if reward_list else "Reward 標準差: 0")

print("\n=== 賽道Waypoints統計結果 ===")
count_waypoints_from_array(waypoints_np)

check_waypoint_consistency(waypoints_np, csv_path)
