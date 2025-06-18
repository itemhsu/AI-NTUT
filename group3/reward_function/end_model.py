import math

FAIL_VALUE = 1e-26
MAX_TURN = 30
TURN_STEP = 5
TOP_SPEED = 8
RANGE = 6
TURN_THRESHOLD = 10  # 超過此角度才認定為真正轉彎

LAST_POINT = -1

def calc_distance(x1, x2, y1, y2):
    return ((x1 - x2)**2 + (y1 - y2)**2)**0.5

def gaussian_val(x, mu, sigma):
    return math.exp(-((x - mu) ** 2) / (2 * (sigma ** 2)))

def calc_turn_angle(waypoints, current_idx):
    total_wp = len(waypoints)
    prev_idx = (current_idx - 1) % total_wp
    next_idx = (current_idx + 1) % total_wp

    x1, y1 = waypoints[prev_idx]
    x2, y2 = waypoints[current_idx]
    x3, y3 = waypoints[next_idx]

    vec1 = (x2 - x1, y2 - y1)
    vec2 = (x3 - x2, y3 - y2)

    dot = vec1[0]*vec2[0] + vec1[1]*vec2[1]
    mag1 = math.hypot(*vec1)
    mag2 = math.hypot(*vec2)

    if mag1 == 0 or mag2 == 0:
        return 0

    cos_theta = max(min(dot / (mag1 * mag2), 1), -1)
    angle = math.acos(cos_theta) * 180 / math.pi
    return angle

def reward_function(params):
    global LAST_POINT

    pos_x = params['x']
    pos_y = params['y']
    curr_speed = params['speed']
    steer = abs(params['steering_angle'])
    on_track = params['all_wheels_on_track']
    reversed_drive = params.get('is_reversed', False)
    track_w = params['track_width']
    center_dist = params['distance_from_center']
    progress = params['progress']

    if not on_track or reversed_drive:
        return float(FAIL_VALUE)

    waypoints = params['waypoints']
    current_wp = params['closest_waypoints'][1]
    total_wp = len(waypoints)

    wp_prev = (current_wp - RANGE) % total_wp
    wp_next = (current_wp + RANGE) % total_wp
    x1, y1 = waypoints[wp_prev]
    x2, y2 = waypoints[wp_next]
    mid_x = (x1 + x2) / 2
    mid_y = (y1 + y2) / 2

    to_line_dist = calc_distance(mid_x, pos_x, mid_y, pos_y)

    rline_reward = 1e-3
    if current_wp == LAST_POINT:
        rline_reward = 1e-3
    else:
        LAST_POINT = current_wp
        if to_line_dist < 0.05:
            rline_reward = 1.0
        elif to_line_dist < 0.1:
            rline_reward = 0.8
        elif to_line_dist < 0.15:
            rline_reward = 0.5

    center_reward = gaussian_val(center_dist, 0.3, track_w / 4) * 10

    # 判斷是否是大轉彎區段
    turn_angle = calc_turn_angle(waypoints, current_wp)

    # 抖動懲罰或轉彎獎勵
    steer_reward = 1.0
    if turn_angle < TURN_THRESHOLD:
        if steer > 1.0:  # 小幅度亂轉
            steer_reward *= 0.3
    else:
        if steer > 1.0:  # 真正轉彎時給獎勵
            steer_reward *= 1.5

    spd_reward = (curr_speed / TOP_SPEED) * 100

    total_reward = rline_reward + center_reward + steer_reward + spd_reward

    if 70 < progress < 100:
        total_reward *= (1 + progress / 100)
    elif progress >= 100:
        total_reward *= 5

    return float(total_reward)
