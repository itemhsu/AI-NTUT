import math

FAIL_VALUE = 1e-26
MAX_TURN = 30
TURN_STEP = 5
TOP_SPEED = 8
RANGE = 6

LAST_POINT = -1

def calc_distance(x1, x2, y1, y2):
    return ((x1 - x2)**2 + (y1 - y2)**2)**0.5

def gaussian_val(x, mu, sigma):
    return math.exp(-((x - mu) ** 2) / (2 * (sigma ** 2)))

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

    center_reward = gaussian_val(center_dist, 0, track_w / 4) * 10

    steer_reward = 1.0
    if steer > MAX_TURN / TURN_STEP:
        steer_reward *= 0.5

    spd_reward = (curr_speed / TOP_SPEED) * 100

    total_reward = rline_reward + center_reward + steer_reward + spd_reward

    if 70 < progress < 100:
        total_reward *= (1 + progress / 100)
    elif progress >= 100:
        total_reward *= 5

    return float(total_reward)