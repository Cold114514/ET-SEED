import torch
import numpy as np
import matplotlib.pyplot as plt
import json
from pdb import set_trace as bp

def geometry_interpolation(keypoints, num_steps, is_closed=False):
    # This function is for interpolation in the drawing task
    # Input is a set of keypoints of the brush, describing the trajectory of the drawing
    # This function will interpolate the keypoints to more dense points
    # input: 
    #   keypoints :[num_keypoints, 2]
    #   num_points: int
    assert type(keypoints) == np.ndarray
    assert keypoints.shape[1] == 2
    num_keypoints = keypoints.shape[0]
    # calculate the perimeter of the drawing
    if is_closed:
        perimeter = np.sum(np.linalg.norm(keypoints - np.roll(keypoints, 1, axis=0), axis=1))
    else:
        perimeter = np.sum(np.linalg.norm(keypoints - np.roll(keypoints, 1, axis=0), axis=1)[1:])
    point_distance = perimeter / num_steps
    # interpolate
    interpolated_points = []
    for i in range(num_keypoints):
        if (i == num_keypoints - 1):
            if is_closed:
                p1 = keypoints[i]
                p2 = keypoints[0]
            else:
                break
        else:
            p1 = keypoints[i]
            p2 = keypoints[i + 1]
        segment_length = np.linalg.norm(p2 - p1)
        num_points = int(segment_length / point_distance)
        if num_points == 0:
            num_points = 1
        for j in range(num_points):
            interpolated_points.append(p1 + (p2 - p1) * j / num_points)
    interpolated_points = np.array(interpolated_points)
    return interpolated_points


def draw_5PointedStar(r = 1):
    pi_val = np.pi / 180
    min_r = r * np.sin(18 * pi_val) / np.cos(36 * pi_val)
    a = [0,r]
    b = [r * np.cos(18 * pi_val), r * np.sin(18 * pi_val)]
    c = [r * np.cos(54 * pi_val), - r * np.sin(54 * pi_val)]
    d = [- r * np.cos(54 * pi_val), - r * np.sin(54 * pi_val)]
    e = [- r * np.cos(18 * pi_val), r * np.sin(18 * pi_val)]
    in_a = [min_r * np.cos(54 * pi_val),min_r * np.sin(54 * pi_val)]
    in_b = [min_r * np.cos(18 * pi_val),- min_r * np.sin(18 * pi_val)]
    in_c = [0, - min_r]
    in_d = [- min_r * np.cos(18 * pi_val),- min_r * np.sin(18 * pi_val)]
    in_e = [- min_r * np.cos(54 * pi_val),min_r * np.sin(54 * pi_val)]
    key_points = np.array([a, in_a, b, in_b, c, in_c, d, in_d, e, in_e])
    return key_points

def draw_HeartCurve(a = 1, num_keypoints = 7):
    #heart curve r = a(1-cos(\theta))
    theta = np.linspace(0, 2*np.pi, num_keypoints)
    x = a*(1-np.cos(theta))*np.sin(theta)
    y = a*(1-np.cos(theta))*np.cos(theta)
    key_points = np.stack([x,y], axis = 1)
    return key_points

def calligraphy(pinyin='dui'):
    data_file = 'assets/calligraphy/data1.json'
    with open(data_file, 'r') as f:
        data = json.load(f)
    data = data[pinyin]
    num_strokes = len(data)
    key_points = []
    save_interval = 8
    print('num_strokes', num_strokes)
    for i in range(num_strokes):
        num_keypoints = len(data[i])
        for j in range(num_keypoints):
            if j % save_interval != 0:
                continue
            key_points.append(data[i][j][0])
    key_points = np.array(key_points)/255 - 0.5
    key_points = key_points * 0.3
    key_points = key_points * np.array([1, -1])
    
    return key_points


