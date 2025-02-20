import torch
import numpy as np
import matplotlib.pyplot as plt
import json
from pdb import set_trace as bp

def threedim_geometry_interpolation(keypoints, num_steps, is_closed=False):
    # This function is for interpolation in the drawing task
    # Input is a set of keypoints of the brush, describing the trajectory of the drawing
    # This function will interpolate the keypoints to more dense points
    # input: 
    #   keypoints :[num_keypoints, 2]
    #   num_points: int
    assert type(keypoints) == np.ndarray
    assert keypoints.shape[1] == 3
    num_keypoints = keypoints.shape[0]
    # calculate the perimeter of the drawing
    if is_closed:
        perimeter = np.sum(np.linalg.norm(keypoints - np.roll(keypoints, 1, axis=0), axis=1))
    else:
        perimeter = np.sum(np.linalg.norm(keypoints - np.roll(keypoints, 1, axis=0), axis=1)[1:])
        # print(perimeter)
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
        # print("seg_length:",segment_length)
        num_points = int(segment_length / point_distance)
        if num_points == 0:
            num_points = 1
        for j in range(num_points):
            interpolated_points.append(p1 + (p2 - p1) * j / num_points)
    interpolated_points = np.array(interpolated_points)
    # print("int_points:",interpolated_points)
    return interpolated_points

def rotate_cap(d=0.5):
    a = [0,0,0]
    b = [0,d,0]
    key_points = np.array([a, b])
    return key_points