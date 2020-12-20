import pandas as pd
import glob
import os.path
import datetime
import os
import matplotlib.pyplot as plt
import utm
import math
import numpy as np


def read_file(route_file):
    # Read a single trajectory file from taxi, walk, etc. folder
    points = pd.read_csv(route_file, skiprows=1, header=None)

    return points


def convert_lat(data):
    # This function converts latitude to utm coordinate and places all the y coordinates in a list

    for x in range(len(data)):
        u = utm.from_latlon(data[2][x], data[3][x])
        data[2][x] = u[0]
        data[3][x] = u[1]


def compute_dist(prevx, prevy, currx, curry):
    dist = math.sqrt((currx - prevx) ** 2 + (curry - prevy) ** 2)
    return dist


def get_avg_time(data):
    list1 = []
    avg = 0

    for x in range(len(data) - 1):
        # Add the difference between the next timestamp and the previous time stamp to a lost
        # Multiply by 86400 to converts days to seconds
        list1.append((data[0][x + 1] - data[0][x]) * 86400)
    if len(list1) > 0:
        avg = sum(list1) / len(list1)
    return avg


def get_distance(data):
    all_dist = []

    for x in range(len(data) - 1):
        # print('Count: ', count)
        dist = compute_dist(data[2][x], data[3][x], data[2][x + 1], data[3][x + 1])
        all_dist.append(dist)

    total_dist = sum(all_dist)
    return total_dist


def get_time(data):
    total_time = data[1].iloc[-1] - data[1][1]
    return total_time*86400


def get_min_max_speed(data):
    times = []
    dists = []
    for x in range(len(data) - 1):
        # Add the difference between the next timestamp and the previous time stamp to a lost
        # Multiply by 86400 to converts days to seconds
        time = (data[1][x + 1] - data[1][x]) * 86400
        if time == 0:
            time = 0.001
        times.append(time)
        dist = compute_dist(data[2][x], data[3][x], data[2][x + 1], data[3][x + 1])
        if dist == 0:
            dist = 0.001
        dists.append(dist)
    dists = np.array(dists)
    times = np.array(times)
    speeds = dists / times
    min_speed = np.amin(speeds)
    max_speed = np.amax(speeds)

    return min_speed, max_speed


def print_to_file(total_time, total_distance, avg_speed, min_speed, max_speed, target):
    with open(r'train_data.csv', "a") as f:
        f.write("{:.2f}".format(total_time))
        f.write(",")
        f.write("{:.2f}".format(total_distance))
        f.write(",")
        f.write("{:.2f}".format(avg_speed))
        f.write(",")
        f.write("{:.2f}".format(min_speed))
        f.write(",")
        f.write("{:.2f}".format(max_speed))
        f.write(",")
        f.write(str(target))
        f.write("\n")


def read_trajectory(folder):
    labels = None
    # Find all .plt file in the trajectory file of the user
    plt_files = glob.glob(os.path.join(folder, 'Train', '*.csv'))
    for f in plt_files:
        print(f)
        df = read_file(f)
        convert_lat(df)
        total_distance = get_distance(df)
        if total_distance == 0:
            continue
        total_time = get_time(df)
        if total_time == 0:
            continue
        avg_speed = total_distance / total_time
        if avg_speed > 90:
            continue
        min_speed, max_speed = get_min_max_speed(df)
        # 14 for bikes, 45 for motor, 6 for walk, 90 for train
        if max_speed > 90:
            continue
        print_to_file(total_time, total_distance, avg_speed, min_speed, max_speed, 4)


if __name__ == '__main__':

    file = r''
    read_trajectory(file)



