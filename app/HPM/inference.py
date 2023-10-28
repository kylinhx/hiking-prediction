import torch
import numpy as np
import pandas as pd
import xml.dom.minidom as xmldom
import os
from math import sin, asin, cos, radians, fabs, sqrt
from torch.utils.data import Dataset
import random
import torch
import pandas as pd 
import numpy as np
import csv

import torch
import torch.nn as nn
from torchsummary import summary

# 多层感知机
class MLP(nn.Module):
    def __init__(self):

        super(MLP, self).__init__()
        self.fc1 = nn.Linear(4, 16)
        self.fc2 = nn.Linear(16, 32)
        self.fc3 = nn.Linear(32, 64)
        self.fc4 = nn.Linear(64, 128)
        self.fc5 = nn.Linear(128, 512)
        self.fc6 = nn.Linear(512, 1)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.relu(self.fc5(x))

        x = self.fc6(x)
        return x

class HPM(nn.Module):
    def __init__(self):
        super(HPM, self).__init__()
        self.layer1 = MLP()

    def forward(self, x):
        return self.layer1(x)



EARTH_RADIUS = 6371      # 地球平均半径大约6371km

# 处理时间，将时间换算成秒数
def _parse_time(time_points):
    #[['2022-04-09T11:15:45Z']]

    begin_second = int(time_points[0][0][-3:-1])
    begin_mimute = int(time_points[0][0][-6:-4])
    begin_hour = int(time_points[0][0][-9:-7])
    begin_data = int(time_points[0][0][-12:-10])

    end_second = int(time_points[0][0][-3:-1])
    end_mimute = int(time_points[0][0][-6:-4])
    end_hour = int(time_points[0][0][-9:-7])

    begin = begin_second + begin_mimute * 60 + begin_hour * 3600
    end = end_second + end_mimute * 60 + end_hour * 3600

    for index, item in enumerate(time_points):
        second = int(item[0][-3:-1])
        minute = int(item[0][-6:-4])
        hour = int(item[0][-9:-7])
        data = int(item[0][-12:-10])

        if data == begin_data:
            time = second + minute*60 + hour*3600 - begin
        else:
            time = second + minute*60 + hour*3600 - begin + 3600 * 24
        time_points[index] = [time]
    
    return time_points

def parse_xml(fn):
    xml_file = xmldom.parse(fn)
    print(fn)
    root = xml_file.documentElement

    track = root.getElementsByTagName('gx:Track')[0]
    point = track.getElementsByTagName('gx:coord')
    when = track.getElementsByTagName('when')

    track_points = [item.firstChild.data.split(' ') for item in point]
    time_points = [item.firstChild.data.split(' ') for item in when]
    


    return track_points, time_points

def hav(theta):
    s = sin(theta / 2)
    return s * s

def get_distance_hav(lat0, lng0, lat1, lng1):
    # 用haversine公式计算球面两点间的距离
    # 经纬度转换成弧度
    lat0 = radians(lat0)
    lat1 = radians(lat1)
    lng0 = radians(lng0)
    lng1 = radians(lng1)
    dlng = fabs(lng0 - lng1)
    dlat = fabs(lat0 - lat1)
    h = hav(dlat) + cos(lat0) * cos(lat1) * hav(dlng)
    distance = 2 * EARTH_RADIUS * asin(sqrt(h))      # km
    return distance

def _parse_point(track_points):
    # [['116.78209755023478', '40.59671728178273', '181.48163']]
    pre_latitude = float(track_points[0][0])
    pre_altitude = float(track_points[0][1])
    pre_height = float(track_points[0][2])

    first_height = float(track_points[0][2])

    x = 0
    for index, item in enumerate(track_points):
        latitude = float(item[0])
        altitude = float(item[1])
        height  = float(item[2])

        dh = height - pre_height

        dx = get_distance_hav(pre_altitude, pre_latitude, altitude, latitude)
        x += dx

        pre_latitude = latitude
        pre_altitude = altitude
        pre_height = height

        track_points[index] = [x, height-first_height, dx, dh]

    #print(track_points)
    return track_points

class HPM_inference(object):
    def __init__(self, model_path):
        self.model = HPM()
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

    def convert_kml_to_data(self, kml_file):
        track_points, time_points = parse_xml(kml_file)
        time_points = _parse_time(time_points)
        track_points = _parse_point(track_points)
        a = np.array(track_points)
        b = np.array(time_points)

        e=np.append(a,b,axis=1)

        with torch.no_grad():
            X_new_tensor = torch.tensor(e[:,:-1], dtype=torch.float32)
            y_new_pred = self.model(X_new_tensor).numpy()

        return y_new_pred
    
    def predict(self, data):
        with torch.no_grad():
            output = self.model(data)
            prediction = torch.argmax(output, dim=1)
            return prediction.numpy()