import xml.dom.minidom as xmldom
import os
from math import sin, asin, cos, radians, fabs, sqrt
from torch.utils.data import Dataset
import random
import torch
import pandas as pd 
import numpy as np
import csv

EARTH_RADIUS = 6371      # 地球平均半径大约6371km

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

def get_xml(path='./dataset/kml'):
    xml_name = os.listdir(path)
    
    for name in xml_name:
        track_points, time_points = parse_xml(path + '/' + name)
        #print(len(track_points),end=' ')
        #print(len(time_points))
        # 对提取到的信息进行处理
        csv_path = './dataset/csv'   
        time_points = _parse_time(time_points)
        track_points = _parse_point(track_points)
        a = np.array(track_points)
        b = np.array(time_points)

        e=np.append(a,b,axis=1)
        
        write_csv(e,csv_path+'/'+name.split('.')[0]+'.csv')

        print("succssfully wirte {}".format(name.split('.')[0]+'.csv'))
    return 

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

# 处理点，两个经纬度坐标转换为[x,h,dx,dh]
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

def write_csv(target, path):
    
    # 在CSV文件中写入标题行
    with open(path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['x', 'h', 'dx', 'dh', 'time'])
    file.close()

    # 将NumPy数组写入CSV文件
    with open(path, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(target)
    file.close()
    
    return

if __name__ == '__main__':
    get_xml()