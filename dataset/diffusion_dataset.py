import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
from torch.utils.data  import Dataset

from utils.utils_3d import get_extrinsic_matrix, pcd2pointmap
from truckscenes import TruckScenes
from truckscenes.utils.splits import train, val, test, mini_train, mini_val
from pypcd4 import PointCloud
from collections import defaultdict
from typing import List, Dict, Tuple, Union


sensor_meta = dict(
    RADAR_RIGHT_BACK=False, 
    RADAR_RIGHT_SIDE=False, 
    RADAR_RIGHT_FRONT=False, 
    RADAR_LEFT_FRONT=True,
    RADAR_LEFT_SIDE=False, 
    RADAR_LEFT_BACK=False, 
    LIDAR_LEFT= False,
    LIDAR_RIGHT=False, 
    LIDAR_TOP_FRONT=False, 
    LIDAR_TOP_LEFT=True, 
    LIDAR_TOP_RIGHT=False, 
    LIDAR_REAR=False, 
    CAMERA_LEFT_FRONT=True, 
    CAMERA_LEFT_BACK=False, 
    CAMERA_RIGHT_FRONT=False, 
    CAMERA_RIGHT_BACK=False
)
class TruckScenesDiffusionDataset(object):
    def __init__(self, truckscenes:TruckScenes, split:str, meta=None):
        '''split: {"train", "val", "test", "mini_train", "mini_val"}'''
        if meta is None:
            meta = sensor_meta
        self.meta = meta
        self.trucksc = truckscenes
        self.split = globals()[split]
        
    def reshuffle(self):
        ...
    def get_intrinsic(self, sensor:str)-> Dict[str, np.ndarray]:
        '''sensor: sensor channel name '''
        token = self.trucksc.field2token('sensor', 'channel', sensor)[0]
        calibrated_token = self.trucksc.field2token('calibrated_sensor', 'sensor_token', token)[0]
        sensor = self.trucksc.get('calibrated_sensor', calibrated_token)
        intrinsic = np.array(sensor['camera_intrinsic'], dtype=np.float32).reshape(3, 3)
        return intrinsic
    
    def get_tf(self, srcs:Union[List, str], tgt)-> Dict[str, np.ndarray]:
        '''src & tgt: sensor channel name '''
        if type(srcs) is str: srcs=[srcs]
        tgt_token = self.trucksc.field2token('sensor', 'channel', tgt)[0]
        calibrated_token = self.trucksc.field2token('calibrated_sensor', 'sensor_token', tgt_token)[0]
        tgt_sensor = self.trucksc.get('calibrated_sensor', calibrated_token)
        tgt_extrinsic = get_extrinsic_matrix(tgt_sensor['rotation'], tgt_sensor['translation'], scalar_first=True)
        
        extrinsics = dict()
        for src in srcs:
            src_token = self.trucksc.field2token('sensor', 'channel', src)[0]
            calibrated_token = self.trucksc.field2token('calibrated_sensor', 'sensor_token', src_token)[0]
            src_sensor = self.trucksc.get('calibrated_sensor', calibrated_token)
            src_extrinsic = get_extrinsic_matrix(src_sensor['rotation'], src_sensor['translation'], scalar_first=True)    
            extrinsics[src]= tgt_extrinsic @ np.linalg.inv(src_extrinsic)
        return extrinsics
    
    def aggregate(self, sensors:Union[List, str], sample_token:str):
        """Aggregate the data for the given sensors and sample token."""
        
        # * Utilize truckscenes utility to get sample data.
        if type(sensors) is not list:
            sensors = [sensors]
        
        dict_item = {sensor:defaultdict(list) for sensor in sensors}
        timestmaps = list()
        
        while sample_token != '':
            sample = self.trucksc.get('sample', sample_token)
            for sensor in sensors:
                fpth, label, meta = self.trucksc.get_sample_data(sample['data'][sensor])
                dict_item[sensor]['file'].append(fpth)
                dict_item[sensor]['label'].append(label)
                dict_item[sensor]['intrinsic'].append(meta)
            timestmaps.append(sample['timestamp'])
            sample_token = sample['next']
        
        return dict_item, timestmaps
    
    def __getitem__(self, index):
        # Get the data for the given index
        data = self.trucksc.scene[index]    
        available_sensors = list(sensor for sensor in self.meta if self.meta[sensor])
        dict_item, timestamps = self.aggregate(available_sensors, data['first_sample_token'])
        return dict_item, timestamps
    
    def __len__(self):
        # Return the length of the dataset
        return len(self.truckscenes)
    
    
if __name__ == "__main__":
    # Example usage
    import cv2 
    truckscenes = TruckScenes(version='v1.0-mini', dataroot='/data/truckscenes', verbose=False)
    dataset = TruckScenesDiffusionDataset(truckscenes, split="train")
    
    dict_item, timestamps = dataset[0]
    tgt = "CAMERA_LEFT_FRONT"
    
    img = cv2.imread(dict_item[tgt]['file'][0])
    lidar = PointCloud.from_path(dict_item['LIDAR_TOP_LEFT']['file'][0])[('x','y','z')]
    radar = PointCloud.from_path(dict_item['RADAR_LEFT_FRONT']['file'][0])
    dict_extrinsic = dataset.get_tf(['LIDAR_TOP_LEFT', 'RADAR_LEFT_FRONT'] ,tgt)
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(lidar.numpy())
    o3d.visualization.draw_geometries([pcd])
    
    
    lidar_pointmap = pcd2pointmap(lidar.numpy(), dict_extrinsic['LIDAR_TOP_LEFT'], dataset.get_intrinsic(tgt), img.shape[:2])
    
    import matplotlib.pyplot as plt
    i, j, _ = np.where(lidar_pointmap>0)
    lidar_pointmap = np.linalg.norm(lidar_pointmap, axis=2)
    norm = plt.Normalize(vmin=np.min(lidar_pointmap), vmax=np.max(lidar_pointmap))
    cmap = plt.cm.plasma
    colors = cmap(norm(lidar_pointmap))[...,:3]*255
    colors = colors.astype(np.uint8)
    img[i, j, :] = colors[i, j , :3]
    
    
    cv2.imshow('img', img)
    cv2.waitKey(0)
    exit()
    
    