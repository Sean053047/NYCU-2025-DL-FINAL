import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
from torch.utils.data  import Dataset

from PIL import Image
from truckscenes import TruckScenes
from truckscenes.utils.splits import train, val, test, mini_train, mini_val
from truckscenes.utils.data_classes import LidarPointCloud, RadarPointCloud
from collections import defaultdict
from typing import List, Dict, Tuple, Union

from truckscenes.utils.visualization_utils import TruckScenesExplorer
sensor_meta = dict(
    RADAR_RIGHT_BACK=False, 
    RADAR_RIGHT_SIDE=False, 
    RADAR_RIGHT_FRONT=False, 
    RADAR_LEFT_FRONT=True,
    RADAR_LEFT_SIDE=True, 
    RADAR_LEFT_BACK=False, 
    LIDAR_LEFT= True,
    LIDAR_RIGHT=False, 
    LIDAR_TOP_FRONT=True, 
    LIDAR_TOP_LEFT=False, 
    LIDAR_TOP_RIGHT=False, 
    LIDAR_REAR=False, 
    CAMERA_LEFT_FRONT=True, 
    CAMERA_LEFT_BACK=True, 
    CAMERA_RIGHT_FRONT=True, 
    CAMERA_RIGHT_BACK=True
)

class TruckScenesDiffusionDataset(object):
    def __init__(self, data_root:str, version:str,  split:str, chunk_size:int, meta=None, image_transform=None, point_transform=None):
        '''split: {"train", "val", "test", "mini_train", "mini_val"}'''
        if meta is None:
            meta = sensor_meta
        self.meta = meta
        self.trucksc = TruckScenes(version, data_root, verbose=False)
        self.chunk_size = chunk_size
        
        self.split = globals()[split]
        self.scene_tokens, self.sd_chunks, self.t_chunks = self._split_scene2chunks()
        self.image_transform = image_transform
        self.point_transform = point_transform
        
        self.locate = "lf" # Position locate : lf, lb, rf, rb
    def __len__(self):
        return len(self.sd_chunks)
    
    def _split_scene2chunks(self, key_frame_only=True):
        '''Iterate over all scenes and split them into chunks.
        1. Freely split chunk 
        '''
        print("=============Splitting video into chunks...=============")
        scene_tokens = []
        sd_chunks = []
        t_chunks = []
        for scene_name in self.split:
            scene_tk = self.trucksc.field2token('scene', 'name', scene_name)[0]
            scene_tokens.append(scene_tk)
            scene = self.trucksc.get('scene', scene_tk)
            for sd_tks, tts in self.aggregate_scene(scene):
                sd_chunks.append(dict(sd_tks))
                t_chunks.append(dict(tts))
        
        return scene_tokens, sd_chunks, t_chunks
        
    def aggregate_scene(self, scene):
        """Aggregate the data for the given sensors and sample token."""
        # * Utilize truckscenes utility to get sample data.
        sd_tokens = defaultdict(list)
        timestamps = defaultdict(list)
        sample_token = scene['first_sample_token']
        count = 0
        while sample_token != '':
            sample = self.trucksc.get('sample', sample_token)
            timestamps['sample'].append(sample['timestamp'])
            for sensor, v in self.meta.items():
                if not v: continue # meta[sensor] == False
                sample_data = self.trucksc.get('sample_data', sample['data'][sensor])
                timestamps[sensor].append(sample_data['timestamp'])
                sd_tokens[sensor].append(sample['data'][sensor])
            
            count += 1
            if count % self.chunk_size == 0: # Return sample per N data
                yield sd_tokens, timestamps
                sd_tokens.clear()
                timestamps.clear()
            sample_token = sample['next']
        if len(timestamps) > 0:
            print(f'{scene["name"]} abandon redundant ', len(timestamps), 'samples.')
    
    def read_image_from_token(self, tokne:str):
        '''Read image from token'''
        sample_data = self.trucksc.get('sample_data', tokne)
        file_path = os.path.join(self.trucksc.data_root, sample_data['filename'])
        image = Image.Open(file_path)
        image = image.convert('RGB')
        if self.image_transform is not None:
            image = self.image_transform(image)
        return image
    
    def relocate(self):
        '''This function aims to set different location for training.
        Use different pairs of sensors to train the model.
        e.g. (CAM RIGHT FRONT,  RADAR RIGHT BACK), (CAM LEFT FRONT, RADAR LEFT BACK)
        '''
        ...
    
    def __getitem__(self, index):
        # Get the data for the given index
        # Todo
        sd_tokens, timestamps = self.sd_chunks[index], self.t_chunks[index]    
        video_chunk = [self.read_image_from_token(img_tk)
                       for img_tk in sd_tokens['CAMERA_LEFT_FRONT']]
        
        for sd_tk in sd_tokens:
            if 'CAMERA' in sd_tk: continue
            
    
    def collate_bn(self, batch_data):
        # Todo
        ...
    
from truckscenes.utils.visualization_utils import TruckScenesExplorer


if __name__ == "__main__":
    # Example usage
    import cv2 
    
    dataset = TruckScenesDiffusionDataset(data_root='/data/truckscenes', version='v1.0-mini', split='mini_train', chunk_size=12)
    dataset[0]