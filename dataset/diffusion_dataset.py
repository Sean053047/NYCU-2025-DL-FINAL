import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
from torch.utils.data  import Dataset

from PIL import Image
from truckscenes import TruckScenes
from truckscenes.utils.geometry_utils import view_points
from truckscenes.utils.splits import train, val, test, mini_train, mini_val
from truckscenes.utils.data_classes import LidarPointCloud, RadarPointCloud, PointCloud
from collections import defaultdict
from typing import List, Dict, Tuple, Union
from pyquaternion import Quaternion


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

class TruckScenesDiffusionDataset(Dataset):
    def __init__(self, 
                 data_root:str, 
                 version:str,  
                 split:str, 
                 chunk_size:int, 
                 image_shape:Tuple[int, int]=(480, 832),
                 meta=None, image_transform=None, point_transform=None):
        '''split: {"train", "val", "test", "mini_train", "mini_val"}'''
        # Available sensors
        if meta is None:
            meta = sensor_meta
        self.meta = meta
        self.trucksc = TruckScenes(version, data_root, verbose=False)
        self.chunk_size = chunk_size
        self.image_shape = np.array(image_shape) # (row, col)
        
        self.split = globals()[split]
        self.scene_tokens, self.sd_chunks, self.t_chunks = self._split_scene2chunks()
        self.image_transform = image_transform
        self.point_transform = point_transform
        
        # ? Currently used devices
        self.CAM = "CAMERA_LEFT_FRONT"
        self.POINT  = "LIDAR_LEFT"
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
    
    def read_image_from_token(self, token:str):
        '''Read image from token'''
        # sample_data = self.trucksc.get('sample_data', token)
        file_path = self.trucksc.get_sample_data_path(token)
        image = Image.open(file_path)
        image = image.convert('RGB')
        if self.image_transform is not None:
            image = self.image_transform(image)
        
        return image
    
    def read_pointmap_from_token(self, pt_token:str, img_token:str, keep_intensity=False):
        if 'LIDAR' in self.POINT:
            pc = LidarPointCloud.from_file(self.trucksc.get_sample_data_path(pt_token))
        else:
            pc = RadarPointCloud.from_file(self.trucksc.get_sample_data_path(pt_token))
        pointsensor = self.trucksc.get('sample_data', pt_token)
        cam = self.trucksc.get('sample_data', img_token)
        org_im_shape = np.array([cam['height'], cam['width']])
        # Points live in the point sensor frame. So they need to be transformed
        # via global to the image plane.
        # First step: transform the pointcloud to the ego vehicle frame
        # for the timestamp of the sweep.
        cs_record = self.trucksc.get('calibrated_sensor', pointsensor['calibrated_sensor_token'])
        pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix)
        pc.translate(np.array(cs_record['translation']))

        # Second step: transform from ego to the global frame.
        poserecord = self.trucksc.get('ego_pose', pointsensor['ego_pose_token'])
        pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix)
        pc.translate(np.array(poserecord['translation']))

        # Third step: transform from global into the ego vehicle frame
        # for the timestamp of the image.
        poserecord = self.trucksc.get('ego_pose', cam['ego_pose_token'])
        pc.translate(-np.array(poserecord['translation']))
        pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix.T)

        # Fourth step: transform from ego into the camera.
        cs_record = self.trucksc.get('calibrated_sensor', cam['calibrated_sensor_token'])
        pc.translate(-np.array(cs_record['translation']))
        pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix.T)
        
        # modified intrinsic matrix to new image shape, and project
        intrinsic = np.array(cs_record['camera_intrinsic'])
        ratio_r, ratio_c = self.image_shape / org_im_shape
        intrinsic[0, :] *= ratio_c
        intrinsic[1, :] *= ratio_r
        depths = pc.points[2, :]
        pc_im_idx = view_points(pc.points[:3, :], intrinsic, normalize=True) # (col, row, 1)
        
        mask = np.ones(depths.shape[0], dtype=bool)
        min_dist = 0.5
        mask = np.logical_and(mask, depths > min_dist)
        mask = np.logical_and(mask, pc_im_idx[0, :] > 1)
        mask = np.logical_and(mask, pc_im_idx[0, :] < self.image_shape[1] - 1) # dim: col
        mask = np.logical_and(mask, pc_im_idx[1, :] > 1)
        mask = np.logical_and(mask, pc_im_idx[1, :] < self.image_shape[0] - 1) # dim: row
        pc_im_idx = pc_im_idx[[1, 0, 2], :][:2, mask] # change to (row, col)
        pc_im_idx = np.round(pc_im_idx).astype(np.int32)
        pc = pc.points[:, mask]

        # * Deal with overlapping points at same pixel.
        # * Keep the point with high intensity
        arg = np.argsort(pc[3, :])[::-1] # sort by intensity
        pc = pc[:, arg]
        pc_im_idx = pc_im_idx[:, arg]
        
        # * Assign pixel with corresponding point location
        pointmap_mask = np.full((self.image_shape[0], self.image_shape[1]), False)
        pointmap = np.full((4, self.image_shape[0], self.image_shape[1]), 0.0)
        unique_pc_im_idx, arr_idx = np.unique(pc_im_idx, axis=1, return_index=True)
        unique_pc = pc[:, arr_idx]
        row, col = unique_pc_im_idx
        
        pointmap_mask[row, col] = True
        pointmap[:, row, col] = unique_pc[:4, :]        
        if not keep_intensity:
            pointmap = pointmap[:3, :, :] # remove intensity channel
        
        # Brutal force (have check above method is equivalent to naive method.)
        # pointmap_mask = np.full((self.image_shape[0], self.image_shape[1]), False)
        # pointmap = np.full((self.image_shape[0], self.image_shape[1], 4), 0.0)
        # for i in range(pc_im_idx.shape[1]):
        #     row, col = pc_im_idx[:, i]
        #     if not pointmap_mask[row, col]:
        #         pointmap[row, col, :] = pc[:, i]
        #         pointmap_mask[row, col] = True
        
        return torch.from_numpy(pointmap), torch.from_numpy(pointmap_mask)
        
    def relocate(self):
        '''This function aims to set different location for training.
        Use different pairs of sensors to train the model.
        e.g. (CAM RIGHT FRONT,  RADAR RIGHT BACK), (CAM LEFT FRONT, RADAR LEFT BACK)
        '''
        # Todo 
        self.CAM = "CAMERA_LEFT_FRONT"
        self.POINT = "LIDAR_TOP_FRONT"
    
    def render_point(self, pc:PointCloud):
        import open3d as o3d
        pc = pc.points[:3, :].T
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc)
        o3d.visualization.draw_geometries([pcd])        
    def render_image(self, img:np.ndarray):
        cv2.imshow('tmp', img)
        cv2.waitKey(0)
        
    def __getitem__(self, index):
        # Get the data for the given index
        # Todo
        sd_tokens, timestamps = self.sd_chunks[index], self.t_chunks[index]    
        video = list(); pointmap = list(); pointmap_mask = list()
        for img_tk, pt_tk in zip(sd_tokens[self.CAM], sd_tokens[self.POINT]):
            video.append(self.read_image_from_token(img_tk))
            pm, pm_msk = self.read_pointmap_from_token(pt_tk, img_tk)
            pointmap.append(pm)
            pointmap_mask.append(pm_msk)
        
        video = torch.stack(video, dim=0,)
        video_t = torch.tensor(timestamps[self.CAM])
        
        pointmap = torch.stack(pointmap, dim=0)
        pointmap_mask = torch.stack(pointmap_mask, dim=0)
        pointmap_t = torch.tensor(timestamps[self.POINT])
        return video, video_t, pointmap, pointmap_mask, pointmap_t
    


if __name__ == "__main__":
    # Example usage
    import cv2 
    from torchvision import transforms
    image_transform = transforms.Compose([
        transforms.Resize((480, 832)),
        transforms.ToTensor(),
    ])
    dataset = TruckScenesDiffusionDataset(data_root='/data/truckscenes', version='v1.0-mini', split='mini_train', chunk_size=40,
                                          image_transform=image_transform)
    # Plot points onto images
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    cmap = plt.get_cmap('viridis')
    count = 0
    for  video, _ , pointmap, pointmap_mask, _  in dataset:
        video_recorder = cv2.VideoWriter(f'video_{count}.avi', cv2.VideoWriter_fourcc(*'XVID'), 15, (832, 480))
        count += 1
        for i in range(video.shape[0]):
            img = video[i].permute(1, 2, 0).numpy()
            img = (img * 255).astype(np.uint8)
            img[:, :,[0, 1, 2]] = img[:, :, [2, 1, 0]] # BGR to RGB
            img = np.ascontiguousarray(img)
            
            pm = pointmap[i].permute(1, 2, 0).numpy()
            pm_msk = pointmap_mask[i].numpy()
            norm_pm = np.linalg.norm(pm, axis=-1, ord=2)
            v_min, v_max = np.min(norm_pm[pm_msk]), np.max(norm_pm[pm_msk])
            normalize = Normalize(vmin=v_min, vmax=v_max)
            
            colors = cmap(normalize(norm_pm))        
            for i in range(pm_msk.shape[0]):
                for j in range(pm_msk.shape[1]):
                    if pm_msk[i, j]:
                        cc = (colors[i, j, :3]*255).astype(np.int32)
                        cv2.circle(img, (j, i), 2, tuple(map(int, cc)), -1)
            
            video_recorder.write(img)
        video_recorder.release()