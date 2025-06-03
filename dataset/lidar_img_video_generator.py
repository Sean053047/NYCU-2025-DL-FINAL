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
import cv2
from abc import abstractmethod
from scipy.spatial import KDTree
class BaseSaver(object):
    def __init__(self, save_dir:str):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
    def __repr__(self):
        return f'{self.__class__.__name__}(save_dir={self.save_dir})'
    @ abstractmethod
    def write(self, *args, **kwargs):
        ...
        
class PointmapWriter(BaseSaver):
    def __init__(self, save_dir:str):
        super().__init__(save_dir)
    def write(self, pc:np.ndarray, fname:str):
        '''Save point cloud to pcd file'''
        file_path = os.path.join(self.save_dir, fname)
        np.save(file_path, pc)

class VideoWriter(BaseSaver):
    def __init__(self, save_dir:str, fname, image_shape:Tuple[int, int]=(480, 832)):
        super().__init__(save_dir)
        self.image_shape = image_shape
        assert fname.split('.')[-1] in ['avi', 'mp4'], 'File name must be .avi or .mp4'
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.fps = 20.0
        self.video_writer = cv2.VideoWriter(os.path.join(self.save_dir, fname), self.fourcc, self.fps, (self.image_shape[1], self.image_shape[0]))    
    def write(self, image:np.ndarray):
        self.video_writer.write(image)
    def release(self):
        self.video_writer.release()
    
class SensorData(object):
    def __init__(self, token, sample_token, ego_pose_token, 
                    calibrated_sensor_token, timestamp, fileformat, 
                    is_key_frame, height, width, filename, prev, next, sensor_modality, channel, pc=None):
        self.token = token
        self.sample_token = sample_token
        self.ego_pose_token = ego_pose_token
        self.calibrated_sensor_token = calibrated_sensor_token
        self.timestamp = timestamp
        self.fileformat = fileformat
        self.is_key_frame = is_key_frame
        self.height = height
        self.width = width
        self.filename = filename
        self.prev = prev
        self.next = next
        self.sensor_modality = sensor_modality
        self.channel = channel
        self.pc = pc
    @classmethod
    def deserialize(cls, item:Dict):
        return cls(**item)
        
    def __repr__(self):
        return f"<SD:{self.channel}, tk:{self.token}, t:{self.timestamp}>"

class DiffusionGetVideo(Dataset):
    def __init__(self, 
                 data_root:str, 
                 version:str,  
                 split:str, 
                 chunk_size:int, 
                 pair:Dict[str, List[str]],
                 image_shape:Tuple[int, int]=(480, 832),
                 meta=None, image_transform=None, point_transform=None):
        '''split: {"train", "val", "test", "mini_train", "mini_val"}'''
        # Available sensors
        if meta is None:
            meta = sensor_meta
        self.meta = meta
        self.pair = pair
        self.trucksc = TruckScenes(version, data_root, verbose=False)
        self.chunk_size = chunk_size
        self.image_shape = np.array(image_shape) # (row, col)
        
        self.split = globals()[split]
        self.scene_tokens  = [self.trucksc.field2token('scene', 'name', scene_name)[0] for scene_name in self.split]
        self.sd_chunks = []
        
        self.image_transform = image_transform
        self.point_transform = point_transform
        self.pcd_min_depth = 0.5
    
    def dump_pair_video(self, save_dir:str, ):
        import multiprocessing as mp
        process = list()
        for cam, pc_list in self.pair.items():
            self.dump_video_per_pair(save_dir, cam, pc_list)
        #     p = mp.Process(target=self.dump_video_per_pair, args=(save_dir, cam, pc))
        #     p.start()
        #     process.append(p)
        # for p in process:
        #     p.join()
    
    def dump_video_per_pair(self,save_dir, cam, pc_list:List):
        from tqdm import tqdm
        count_cam_vid = 0
        bar = tqdm(total=len(self.scene_tokens), desc=f'Dumping {cam} and {pc_list} video')
        for scene_tk in self.scene_tokens:
            bar.update(1)
            for sds_sweeps in self.aggregate_pair_sweep(scene_tk, cam):
                count_cam_vid += 1                    
                
                cam_writer = VideoWriter(save_dir=os.path.join(save_dir, 'cam_video'),
                                            fname=f"{cam}_{str(count_cam_vid).zfill(5)}.mp4",
                                            image_shape=self.image_shape)
                pc_writer = VideoWriter(save_dir=os.path.join(save_dir, 'lidar_video',),
                                            fname=f"{cam}_{str(count_cam_vid).zfill(5)}.mp4",
                                            image_shape=self.image_shape)
                for i in range(self.chunk_size):                 
                    image = self.read_image_from_token(sds_sweeps[i][cam].token)
                    image = image[:, :, ::-1] # Convert to BGR format for OpenCV
                    cam_writer.write(image.astype(np.uint8))    
                    
                    final_mask = np.zeros((self.image_shape[0], self.image_shape[1]), dtype=bool)
                    for range_sensor in self.pair[cam]:
                        if sds_sweeps[i][range_sensor] is not None: 
                            pointmap, mask = self.get_pointmap(sds_sweeps[i][range_sensor].token, sds_sweeps[i][cam].token, keep_intensity=False)
                            final_mask = np.bitwise_or(final_mask, mask)
                    image[~final_mask, :] = 0 # Set the background to black
                    pc_writer.write(image.astype(np.uint8))
                pc_writer.release()
                cam_writer.release()
    
    def dump_scene_inference(self, save_dir:str,):
        ...
    
    def dump_inference_per_scene(self, save_dir, scene_tk, cam, extra_transforms:List[Dict]=None):
        '''
        Based on certain camera, move the orientation and position.
        '''
        from tqdm import tqdm
        import shutil 
        
        import json
        sds_list = {sensor: self.get_sensor_sweep(scene_tk, sensor) 
                        for sensor, v in self.meta.items() if v}
        base_t = np.array([sd.timestamp for sd in sds_list[cam]], dtype=np.int64).reshape(-1, 1)
        
        sds_storage  = [dict() for _ in range(len(base_t))]
        # * Get the time-closet pair from multiple sensor to 1 certain sensor.
        # * Save it to time-based storage.
        for sensor, sd_list in sds_list.items():
            cmp_t = np.array([sd.timestamp for sd in sd_list], dtype=np.int64).reshape(-1, 1)
            distance, index = KDTree(cmp_t).query(base_t, k=1, workers=-1)                        
            for i, idx in enumerate(index.flatten()):
                sds_storage[i][sensor] = sd_list[idx]
        # * Query per frame point cloud.
        for sds in tqdm(sds_storage, total=len(sds_storage), desc=f'Query PointCloud color: '):
            self.query_pc_color(sds)
        
        # * Dump whole scene data
        scene_name = self.trucksc.get('scene', scene_tk)['name']
        ego_poses = []  # Ego poses for the scene, will be dumped to json file.
        for sds in tqdm(sds_storage, total=len(sds_storage), desc=f'Dumping scene {scene_name}'):
            for sensor, sd in sds.items():                
                sensor_save_dir = os.path.join(save_dir, scene_name, sensor)
                os.makedirs(sensor_save_dir, exist_ok=True)
                fstem = f"{sd.timestamp}" # * Add 3 zeros to make it compatible with the original timestamp format.
                if 'LIDAR' in sensor or 'RADAR' in sensor:
                    filename = fstem + '.pcd'
                    self.to_pcd(sd.pc, os.path.join(sensor_save_dir, filename))
                else:
                    filename = fstem + '.jpg'
                    shutil.copyfile(self.trucksc.get_sample_data_path(sd.token), os.path.join(sensor_save_dir, filename))
                ego_pose = self.trucksc.get('ego_pose', sd.ego_pose_token)
                ego_pose['timestamp'] = f"{ego_pose['timestamp']}" # Add 3 zeros to make it compatible with the original timestamp format.
                ego_poses.append({k:v for k,v in ego_pose.items() if k!='token'}) # Get ego pose
        ego_poses = sorted(ego_poses, key=lambda x: x['timestamp']) # Sort ego poses by timestamp
        self.to_json(ego_poses, os.path.join(save_dir, scene_name, 'tf', 'ego_poses.json'))
        
        # * Dump projected pcd condition (May affect the original values of pc)
        if extra_transforms is not None:
            for sds in tqdm(sds_storage, total=len(sds_storage), desc=f'Dumping condition images'):
                for i, extra_T in enumerate(extra_transforms):
                    sensor_save_dir = os.path.join(save_dir, scene_name, f'{cam}_T{i}')
                    os.makedirs(sensor_save_dir, exist_ok=True)
                    filename = f"{sds[cam].timestamp}.jpg"
                    cond_img = self.get_condition_image(sds, cam, extra_T)
                    Image.fromarray((cond_img * 255).astype(np.uint8)).save(os.path.join(sensor_save_dir, filename))

        # * Dump TF, TF here save the transforms between different sensors without considering the ego transforms.     
        __tmp_sds = sds_storage[0] # Use the first sample data to get the sensor list
        for sensor in sds_list.keys():
            # * Dump intrinsic matrix
            if 'CAMERA' in sensor:
                intrinsic = self.get_intrinsic(__tmp_sds[cam].token).flatten().tolist() # Check if intrinsic matrix is correct
                intrinsic_save_dir = os.path.join(save_dir, scene_name, 'intrinsic')
                os.makedirs(intrinsic_save_dir, exist_ok=True)
                self.to_json(intrinsic, os.path.join(intrinsic_save_dir, f'{sensor}.json'),)
                if sensor == cam:
                    for i in range(len(extra_transforms)):
                        self.to_json(intrinsic, os.path.join(intrinsic_save_dir, f'{cam}_T{i}.json'),)
            
            # * Sensor to base camera
            if sensor != cam: # Sensor != base camera    
                tf_save_dir = os.path.join(save_dir, scene_name, 'tf', f'to_{cam}')
                os.makedirs(tf_save_dir, exist_ok=True)
                T = self.get_transform(__tmp_sds[sensor].token, __tmp_sds[cam].token, without_ego=True).flatten().tolist()
                self.to_json(T, os.path.join(tf_save_dir, f'{sensor}.json'),)
            
            # * Sensor to ego
            tf_save_dir = os.path.join(save_dir, scene_name, 'tf', f'to_ego')
            T = self.get_transform(__tmp_sds[sensor].token, to_ego=True).flatten().tolist()
            self.to_json(T, os.path.join(tf_save_dir, f'{sensor}.json'),)
            
            # * Sensor to Extra_TF
            for i, extra_T in enumerate(extra_transforms):
                tf_save_dir = os.path.join(save_dir, scene_name, 'tf', f'to_{cam}_T{i}')
                T = self.get_transform(__tmp_sds[sensor].token, __tmp_sds[cam].token, extra_T=extra_T, without_ego=True).flatten().tolist()
                self.to_json(T, os.path.join(tf_save_dir, f'{sensor}.json'),)
        # * extra_T to ego 
        tf_save_dir = os.path.join(save_dir, scene_name, 'tf', f'to_ego')
        for i, extra_T in enumerate(extra_transforms):
            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = Quaternion(extra_T['rotation']).rotation_matrix
            T[:3, 3] = np.array(extra_T['translation'])
            T = T.flatten().tolist()
            self.to_json(T, os.path.join(tf_save_dir, f'{cam}_T{i}.json'))
            
        
    def get_sensor_sweep(self, scene_tk:str, sensor:str):
        '''Get the timestamps of the sensor data'''
        sweeps = []
        scene = self.trucksc.get('scene', scene_tk)
        sweep_tk = scene['first_sample_token']
        sample = self.trucksc.get('sample', sweep_tk)
        sd_tk = sample['data'][sensor]
        while sd_tk != '':
            sensor_data = self.trucksc.get('sample_data', sd_tk)
            sweeps.append(SensorData.deserialize(sensor_data))
            sd_tk= sensor_data['next']
        return sweeps
        
    def aggregate_pair_sweep(self, scene_tk, cam):
        '''Get the corresponding camera images and lidar'''
        sds_list = {sensor: self.get_sensor_sweep(scene_tk, sensor) for sensor in self.pair[cam] }
        sds_list[cam] = self.get_sensor_sweep(scene_tk, cam)
        base_t = np.array([sd.timestamp for sd in sds_list[cam]], dtype=np.int64).reshape(-1, 1)
        sds_storage  = [dict() for _ in range(len(base_t))]
        # * Find the closet sensor sweep for each camera image
        # * Save it to time-based storage.
        for sensor, sd_list in sds_list.items():
            cmp_t = np.array([sd.timestamp for sd in sd_list], dtype=np.int64).reshape(-1, 1)
            distance, index = KDTree(cmp_t).query(base_t, k=1, workers=-1)                        
            for i, idx in enumerate(index.flatten()):
                sds_storage[i][sensor] = sd_list[idx]
        
        sds_chunk = list()
        for i, sds in enumerate(sds_storage):
            sds_chunk.append(sds)
            if (i+1) % self.chunk_size == 0: # Return sample per N data
                yield sds_chunk
                sds_chunk.clear()
                

    def query_pc_color(self, sds:Dict[str, SensorData])->List[PointCloud]:
        '''Assumption: All images captured at the same time.
        Initiate: colors attribute to PointCloud
        '''
        cameras = [sensor for sensor in sds if 'CAMERA' in sensor]
        range_sensors = [sensor for sensor in sds if 'LIDAR' in sensor or 'RADAR' in sensor]
        for range_sensor in range_sensors:
            rs_sd = sds[range_sensor]
            # * Load pc
            data_path = self.trucksc.get_sample_data_path(rs_sd.token)
            rs_sd.pc = LidarPointCloud.from_file(data_path) if 'LIDAR' in range_sensor else RadarPointCloud.from_file(data_path)
            rs_sd.pc.colors = np.zeros((4, rs_sd.pc.points.shape[1]), dtype=np.float32) # (x, y, z, transparency)
            
            # * Project to each camera and colorize
            for cam in cameras:
                cam_sd = sds[cam]
                # Todo
                # ? Project back to image domain and get corresponding pc indices
                pointmap, mask, pm2pcidx = self.get_pointmap(rs_sd.token, cam_sd.token, keep_intensity=False, return_pc_idx=True)
                pc_idx = pm2pcidx[mask] # * Already check idx is correct.
                # ? Colorize
                image = self.read_image_from_token(cam_sd.token) / 255.0
                color = np.vstack((image[mask,:].T,  np.ones((1, len(pc_idx)), dtype=np.float32))) # (4, N)  r,g,b,a
                rs_sd.pc.colors[:4, pc_idx] = color # ? directly assign. The processing order of camera may influence the color of points.
            # self.render_point(rs_sd.pc) # Debug: render point cloud
    
    def get_transform(  self, 
                        src_tk:str, 
                        dst_tk:str=None, 
                        extra_T:Dict[str, Union[np.ndarray, Quaternion]]=None, 
                        without_ego:bool=False, to_ego:bool=False)->np.ndarray:
        '''Have check the precision error < 1e-10
        extra_T: {'rotation': Quaternion, 'translation': np.ndarray}
        '''
        def _get_T(t_record, inv=False):
            '''Get the transformation matrix from sample_data token\n
            Prevent from using np.linalg.inv() to avoid precision error
            ''' 
            T = np.eye(4, dtype=np.float64)
            if inv:
                inv_rot = T.copy()
                T[:3, 3] = -np.array(t_record['translation'])
                inv_rot[:3, :3] = Quaternion(t_record['rotation']).rotation_matrix.T
                T = inv_rot.dot(T)
            else:    
                T[:3, :3] = Quaternion(t_record['rotation']).rotation_matrix
                T[:3, 3] = np.array(t_record['translation'])
            return T
        src_sd = self.trucksc.get('sample_data', src_tk)
        transforms = _get_T(self.trucksc.get('calibrated_sensor', src_sd['calibrated_sensor_token']))
        if to_ego: return transforms        
        
        dst_sd = self.trucksc.get('sample_data', dst_tk)
        if not without_ego:
            transforms = _get_T(self.trucksc.get('ego_pose', src_sd['ego_pose_token'])).dot(transforms)
            transforms = _get_T(self.trucksc.get('ego_pose', dst_sd['ego_pose_token']), inv=True).dot(transforms)
        transforms = _get_T(
            (self.trucksc.get('calibrated_sensor', dst_sd['calibrated_sensor_token']) if extra_T is None else extra_T), 
            inv=True
            ).dot(transforms)
        
        return transforms        
    
    def get_intrinsic(self, cam_tk:str)->np.ndarray:
        cam = self.trucksc.get('sample_data', cam_tk)
        cs_record = self.trucksc.get('calibrated_sensor', cam['calibrated_sensor_token'])
        intrinsic = np.array(cs_record['camera_intrinsic'], dtype=np.float64)
        org_im_shape = np.array([cam['height'], cam['width']])
        ratio_r, ratio_c = self.image_shape / org_im_shape
        # modified intrinsic matrix to new image shape, and project
        intrinsic[0, :] *= ratio_c
        intrinsic[1, :] *= ratio_r
        return intrinsic
    
    def get_pointmap(self, pc_tk:str, cam_tk:str, keep_intensity=False, return_pc_idx=False, extra_T:Dict=None):
        pointsensor = self.trucksc.get('sample_data', pc_tk)
        if 'LIDAR' in pointsensor['channel']:
            pc = LidarPointCloud.from_file(self.trucksc.get_sample_data_path(pc_tk))
        else:
            pc = RadarPointCloud.from_file(self.trucksc.get_sample_data_path(pc_tk))
        
        # Points live in the point sensor frame. So they need to be transformed
        # via global to the image plane.
        pc.transform(self.get_transform(pc_tk, cam_tk, extra_T=extra_T, without_ego=False)) # Transform point cloud to camera frame
        intrinsic = self.get_intrinsic(cam_tk)
        
        depths = pc.points[2, :]
        pc_im_idx = view_points(pc.points[:3, :], intrinsic, normalize=True) # (col, row, 1)
        mask = np.ones(depths.shape[0], dtype=bool)
        
        mask = np.logical_and(mask, depths > self.pcd_min_depth)
        mask = np.logical_and(mask, pc_im_idx[0, :] > 1)
        mask = np.logical_and(mask, pc_im_idx[0, :] < self.image_shape[1] - 1) # dim: col
        mask = np.logical_and(mask, pc_im_idx[1, :] > 1)
        mask = np.logical_and(mask, pc_im_idx[1, :] < self.image_shape[0] - 1) # dim: row
        
        pc_im_idx = pc_im_idx[[1, 0, 2], :][:2, mask] # change to (row, col)
        pc_im_idx = np.round(pc_im_idx).astype(np.int32)
        pc_idx = np.arange(pc.points.shape[1], dtype=np.int32)
        pc_idx = pc_idx[mask]
        pc = pc.points[:, mask]
        
        # * Deal with overlapping points at same pixel.
        # * Keep the point with high intensity
        arg = np.argsort(pc[3, :])[::-1] # sort by intensity
        pc = pc[:, arg]
        pc_idx = pc_idx[arg]
        pc_im_idx = pc_im_idx[:, arg]
        
        # * Assign pixel with corresponding point location
        pointmap_mask = np.full((self.image_shape[0], self.image_shape[1]), False)
        pointmap = np.full((4, self.image_shape[0], self.image_shape[1]), 0.0)
        pm2pcidx = np.full((self.image_shape[0], self.image_shape[1]), -1, dtype=np.int32)
        unique_pc_im_idx, arr_idx = np.unique(pc_im_idx, axis=1, return_index=True)
        unique_pc = pc[:, arr_idx]
        unique_pc_idx = pc_idx[arr_idx]
        row, col = unique_pc_im_idx
        
        pointmap_mask[row, col] = True
        pointmap[:, row, col] = unique_pc[:4, :]
        pm2pcidx[row, col] = unique_pc_idx      
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
        if return_pc_idx:
            return pointmap, pointmap_mask, pm2pcidx
        else:
            return pointmap, pointmap_mask
    
    def get_condition_image(self, sds, cam, extra_T:Dict[str, Union[np.ndarray, Quaternion]])-> np.ndarray:
        # * 1. Merge different lidar piont into same coordinate
        # * 2. Use those colorized pc to project to camera and get images.
        # * p.s. Currently only use lidar points.
        agg_points = list()
        agg_colors = list()
        import copy
        for sensor, sd in sds.items():
            if "LIDAR" not in sensor: continue
            pc = copy.deepcopy(sd.pc)
            # * Remove transparent points
            valid_mask = pc.colors[3, :] > 0.0
            pc.points = pc.points[:, valid_mask]
            pc.colors = pc.colors[:, valid_mask]
            pc.transform(self.get_transform(sd.token, sds[cam].token, extra_T=extra_T))
            agg_points.append(pc.points)
            agg_colors.append(pc.colors)    
        pc_arr = np.concat(agg_points, axis=1)
        pc_colors = np.concat(agg_colors, axis=1)
        _pc = LidarPointCloud(pc_arr, ) # Create a point cloud object
        _pc.colors = pc_colors
        intrinsic = self.get_intrinsic(sds[cam].token)
        depths = pc_arr[2, :]
        pc_im_idx = view_points(pc_arr[:3, :], intrinsic, normalize=True) # (col, row, 1)
        mask = np.ones(depths.shape[0], dtype=bool)
        mask = np.logical_and(mask, depths > self.pcd_min_depth)
        mask = np.logical_and(mask, pc_im_idx[0, :] > 1)
        mask = np.logical_and(mask, pc_im_idx[0, :] < self.image_shape[1] - 1) # dim: col
        mask = np.logical_and(mask, pc_im_idx[1, :] > 1)
        mask = np.logical_and(mask, pc_im_idx[1, :] < self.image_shape[0] - 1) # dim: row
        
        pc_im_idx = pc_im_idx[[1, 0], :][:, mask] # change to (row, col)
        pc_im_idx = np.round(pc_im_idx).astype(np.int32)
        pc_colors = pc_colors[:3, mask]
        intensity = pc_arr[3, mask]
        
        arg = np.argsort(intensity)[::-1] # sort by intensity
        pc_colors = pc_colors[:, arg]
        pc_im_idx = pc_im_idx[:, arg]
        unique_pc_im_idx, arr_idx = np.unique(pc_im_idx, axis=1, return_index=True)
        unique_color = pc_colors[:, arr_idx]
        row, col = unique_pc_im_idx
        condition = np.full(( self.image_shape[0], self.image_shape[1], 3), 0.0)
        condition[row, col, :] = unique_color.T # (row, col, 3)
        return condition.astype(np.float32)

    def read_image_from_token(self, token:str)->np.ndarray:
        '''Read image from token. RGB'''
        # sample_data = self.trucksc.get('sample_data', token)
        file_path = self.trucksc.get_sample_data_path(token)
        image = Image.open(file_path)
        image = image.convert('RGB')
        if self.image_transform is not None:
            image = self.image_transform(image)
        image = np.array(image, dtype=np.float32)
        assert np.all(image.shape[:2] == self.image_shape), f'Image shape {image.shape} does not match expected shape {self.image_shape}'
        return image
    
    def render_point(self, pc:PointCloud):
        import open3d as o3d
        
        pc_arr = pc.points[:3, :].T
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc_arr)
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        if hasattr(pc, 'colors'):
            colors = pc.colors.T.astype(np.float32)
            mask = colors[:, 3] > 0.0 # Remove transparent points
            colors = colors[mask, :3] # Keep only RGB channels
            pc_arr = pc_arr[mask, :]
            pcd.points = o3d.utility.Vector3dVector(pc_arr)
            pcd.colors = o3d.utility.Vector3dVector(colors[:,:3])
        o3d.visualization.draw_geometries([pcd, frame])        
    
    def render_image(self, img:np.ndarray):
        import cv2
        cv2.imshow('tmp', img)
        cv2.waitKey(0)
    
    @staticmethod
    def to_json(data, fpth:str):
        import json
        os.makedirs(os.path.dirname(fpth), exist_ok=True)
        with open(fpth, 'w') as f: 
            json.dump(data, f, indent=4)
    
    @staticmethod
    def to_pcd( pc, fpth:str, rm_transparent_pc:bool=False):
        import pypcd4
        fields = ('x', 'y', 'z', 'r', 'g', 'b', 'a')
        types = (np.float32, np.float32, np.float32, np.float32, np.float32, np.float32, np.float32)
        arr = np.concat((pc.points[:3, :], pc.colors), axis=0).T
        if rm_transparent_pc:
            mask = arr[:, 6] > 0.0 
            arr = arr[mask, :]
        pcd = pypcd4.PointCloud.from_points(arr, fields=fields, types=types)
        pcd.save(fpth)
        
    def __official_pc_transform(self, pc:PointCloud, src_tk:str, dst_tk:str):
        src_sd = self.trucksc.get('sample_data', src_tk)
        dst_sd = self.trucksc.get('sample_data', dst_tk)
        # First step: transform the pointcloud to the ego vehicle frame
        # for the timestamp of the sweep.
        cs_record = self.trucksc.get('calibrated_sensor', src_sd['calibrated_sensor_token'])
        pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix)
        pc.translate(np.array(cs_record['translation']))
        # Second step: transform from ego to the global frame.
        poserecord = self.trucksc.get('ego_pose', src_sd['ego_pose_token'])
        pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix)
        pc.translate(np.array(poserecord['translation']))
        # org_pc.transform(_get_T(poserecord)) # Transform original point cloud to camera frame for debug
        # Third step: transform from global into the ego vehicle frame
        # for the timestamp of the image.
        poserecord = self.trucksc.get('ego_pose', dst_sd['ego_pose_token'])
        pc.translate(-np.array(poserecord['translation']))
        pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix.T)
        # org_pc.transform(_get_T(poserecord, inv=True)) # Transform original point cloud to camera frame for debug
        # Fourth step: transform from ego into the camera.
        cs_record = self.trucksc.get('calibrated_sensor', dst_sd['calibrated_sensor_token'])
        pc.translate(-np.array(cs_record['translation']))
        pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix.T)
    
    
    
sensor_meta = dict(
    RADAR_RIGHT_BACK=False, 
    RADAR_RIGHT_SIDE=False, 
    RADAR_RIGHT_FRONT=False, 
    RADAR_LEFT_FRONT=False,
    RADAR_LEFT_SIDE=False, 
    RADAR_LEFT_BACK=False, 
    LIDAR_LEFT= True,
    LIDAR_RIGHT=True, 
    LIDAR_TOP_FRONT=False, 
    LIDAR_TOP_LEFT=False, 
    LIDAR_TOP_RIGHT=False, 
    LIDAR_REAR=True, 
    CAMERA_LEFT_FRONT=True, 
    CAMERA_LEFT_BACK=True, 
    CAMERA_RIGHT_FRONT=True, 
    CAMERA_RIGHT_BACK=True
)
pair = dict(
    CAMERA_LEFT_FRONT= ['LIDAR_LEFT', 'LIDAR_RIGHT', 'LIDAR_REAR'],
    CAMERA_LEFT_BACK=['LIDAR_LEFT', 'LIDAR_RIGHT', 'LIDAR_REAR'], 
    CAMERA_RIGHT_FRONT=['LIDAR_LEFT', 'LIDAR_RIGHT', 'LIDAR_REAR'],
    CAMERA_RIGHT_BACK=['LIDAR_LEFT', 'LIDAR_RIGHT', 'LIDAR_REAR']
)

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Dump lidar condition video and video.')
    parser.add_argument('--data-root', type=str, default='/data/truckscenes', help='data root path')
    parser.add_argument('--version', type=str, default='v1.0-mini', help='dataset version')
    parser.add_argument('--save-pcvid', action='store_true', help='save point cloud video')
    parser.add_argument('--split', type=str, default='mini_train', help='dataset split')
    parser.add_argument('--chunk_size', type=int, default=81, help='chunk size')
    parser.add_argument('--im_height', type=int, default=480, help='image height')
    parser.add_argument('--im_width', type=int, default=832, help='image width')
    parser.add_argument('--save-dir', type=str, default='results', help='save directory')
    args = parser.parse_args()
    return args

def get_transforms(rotations:List[Quaternion], translations:List[np.ndarray])->List[np.ndarray]:
    transforms = list()
    for rot, trans in zip(rotations, translations):
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = rot.rotation_matrix
        T[:3, 3] = trans
        transforms.append(T)
    return transforms

if __name__ == "__main__":
    '''This code aims to dump lidar condition video and video.'''
    # Example usage
    args = parse_args()
    image_shape = (args.im_height, args.im_width)
    
    from torchvision import transforms
    image_transform = transforms.Compose([
        transforms.Resize(image_shape),
    ])
    dataset = DiffusionGetVideo(data_root=args.data_root, version='v1.0-mini', split='mini_val', chunk_size=81,
                                pair=pair, image_transform=image_transform)
    
    dataset.dump_pair_video(save_dir=args.save_dir)
    exit()
    '''
    example:
    
    CAMERA_LEFT_FRONT
        translation:[5.218337, 1.289023, 2.110627]
        quaternion:[0.5243729158425541, -0.5347430148556852, 0.46913802442248353, -0.46796630993566085]
    CAMERA_LEFT_BACK 
        translation:[5.049643, 1.368585, 2.115646]
        quaternion:[0.6815585306659455, -0.67419642629349, -0.2069280647012522, 0.19523812150437736]
    CAMERA_RIGHT_FRONT
        translation:[5.226811, -1.310934, 2.092867]
        quaternion:[0.46356873659542747, -0.46682641092971633, 0.5298264899369438, -0.5352205331177945]
    CAMERA_RIGHT_BACK
        translation:[5.048685, -1.39767, 2.090194], 
        quaternion:[0.20680413770088169, -0.19142068501963191, -0.6846868589399366, 0.6721562877570766]
    '''
    
    rotations = [Quaternion([0.46356873659542747, -0.46682641092971633, 0.5298264899369438, -0.5352205331177945])] * 2 + \
                [Quaternion([0.5243729158425541, -0.5347430148556852, 0.46913802442248353, -0.46796630993566085])] * 2
    base = np.array([5.218337, 1.289023, 2.110627]) # Base on Camera left front
    translations = [np.array([1, 5 ,0]), np.array([0.5, 5 ,0]), np.array([-0.5, 5 ,0]), np.array([-1, 5 ,0])]
    translations = [ tr + base for tr in translations]
    extra_transforms = [ {'rotation':rot, 'translation': trans}
                        for rot, trans in zip(rotations, translations)]
    dataset.dump_inference_per_scene(save_dir=args.save_dir, 
                                     scene_tk=dataset.scene_tokens[1],
                                     cam='CAMERA_LEFT_FRONT',
                                     extra_transforms=extra_transforms)
    