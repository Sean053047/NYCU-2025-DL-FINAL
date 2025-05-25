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
        from tqdm import tqdm
        import multiprocessing as mp
        process = list()
        for cam, pc in self.pair.items():
            p = mp.Process(target=self.dump_video_per_pair, args=(save_dir, cam, pc))
            p.start()
            process.append(p)
        for p in process:
            p.join()
    def dump_video_per_pair(self,save_dir, cam, pc):
        from tqdm import tqdm
        count_cam_vid = 0
        bar = tqdm(total=len(self.scene_tokens), desc=f'Dumping {cam} and {pc} video')
        for scene_tk in self.scene_tokens:
            bar.update(1)
            scene = self.trucksc.get('scene', scene_tk)
            for sw_tokens, timestamps in self.aggregate_pair_sweep(scene, cam):
                count_cam_vid += 1                    
                
                cam_writer = VideoWriter(save_dir=os.path.join(save_dir, 'cam_video'),
                                            fname=f"{cam}_{str(count_cam_vid).zfill(5)}.mp4",
                                            image_shape=self.image_shape)
                pc_writer = VideoWriter(save_dir=os.path.join(save_dir, 'lidar_video',),
                                            fname=f"{cam}_{str(count_cam_vid).zfill(5)}.mp4",
                                            image_shape=self.image_shape)
                for i in range(self.chunk_size):
                    image = self.read_image_from_token(sw_tokens[cam][i])
                    image = np.array(image)[:, :, ::-1] # Convert to BGR format for OpenCV
                    cam_writer.write(image)    
                    if sw_tokens[pc][i] is not None: 
                        pointmap, mask = self.read_pointmap_from_token(sw_tokens[pc][i], sw_tokens[cam][i], keep_intensity=False)
                        image[~mask, : ] = 0
                    else:
                        image = np.zeros((self.image_shape[0], self.image_shape[1], 3), dtype=np.uint8)
                    pc_writer.write(image)
                    
                pc_writer.release()
                cam_writer.release()
    def aggregate_pair_sweep(self, scene, camera):
        '''Get the corresponding camera images and lidar'''
        sw_tokens = defaultdict(list)
        timestamps = defaultdict(list)
        sample = self.trucksc.get('sample', scene['first_sample_token'])
        camera_tk = sample['data'][camera]
        
        count = 0
        while camera_tk != '':
            count += 1
            sw_tokens[camera].append(camera_tk)
            camera_data = self.trucksc.get('sample_data', camera_tk)            
            timestamps[camera].append(camera_data['timestamp'])
            pc = self.pair[camera]
            
            sample = self.trucksc.get('sample', camera_data['sample_token'])
            pc_data = self.trucksc.get('sample_data', sample['data'][pc])
            sw_tokens[pc].append(pc_data['token'])
            timestamps[pc].append(pc_data['timestamp'])
            
            if count > 1 and count % self.chunk_size == 0: # Return sample per N data
                yield sw_tokens, timestamps
                sw_tokens.clear()
                timestamps.clear()
            camera_tk = camera_data['next']
        
        if len(timestamps) > 0:
            print(f'{scene["name"]} abandon redundant ', len(timestamps), 'samples.')
        
    def aggregate_samples(self, scene):
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
        pointsensor = self.trucksc.get('sample_data', pt_token)
        if 'LIDAR' in pointsensor['channel']:
            pc = LidarPointCloud.from_file(self.trucksc.get_sample_data_path(pt_token))
        else:
            pc = RadarPointCloud.from_file(self.trucksc.get_sample_data_path(pt_token))
        
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
        
        mask = np.logical_and(mask, depths > self.pcd_min_depth)
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
        
        return pointmap,pointmap_mask
            
    def render_point(self, pc:PointCloud):
        import open3d as o3d
        pc = pc.points[:3, :].T
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc)
        o3d.visualization.draw_geometries([pcd])        
    
    def render_image(self, img:np.ndarray):
        import cv2
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
    CAMERA_LEFT_FRONT= 'LIDAR_LEFT', 
    CAMERA_LEFT_BACK='LIDAR_LEFT', 
    CAMERA_RIGHT_FRONT='LIDAR_RIGHT', 
    CAMERA_RIGHT_BACK='LIDAR_RIGHT'
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


if __name__ == "__main__":
    '''This code aims to dump lidar condition video and video.'''
    # Example usage
    args = parse_args()
    image_shape = (args.im_height, args.im_width)
    
    from torchvision import transforms
    image_transform = transforms.Compose([
        transforms.Resize(image_shape),
    ])
    dataset = DiffusionGetVideo(data_root='/data/truckscenes', version='v1.0-mini', split='mini_train', chunk_size=81,
                                pair=pair, image_transform=image_transform)
    
    dataset.dump_pair_video(save_dir=args.save_dir)
    
    