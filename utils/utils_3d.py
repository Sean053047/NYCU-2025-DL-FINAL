import torch
import numpy as np 
from scipy.spatial.transform import Rotation as R

def get_extrinsic_matrix(rot, trans, scalar_first=True):
    rot_matrix = R.from_quat(rot, scalar_first=scalar_first).as_matrix()
    extrinsic = np.identity(4, dtype=np.float32)
    extrinsic[:3, :3] = rot_matrix
    extrinsic[:3, 3] = trans
    return extrinsic

def pcd2pointmap(pcd:np.array, extrinsic, intrinsic, image_shape):
    '''pointmap: (N, 2); for dim=1, (x, y) '''
    N, _ = pcd.shape
    pcd = np.concatenate((pcd, np.ones((N, 1), dtype=np.float32)), axis=1)
    pcd = np.dot(pcd, extrinsic.T)
    pcd = pcd[:, :3] / pcd[:, 3:]
    pcd = np.dot(pcd, intrinsic.T)
    image_index = np.round(pcd[:, :2] / pcd[:, 2:]).astype(np.int32)
    valid = np.bitwise_and(
        np.bitwise_and(image_index[:, 0] >= 0, image_index[:, 0] < image_shape[1]), 
        np.bitwise_and(image_index[:, 1] >= 0, image_index[:, 1] < image_shape[0])
    )
    
    
    valid_pcd = pcd[valid]
    depth_order = np.argsort(valid_pcd[:, 2]) # * Sort along z axis
    valid_pcd = valid_pcd[depth_order]
    image_index = image_index[valid][depth_order, :] # * (x, y)
    
    uimg_id = image_index[:, 1] * image_shape[1] + image_index[:, 0]
    image_index, idx = np.unique(uimg_id, return_index=True)
    
    pointmap = np.full((image_shape[0]*image_shape[1], 3), -1, dtype=np.int32)
    pointmap[image_index, :] = valid_pcd[idx, :]
    pointmap = pointmap.reshape((image_shape[0], image_shape[1], 3))
    return pointmap


def lidar_motion_compensation(pcd:np.array, ):
    ...
    


if  __name__  == "__main__":
    from truckscenes import TruckScenes
    tsc = TruckScenes(dataroot='/data/truckscenes', split='v1.0-mini', verbose=False)
    
    
    pcd = tsc.get_pcd(0)
    intrinsic = tsc.get_intrinsic(0)