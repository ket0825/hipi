import os

import cv2
import numpy as np
from sklearn.decomposition import PCA


class ImgBase:    
    """
    ## Base class for image processing
    
    ### Matched parameters:
    #### In Bounding circle
    
    - _num_radius_divisions
    - _num_angle_divisions
    
    """
    _num_radius_divisions = 4
    _num_angle_divisions = 8
    
    def __init__(self, src:os.PathLike, num_radius_divisions=4, num_angle_divisions=8):
        self.src = src
        self.img = cv2.imread(src)
        self.__class__._num_radius_divisions = num_radius_divisions
        self.__class__._num_angle_divisions = num_angle_divisions

    @classmethod        
    def set_division(cls, num_radius_divisions:int, num_angle_divisions:int):
        cls._num_radius_divisions = num_radius_divisions
        cls._num_angle_divisions = num_angle_divisions
        
    @classmethod
    def non_max_suppression(cls, keypoints, min_distance=5):
        keypoints = sorted(keypoints, key=lambda x: x.response, reverse=True)
        selected = []
        for kp in keypoints:            
            if any(cls.distance(kp, s) < min_distance for s in selected):
                continue
            selected.append(kp)
            
        return selected
    
    @staticmethod
    def distance(kp1, kp2):
        return np.sqrt((kp1.pt[0] - kp2.pt[0])**2 + (kp1.pt[1] - kp2.pt[1])**2)    
    
    @staticmethod
    def calculate_max_distance(pca_points):
    # 각 점에서 다른 모든 점까지의 벡터 계산
        diff = pca_points[:, np.newaxis, :] - pca_points[np.newaxis, :, :]    
        
        # 거리의 제곱 계산
        squared_distances = np.sum(diff**2, axis=-1)    
        
        # 대각선(자기 자신과의 거리)을 제외하고 최대값 찾기
        max_squared_distance = np.max(squared_distances[~np.eye(squared_distances.shape[0], dtype=bool)])
        
        # 제곱근을 취해 실제 거리 계산
        max_distance = np.sqrt(max_squared_distance)
        
        return max_distance

    def set_orb(self, nfeatures:int, nms_distance:int, debug=False):
        pass
    
    def set_pca(self):
        return PCA(n_components=2)
    