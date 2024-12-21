# base class
import os
import io

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
    _num_radius_divisions = 5
    _num_angle_divisions = 12
    
    
    def __init__(self, src:os.PathLike, img_bytes:bytes = None, num_radius_divisions=5, num_angle_divisions=12, debug:bool = False):        
        if img_bytes:
            self.src = None
            self.img_origin = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
            if self.img_origin.shape[2] == 4: # 알파 채널이 있는 경우:                
                self.img_origin = cv2.cvtColor(self.img_origin, cv2.COLOR_BGRA2RGBA)
                self.img = self.img_origin[:,:,:3]
                self.mask = self.img_origin[:,:,3] > 0
                self.mask_color = [255, 255, 255]
                self.img[~self.mask] = self.mask_color                
            elif self.img_origin.shape[2] == 3: # 알파 채널이 없는 경우:
                self.img_origin = cv2.cvtColor(self.img_origin, cv2.COLOR_BGR2RGB)
                self.img = self.img_origin.copy()                                
        else:
            self.src = src
        
            # self.img = cv2.cvtColor(cv2.imread(src), cv2.COLOR_BGR2RGB)
            
            # IMREAD_UNCHANGED로 알파 채널 포함해서 읽기 (alpha가 0인 것이 투명한 배경)
            img_with_alpha = cv2.imread(src, cv2.IMREAD_UNCHANGED)
            if img_with_alpha.shape[2] == 3: # 알파 채널이 없는 경우
                self.img = cv2.cvtColor(img_with_alpha, cv2.COLOR_BGR2RGB)
            else:            
                self.img = cv2.cvtColor(img_with_alpha[:,:,:3], cv2.COLOR_BGR2RGB)
                # # 알파 채널로 마스크 생성
                self.mask = img_with_alpha[:,:,3] > 0
                self.mask_color = [255, 255, 255]
                # # 마스크 적용하여 배경 제거
                self.img[~self.mask] = self.mask_color  # 또는 원하는 배경색
        
        self.debug = debug
        self.__class__._num_radius_divisions = num_radius_divisions
        self.__class__._num_angle_divisions = num_angle_divisions
    
    def set_method_debug(self, method_name:str, debug:bool):
        setattr(self, f"_{method_name}_debug", debug)

    @classmethod        
    def set_division(cls, num_radius_divisions:int, num_angle_divisions:int):
        cls._num_radius_divisions = num_radius_divisions
        cls._num_angle_divisions = num_angle_divisions
    
    # TODO: non_max suppression opt. To be used in set_orb method.
    # def non_max_suppression(cls, keypoints, min_distance=5):
    #     if not keypoints:
    #         return []

    #     # Convert keypoints to a numpy array of coordinates and responses
    #     kp_array = np.array([(kp.pt[0], kp.pt[1], kp.response) for kp in keypoints], dtype=np.float32)
        
    #     # Sort keypoints by response (assuming higher is better)
    #     sorted_indices = np.argsort(-kp_array[:, 2])
    #     kp_array = kp_array[sorted_indices]
        
    #     # Create FLANN matcher
    #     FLANN_INDEX_KDTREE = 1
    #     index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    #     search_params = dict(checks=50)
    #     flann = cv2.FlannBasedMatcher(index_params, search_params)
        
    #     # Convert keypoints to the format expected by FLANN matcher
    #     kp_locations = kp_array[:, :2].reshape(-1, 1, 2)
        
    #     selected = []
    #     for i, kp in enumerate(kp_array):
    #         if i == 0:
    #             selected.append(keypoints[sorted_indices[i]])
    #             continue
            
    #         # Find neighbors within min_distance
    #         matches = flann.radiusMatch(kp[:2].reshape(1, 1, 2).astype(np.float32), 
    #                                     kp_locations[:len(selected)], 
    #                                     min_distance)
            
    #         if not matches[0]:  # No neighbors within the radius
    #             selected.append(keypoints[sorted_indices[i]])
        
    #     return selected
            
    @classmethod
    def non_max_suppression(cls, keypoints, min_distance=5, max_keypoints=0):
        keypoints = sorted(keypoints, key=lambda x: x.response, reverse=True) # TODO: response에 따른 필터링도 가능.
        selected = []        
        # TODO: 아래 부분은 sweep line 알고리즘으로 대체 가능.
        for kp in keypoints:
            if max_keypoints and len(selected) >= max_keypoints:
                break                        
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
        pass
    