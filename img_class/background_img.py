import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from img_class.base import ImgBase
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

from utils.common_utils import set_timer

class BackgroundImg(ImgBase):    
    """
    SHOULD KEEP:
    
    self.pca_class_list (to inverse_transform): # shape: M
    
    self.pca_back_kps (pca로 변환된 keypoints): M x N x 2
    
    self.kp1 (원본)
    
    self.img (원본)
    
    self.radii (scale 조절에 사용. array) # shape: M
    
    self.is_rotated (Object의 회전 여부. array) # shape: M
    """
    
    @property
    def __name__(self):
        return "BackgroundImg" if not hasattr(self, "src") else f"BackgroundImg - {self.src} "
    
    def __init__(self, src:os.PathLike):
        super().__init__(src, num_radius_divisions=4, num_angle_divisions=8)
        
    @set_timer()
    def set_orb(self, nfeatures:int, nms_distance=3):
        """        
        충분히 큰 nfeatures 사용 필요.        
        """
                        
        # blur된 이미지를 gray로 변환 (바로 gray로 변환하지 않음)
        blurred = cv2.GaussianBlur(self.img, (5, 5), 0)  
        gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)     
        
        # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        orb = cv2.ORB()
        orb = orb.create(
            nfeatures=nfeatures, # 상위 몇개의 특징점을 사용할 것인지
            scaleFactor=1.2, 
            nlevels=8,
            edgeThreshold=15, # edgeThreshold와 patchSize는 서로 비례해야 함. edgeThreshold는 제외할 이미지 경계에 사용되는 값
            # Backgroud image는 이 값을 줄여야 함.
            firstLevel=0,
            WTA_K=2, # BRIEF descriptor가 사용할 bit 가짓수. binary이므로 1bit임.
            scoreType=cv2.ORB_HARRIS_SCORE, 
            patchSize=15, # edgeThreshold와 크거나 같은 값으로 설정해야 함.
            fastThreshold=10, # FAST detector에서 근처 픽셀들이 얼마나 밝거나 어두워야 하는지에 대한 임계값
        )

        self.kp1, _ = orb.detectAndCompute(gray, None)                
        self.kp1 = ImgBase.non_max_suppression(self.kp1, nms_distance)
        if getattr(self, f"_set_orb_debug"):
            result = self.img.copy()
            for kp in self.kp1:
                x, y = kp.pt
                cv2.circle(result, (int(x), int(y)), 1, (255, 0, 0), -1)
                
            plt.imshow(result)
            plt.title('Background image orb')
            plt.show()
            
        print(f"Background keypoints: {len(self.kp1)}")
        
        self._M = len(self.kp1)        
        # return self.img, self.kp1

    def get_kp_length(self):
        return self._M
    
    def get_nn_indices(self):
        return self.nn_indices
    
    def get_nn_distances(self):
        return self.nn_distances
    
    @set_timer()
    def set_pca(self, N, T):
        """
        Unlike object image, background image has neighbors' multiple keypoints.
        """
        back_kp_coords = np.array([kp.pt for kp in self.kp1])
        neighbors = NearestNeighbors(n_neighbors=N, algorithm='ball_tree').fit(back_kp_coords)
        self.pca_class_list = [PCA(n_components=2) for _ in range(self._M)] 
        self.is_rotated = np.zeros(self._M, dtype=bool)
        # 중앙점을 기준으로 가장 가까운 N개의 점들의 index, 거리.
        _, self.N_nn_indices = neighbors.kneighbors(back_kp_coords) # shape: M x N 
        
        # 특정 index에서의 가장 가까운 점들 T에 대한 정보도 필요함...
        t_nn = NearestNeighbors(n_neighbors=T+1, algorithm='kd_tree').fit(back_kp_coords)
        _, self.t_nn_indices = t_nn.kneighbors(back_kp_coords)        
        self.t_nn_indices = self.t_nn_indices[:, 1:] # 자기 자신은 제외.
                        
        self.pca_back_kps = np.array([self.pca_class_list[i].fit_transform(back_kp_coords[idx, :]) 
                             for i, idx in enumerate(self.N_nn_indices)]) # shape: M x N x 2
                                           
        self.radii = np.max(np.linalg.norm(self.pca_back_kps, axis=2), axis=1) # shape: M                    
        
        print(f"Background PCAs shape: {self.pca_back_kps.shape}")        
        
    @set_timer()
    def set_pca2(self, N, T):
        """
        set_pca와 다르게 중심점 기준으로 가장 가까운 N개의 점들에 대한 PCA를 구함.
        """
        back_kp_coords = np.array([kp.pt for kp in self.kp1])
        neighbors = NearestNeighbors(n_neighbors=N, algorithm='ball_tree').fit(back_kp_coords)
        self.pca_class_list = [PCA(n_components=2) for _ in range(self._M)]                 
        
        self.is_rotated = np.zeros(self._M, dtype=bool)
        # 중앙점을 기준으로 가장 가까운 N개의 점들의 index, 거리.
        _, self.N_nn_indices = neighbors.kneighbors(back_kp_coords) # shape: M x N 
        
        self.pca_back_kps = np.zeros((self._M, self.N, 2))
        for i in range(self._M):
            neighbor_indices = self.N_nn_indices[i][:1] # 자기 자신은 제외.
            neighbor_coords = back_kp_coords[neighbor_indices]
            
            center_point = back_kp_coords[i]
            centered_coords = neighbor_coords - center_point
            
            self.pca_class_list[i].fit(centered_coords)
            self.pca_back_kps[i] = self.pca_class_list[i].transform(centered_coords)
            
        
        # self.pca_back_kps = np.array([self.pca_class_list[i].fit_transform(back_kp_coords[idx, :]) 
        #                      for i, idx in enumerate(self.N_nn_indices)]) # shape: M x N x 2
                                           
        self.radii = np.max(np.linalg.norm(self.pca_back_kps, axis=2), axis=1) # shape: M   
        
        # 특정 index에서의 가장 가까운 점들 T에 대한 정보도 필요함...
        t_nn = NearestNeighbors(n_neighbors=T+1, algorithm='kd_tree').fit(back_kp_coords)
        _, self.t_nn_indices = t_nn.kneighbors(back_kp_coords)        
        self.t_nn_indices = self.t_nn_indices[:, 1:] # 자기 자신은 제외.                                                 
        
        print(f"Background PCAs shape: {self.pca_back_kps.shape}")        
        
    @set_timer()
    def set_histograms(self):        
        """        
        """
        # 극좌표계로 변환
        r = np.sqrt(np.sum(self.pca_back_kps**2, axis=2))  # 반경 계산
        theta = np.arctan2(self.pca_back_kps[:,:,1], self.pca_back_kps[:,:,0])  # 각도 계산 (-pi to pi)
        
        # 각도를 0에서 2pi 범위로 조정
        theta = (theta + 2*np.pi) % (2*np.pi)
        
        # 각 차원의 bin 인덱스 계산
        r_max = np.max(r)
                
        r_bins = np.minimum((r / (r_max / self._num_radius_divisions)).astype(int), self._num_radius_divisions - 1)
        theta_bins = (theta / (2*np.pi / self._num_angle_divisions)).astype(int)
        
        # 2D 히스토그램 생성        
        self.histograms = np.zeros((self._M, self._num_radius_divisions, self._num_angle_divisions), dtype=int)
        
        # np.add.at을 사용하여 히스토그램 계산
        for i in range(self._M):
            np.add.at(self.histograms[i], (r_bins[i], theta_bins[i]), 1)
    
    @set_timer()
    def get_candidates_indices(self, scores, num_of_candidates):        
        candidates_indices = np.argsort(scores, axis=0)[-num_of_candidates:][::-1] # 뒤에서부터 num_of_candidates개만 역순으로 추출        
        return candidates_indices
    
    @set_timer()
    def get_neighbor_points(self, i) -> np.ndarray:
        return np.array([kp.pt for kp in self.kp1])[self.N_nn_indices[i]]
    
        
        
        
                
        
