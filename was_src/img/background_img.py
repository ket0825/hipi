# Background image class
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from img.base import ImgBase
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from typing import List, Optional

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
    
    @classmethod
    def from_bytes(cls, img_bytes:bytes) -> "BackgroundImg":
        """
        img_bytes: alpha 채널을 가지고 있는 bytes
        
        이미지를 rgba 순서의 cv2 이미지로 변환.
        """   
        return cls(img_bytes=img_bytes)
        
    
    def __init__(self, src:Optional[os.PathLike] = None, img_bytes:bytes = None):
        import inspect
        
        methods = [name for name, value in inspect.getmembers(self, predicate=inspect.ismethod)
                if not name.startswith('_')]  # private 메소드 제외
                
        # 각 메소드에 대해 debug 속성 초기화
        for method_name in methods:
            setattr(self, f"_{method_name}_debug", False)
        super().__init__(src, img_bytes=img_bytes, num_radius_divisions=5, num_angle_divisions=12)
        
    @set_timer()
    def set_orb(self, nfeatures:int, max_keypoints=1000, nms_distance=3):
        """        
        충분히 큰 nfeatures 사용 필요. 
        
        윤곽선이 중요하지 않음.
        """
        DELIMINATOR = 200
        
        # 이미지 크기에 따른 파라미터 계산
        min_dim = min(self.img.shape[0], self.img.shape[1])        
        nms_distance = min(8, max(nms_distance, min_dim // DELIMINATOR))
        print(f"nms_distance: {nms_distance}")
                                
        # blur된 이미지를 gray로 변환 (바로 gray로 변환하지 않음)        
        gray = cv2.cvtColor(self.img, cv2.COLOR_RGB2GRAY)
        
        # edge-preserving filter
        filtered = cv2.bilateralFilter(gray, -1, 10, 5)
        
        if getattr(self, f"_set_orb_debug"):            
            plt.imshow(filtered)
            plt.title('filtered')
            plt.show()
                
        # sharpening kernel. kernel에 따라 blur of sharpening이 결정됨.
        sharpened = cv2.filter2D(filtered, -1, np.array([[-1, -1, -1], 
                                                         [-1, 9, -1], 
                                                         [-1, -1, -1]]))
        if getattr(self, f"_set_orb_debug"):
            plt.imshow(sharpened)
            plt.title('sharpened')
            plt.show()
        
        # # 적응형 임계값 처리. 오히려 아우라가 생겨 성능 감소.
        # thresh = cv2.adaptiveThreshold(
        #     sharpened, 
        #     255, 
        #     cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        #     cv2.THRESH_BINARY,
        #     blockSize=block_size,
        #     C=2
        #    )
        
        # plt.imshow(thresh)
        # plt.title('thresh')
        # plt.show()
        
        # 윤곽선이 중요하지 않음.
        # edges = cv2.Canny(sharpened, 100, 200)
        
        # plt.imshow(edges)
        # plt.title('edges')
        # plt.show()
                
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
            patchSize=31, # edgeThreshold와 크거나 같은 값으로 설정해야 함.
            fastThreshold=10, # FAST detector에서 근처 픽셀들이 얼마나 밝거나 어두워야 하는지에 대한 임계값
        )

        self.kp1, _ = orb.detectAndCompute(sharpened, None)
        print(f"일차 추출: {len(self.kp1)}")
        self.kp1 = ImgBase.non_max_suppression(self.kp1, nms_distance, max_keypoints=max_keypoints)
        print(f"NMS이후 추출: {len(self.kp1)}")
        if getattr(self, f"_set_orb_debug"):
            result = self.img.copy()
            for kp in self.kp1:
                x, y = kp.pt
                cv2.circle(result, (int(x), int(y)), 1, (255, 0, 0), -1)
                
            plt.imshow(result)
            plt.title(f'Background image orb\nnfeatures: {nfeatures}\nnms_distance: {nms_distance}')
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
        
        self.histograms = self.histograms.reshape(self._M, -1)
            
    def get_histograms(self) -> List[np.ndarray]:
        return self.histograms
    
    @set_timer()
    def get_candidates_indices(self, scores, num_of_candidates):        
        candidates_indices = np.argsort(scores, axis=0)[-num_of_candidates:][::-1] # 뒤에서부터 num_of_candidates개만 역순으로 추출        
        return candidates_indices
    
    @set_timer()
    def get_neighbor_points(self, i) -> np.ndarray:
        return np.array([kp.pt for kp in self.kp1])[self.N_nn_indices[i]]