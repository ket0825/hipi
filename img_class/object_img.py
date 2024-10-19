import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

from img_class.base import ImgBase
from utils.common_utils import set_timer


class ObjectImg(ImgBase):
    """
    SHOULD KEEP:
    
    self.pca (to inverse_transform)
    
    self.pca_obj_kp (pca로 변환된 keypoint)
    
    self.kp1 (원본)
    
    self.img (원본)
    
    self.radius (scale 조절에 사용)
    
    """
    
    @property
    def __name__(self):
        return "ObjectImg" if not hasattr(self, "src") else f"ObjectImg - {self.src} "
    
    def __init__(self, src:os.PathLike):
        super().__init__(src, num_radius_divisions=4, num_angle_divisions=8)

    @set_timer()
    def set_orb(self, nfeatures:int, nms_distance=5):
        """
        removebg 처리하니 edgeThreshold 높아도 좋음.
        얇은 구조에 대해서 너무 많은 특징점이 검출되는 것을 방지하기 위해(clustering) non-maximum suppression을 사용함.
        nfeature에 따라 너무 적은 특징점이 나오는 경우 존재.
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
            patchSize=31, # edgeThreshold와 크거나 같은 값으로 설정해야 함.
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
            plt.title('object image orb')
            plt.show()
            
        print(f"object keypoints: {len(self.kp1)}")
        
        self._N = len(self.kp1)
        # return self.img, self.kp1

    def get_kp_length(self):
        return self._N
    
    @set_timer()
    def set_pca(self):
        self.pca = super().set_pca()
        self.pca_obj_kp = self.pca.fit_transform(np.array([kp.pt for kp in self.kp1]))
    
    @set_timer()    
    def set_histogram(self):
        self.radius = self.calculate_max_distance(self.pca_obj_kp) / 2 # 이후에 scale 조절에 사용.
        print(f"Object bounding circle radius: {self.radius}")
        print(f"super()._num_radius_divisions, super()._num_angle_divisions: {self._num_radius_divisions}, {self._num_angle_divisions}")        
        
        distances = np.sqrt(np.sum(self.pca_obj_kp**2, axis=1))                
        angles = np.arctan2(self.pca_obj_kp[:, 1], self.pca_obj_kp[:, 0])    
        angles = np.degrees(angles) % 360        
        radius_indices = np.minimum((distances / (self.radius / self._num_radius_divisions)).astype(int), self._num_radius_divisions - 1)                
        angle_indices = (angles / (360 / self._num_angle_divisions)).astype(int)
        self.histogram = np.zeros((self._num_radius_divisions, self._num_angle_divisions), dtype=int)
        np.add.at(self.histogram, (radius_indices, angle_indices), 1)
                
        