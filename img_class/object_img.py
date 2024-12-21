# ObjectImg class
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

from img_class.base import ImgBase
from img_class.background_img import BackgroundImg
from utils.common_utils import set_timer

from sklearn.decomposition import PCA


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
        super().__init__(src, num_radius_divisions=5, num_angle_divisions=12)

    # TODO: 난이도, 시간 등 다양한 변인을 고려하여 target_size를 조정해야 함.
    def set_resize_scale(self, background_img: BackgroundImg):
        """
        배경 이미지 크기에 맞게 물체 이미지 크기 조정을 위한 scale 계산
        실제 resize는 set_orb 이후에 수행
        """
        bg_height, bg_width = background_img.img.shape[:2]
        obj_height, obj_width = self.img.shape[:2]
        
        # 배경 이미지의 1/4 크기로 제한
        target_size = min(bg_height, bg_width) // 4
        
        # 종횡비 유지하며 리사이징
        aspect_ratio = obj_width / obj_height
        if obj_width > obj_height:
            new_width = target_size
            new_height = int(new_width / aspect_ratio)
        else:
            new_height = target_size
            new_width = int(new_height * aspect_ratio)
        
        self.scale_x = new_width / obj_width
        self.scale_y = new_height / obj_height
        self.resized_shape = (new_width, new_height)
        

    @set_timer()
    def set_orb(self, nfeatures:int):
        """
        removebg 처리하니 edgeThreshold 높아도 좋음.
        얇은 구조에 대해서 너무 많은 특징점이 검출되는 것을 방지하기 위해(clustering) non-maximum suppression을 사용함.
        nfeature에 따라 너무 적은 특징점이 나오는 경우 존재.
        """                
        DELIMINATOR = 33                        
        
        # 이미지 크기에 따른 파라미터 계산
        min_dim = min(self.img.shape[0], self.img.shape[1])
        # object image의 경우 이미지 크기가 다양하므로 edgeThreshold를 조정해야 함.                
        orb_edgeThreshold = max(3, min(min_dim // DELIMINATOR, 15))
        orb_patchSize = min(orb_edgeThreshold*2 + 1, 31)        
      
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        
        filtered = cv2.bilateralFilter(gray, -1, 10, 5)
        if getattr(self, "_set_orb_debug"):
            plt.imshow(filtered)
            plt.title('filtered')
            plt.show()
        
        edges = cv2.Canny(filtered, 100, 200)
        if getattr(self, "_set_orb_debug"):
            plt.imshow(edges)
            plt.title('edges')
            plt.show()
        
        # orb = cv2.ORB()
        # orb = orb.create(
        #     nfeatures=nfeatures, # 상위 몇개의 특징점을 사용할 것인지
        #     scaleFactor=1.2, 
        #     nlevels=8,
        #     edgeThreshold=orb_edgeThreshold,
        #     # Backgroud image는 이 값을 줄여야 함.
        #     firstLevel=0,
        #     WTA_K=2, # BRIEF descriptor가 사용할 bit 가짓수. binary이므로 1bit임.
        #     scoreType=cv2.ORB_HARRIS_SCORE, 
        #     patchSize=orb_patchSize,
        #     fastThreshold=10, # FAST detector에서 근처 픽셀들이 얼마나 밝거나 어두워야 하는지에 대한 임계값
        # )
        # self.kp1, _ = orb.detectAndCompute(edges, None)
        
        contour_features = int(nfeatures * 0.7)  # 70% for contour
        internal_features = nfeatures - contour_features
        
        # Get contour region
        dilated_edges = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=1)
        
        # Detect features in contour region
        orb = cv2.ORB()
        orb_contour = orb.create(
            nfeatures=contour_features, # 상위 몇개의 특징점을 사용할 것인지
            scaleFactor=1.2, 
            nlevels=8,
            edgeThreshold=orb_edgeThreshold,
            # Backgroud image는 이 값을 줄여야 함.
            firstLevel=0,
            WTA_K=2, # BRIEF descriptor가 사용할 bit 가짓수. binary이므로 1bit임.
            scoreType=cv2.ORB_HARRIS_SCORE, 
            patchSize=orb_patchSize,
            fastThreshold=10, # FAST detector에서 근처 픽셀들이 얼마나 밝거나 어두워야 하는지에 대한 임계값
        )                
        kp_contour, _ = orb_contour.detectAndCompute(edges, mask=dilated_edges)
        
        # Detect features in internal region
        orb_internal = orb.create(
            nfeatures=internal_features, # 상위 몇개의 특징점을 사용할 것인지
            scaleFactor=1.2, 
            nlevels=8,
            edgeThreshold=orb_edgeThreshold,
            # Backgroud image는 이 값을 줄여야 함.
            firstLevel=0,
            WTA_K=2, # BRIEF descriptor가 사용할 bit 가짓수. binary이므로 1bit임.
            scoreType=cv2.ORB_HARRIS_SCORE, 
            patchSize=orb_patchSize,
            fastThreshold=10, # FAST detector에서 근처 픽셀들이 얼마나 밝거나 어두워야 하는지에 대한 임계값
        )                
        internal_mask = ~dilated_edges
        kp_internal, _ = orb_internal.detectAndCompute(gray, mask=internal_mask)
        
        print(f"contour keypoints: {len(kp_contour)}, internal keypoints: {len(kp_internal)}")
        
        # Combine keypoints
        self.kp1 = kp_contour + kp_internal
        if getattr(self, "_set_orb_debug"):
            circle_size = min(self.img.shape[:2]) // 200            
            result = self.img.copy()
            for internal_kp in kp_internal:
                x, y = internal_kp.pt
                cv2.circle(result, (int(x), int(y)), circle_size, (0, 255, 0), -1)
            
            print(f"orb_patchSize: {orb_patchSize}\norb_edgeThreshold: {orb_edgeThreshold}")    
            plt.imshow(result)
            plt.title(
                f"""
                internal object image orb
                circle_size: {circle_size}
                count: {len(kp_internal)}
                orb_patchSize: {orb_patchSize}
                orb_edgeThreshold: {orb_edgeThreshold}                
                    """)
            plt.show()
            
        if getattr(self, "_set_orb_debug"):
            circle_size = min(self.img.shape[:2]) // 200
            result = self.img.copy()
            for contour_kp in kp_contour:
                x, y = contour_kp.pt
                cv2.circle(result, (int(x), int(y)), circle_size, (0, 255, 0), -1)
            
            print(f"orb_patchSize: {orb_patchSize}\norb_edgeThreshold: {orb_edgeThreshold}")    
            plt.imshow(result)
            plt.title(
                f"""
                contour object image orb
                circle_size: {circle_size}
                count: {len(kp_contour)}
                orb_patchSize: {orb_patchSize}
                orb_edgeThreshold: {orb_edgeThreshold}
                    """)
            plt.show()                                        
                                        
        # 이미지와 keypoint 리사이징
        self.resize_image_and_keypoints()
        
        min_dim = min(self.img.shape[0], self.img.shape[1])
        nms_distance = max(3, min(min_dim // DELIMINATOR, 7))
        self.kp1 = ImgBase.non_max_suppression(self.kp1, nms_distance)
        
        if getattr(self, f"_set_orb_debug"):
            circle_size = min(self.img.shape[:2]) // 150
            result = self.img.copy()
            for kp in self.kp1:
                x, y = kp.pt
                cv2.circle(result, (int(x), int(y)), circle_size, (0, 0, 255), -1)
            print(f"orb_patchSize: {orb_patchSize}\norb_edgeThreshold: {orb_edgeThreshold}")    
            plt.imshow(result)
            plt.title(
                f"""
                object image orb
                orb_patchSize: {orb_patchSize}
                orb_edgeThreshold: {orb_edgeThreshold}
                orb_nms_distance: {nms_distance}
                    """)
            plt.show()
            
        print(f"total object keypoints: {len(self.kp1)}")
        
        self._N = len(self.kp1)
        
    def resize_image_and_keypoints(self):
        
        self.img = cv2.resize(self.img, self.resized_shape)
       
       # Keypoint 좌표 조정
        scaled_keypoints = []
        for kp in self.kp1:
            new_kp = cv2.KeyPoint(
                x = kp.pt[0] * self.scale_x,
                y = kp.pt[1] * self.scale_y,
                size = kp.size * min(self.scale_x, self.scale_y),  # size도 scale에 맞게 조정
                angle = kp.angle,
                response = kp.response,
                octave = kp.octave,
                class_id = kp.class_id
            )
            scaled_keypoints.append(new_kp)
        
        self.kp1 = scaled_keypoints

    def get_kp_length(self):
        return self._N
    
    @set_timer()
    def set_pca(self):
        self.pca = PCA(n_components=2)
        self.pca_obj_kp = self.pca.fit_transform(np.array([kp.pt for kp in self.kp1]))
    
    @set_timer()    
    def set_histogram(self):                                                        
        self.radius = np.max(np.linalg.norm(self.pca_obj_kp, axis=1)) # 이후에 scale 조절에 사용.
        print(f"Object bounding circle radius: {self.radius}")
        print(f"super()._num_radius_divisions, super()._num_angle_divisions: {self._num_radius_divisions}, {self._num_angle_divisions}")                
        distances = np.linalg.norm(self.pca_obj_kp, axis=1)
        angles = np.arctan2(self.pca_obj_kp[:, 1], self.pca_obj_kp[:, 0])    
        angles = np.degrees(angles) % 360        
        radius_indices = np.minimum((distances / (self.radius / self._num_radius_divisions)).astype(int), self._num_radius_divisions - 1)                
        angle_indices = (angles / (360 / self._num_angle_divisions)).astype(int)
        self.histogram = np.zeros((self._num_radius_divisions, self._num_angle_divisions), dtype=int)
        np.add.at(self.histogram, (radius_indices, angle_indices), 1)
    
    @set_timer()
    def get_kp_center(self):
        return np.mean(np.array([kp.pt for kp in self.kp1]), axis=0)
                
        