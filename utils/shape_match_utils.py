import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import Rbf
from scipy.stats import median_abs_deviation
import cv2
import matplotlib.pyplot as plt


from img_class.background_img import BackgroundImg
from img_class.object_img import ObjectImg



def compute_similarity(back_hist, obj_hist):
    denominator = np.sqrt(np.sum(back_hist**2) * np.sum(obj_hist**2))
    
    if denominator == 0:
        return 0        
    
    numerator = np.sum(back_hist * obj_hist)        
    return numerator / denominator    

def improved_similarity_measure(back_hist, obj_hist, num_angle_divisions, w1=0.7, w2=0.3):
       """
       이후에 추가 실험 예정.
       
       개선된 shape context 유사도 측정 함수
       back_hist, obj_hist: shape context 히스토그램 (2D numpy array)
       w1: bin-wise 유사도에 대한 가중치
       w2: 분포의 집중도에 대한 가중치        
       """
       # 1. 기존의 bin-wise 유사도. TODO: 사용한다면 rotation 고려 필요!
       bin_similarity = compute_similarity(back_hist, obj_hist)
       
       # 2. 분포의 집중도 고려
       concentration1 = np.sum(back_hist**2) / np.sum(back_hist)**2
       concentration2 = np.sum(obj_hist**2) / np.sum(obj_hist)**2
       concentration_similarity = 1 - abs(concentration1 - concentration2)              
              
       return w1 * bin_similarity + w2 * concentration_similarity

def scale_score(back_img: BackgroundImg, back_idx, obj_img: ObjectImg):
    f = min(back_img.radii[back_idx], obj_img.radius) / max(back_img.radii[back_idx], obj_img.radius)
    return f

def compute_refined_score(back_img: BackgroundImg, obj_img: ObjectImg, back_idx, scale_weight=0.3):     
    initial_score_og = compute_similarity(back_img.histograms[back_idx], obj_img.histogram)   
    # rotate: np.roll(obj_img.histogram, obj_img._num_angle_divisions//2, axis=1)
    rotated_obj_hist = np.roll(obj_img.histogram, obj_img._num_angle_divisions//2, axis=1)
    initial_score_rotated = compute_similarity(back_img.histograms[back_idx], rotated_obj_hist)
    
    t_nn_similarity = []
    if initial_score_rotated > initial_score_og:
        initial_score = initial_score_rotated
        back_img.is_rotated[back_idx] = 1
        for nn_idx in back_img.t_nn_indices[back_idx]:                        
            t_nn_similarity.append(compute_similarity(back_img.histograms[nn_idx], rotated_obj_hist))        
    else:
        initial_score = initial_score_og
        for nn_idx in back_img.t_nn_indices[back_idx]:                        
            t_nn_similarity.append(compute_similarity(back_img.histograms[nn_idx], obj_img.histogram))
        
    # TODO:눈에는 보여야 하니 지나친 scale 차이가 있으면 안됨. 이후에 고려.                     
    t_nn_score = np.average(t_nn_similarity)
    # 식이 약간 특이함...
    return initial_score * t_nn_score + scale_weight * scale_score(back_img, back_idx, obj_img) 
    
    