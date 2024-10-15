from img_class.background_img import BackgroundImg
from img_class.object_img import ObjectImg
import numpy as np

def compute_similarity(back_hist, obj_hist, num_angle_divisions):    
    og_numerator = np.sum(back_hist * obj_hist)    
    rotated_numerator = np.sum(back_hist * np.roll(obj_hist, num_angle_divisions//2, axis=1)) # 180도 회전
        
    denominator = np.sqrt(np.sum(back_hist**2) * np.sum(obj_hist**2))
    
    if denominator == 0:
        return 0
    
    return max(og_numerator, rotated_numerator) / denominator

def improved_similarity_measure(back_hist, obj_hist, num_angle_divisions, w1=0.7, w2=0.3):
       """
       이후에 추가 실험 예정.
       
       개선된 shape context 유사도 측정 함수
       back_hist, obj_hist: shape context 히스토그램 (2D numpy array)
       w1: bin-wise 유사도에 대한 가중치
       w2: 분포의 집중도에 대한 가중치        
       """
       # 1. 기존의 bin-wise 유사도
       bin_similarity = compute_similarity(back_hist, obj_hist, num_angle_divisions=num_angle_divisions)
       
       # 2. 분포의 집중도 고려
       concentration1 = np.sum(back_hist**2) / np.sum(back_hist)**2
       concentration2 = np.sum(obj_hist**2) / np.sum(obj_hist)**2
       concentration_similarity = 1 - abs(concentration1 - concentration2)              
              
       return w1 * bin_similarity + w2 * concentration_similarity

def scale_score(back_img: BackgroundImg, back_idx, obj_img: ObjectImg):
    f = min(back_img.radii[back_idx].astype(int), obj_img.radius) / max(back_img.radii[back_idx].astype(int), obj_img.radius)
    return f

def compute_refined_score(back_img: BackgroundImg, obj_img: ObjectImg, back_idx, scale_weight=0.3):     
    initial_score = compute_similarity(back_img.histograms[back_idx], obj_img.histogram, num_angle_divisions=obj_img._num_angle_divisions)
    
    t_nn_similarity = []
        
    for nn_idx in back_img.t_nn_indices[back_idx]:
        t_nn_similarity.append(compute_similarity(back_img.histograms[nn_idx], obj_img.histogram, num_angle_divisions=obj_img._num_angle_divisions))        
    
    t_nn_score = np.average(t_nn_similarity)
    
    return initial_score * t_nn_score + scale_weight * scale_score(back_img, back_idx, obj_img)
    
    
    
    
    
    
    