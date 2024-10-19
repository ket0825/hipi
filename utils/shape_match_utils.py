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

def interpolate_feature_scores(back_img:BackgroundImg, scores):
    # 이미지 로드 (BGR 형식)    
    height, width = back_img.img.shape[:2]
    feature_points = np.array([kp.pt for kp in back_img.kp1])
    # RBF 보간기 생성 (TPS와 유사)
    rbf = Rbf(feature_points[:, 0], feature_points[:, 1], scores, function='thin_plate')

    # 전체 이미지에 대한 그리드 생성
    x = np.arange(0, width)
    y = np.arange(0, height)
    X, Y = np.meshgrid(x, y)

    # 보간 수행
    Z = rbf(X, Y)

    return X, Y, Z

def visualize_results(X, Y, Z, back_img:BackgroundImg):
    # 결과 정규화 (0-255 범위로)
    Z_normalized = cv2.normalize(Z, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Heatmap 생성 (1채널을 3채널로 변환)
    heatmap = cv2.applyColorMap(Z_normalized, cv2.COLORMAP_JET)

    # 원본 이미지와 heatmap 합성
    alpha = 0.6  # heatmap의 투명도
    image = back_img.img
    overlay = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)
    
    # 결과 저장
    cv2.imwrite('heatmap_overlay.jpg', overlay)

    # Matplotlib을 사용한 시각화
    plt.figure(figsize=(18, 6))

    # 3D 표면 플롯
    ax1 = plt.subplot(131, projection='3d')
    ax1.plot_surface(X, Y, Z, cmap='viridis')
    ax1.set_title('Interpolated Surface')

    # 2D Heatmap
    ax2 = plt.subplot(132)
    im = ax2.imshow(Z, cmap='viridis')
    ax2.set_title('2D Heatmap')
    plt.colorbar(im)

    # Heatmap Overlay
    ax3 = plt.subplot(133)
    ax3.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    ax3.set_title('Heatmap Overlay on Original Image')

    plt.tight_layout()
    plt.savefig('visualization_results.png')
    plt.close()
    