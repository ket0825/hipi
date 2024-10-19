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
    
def center_points(points):
    return points - np.mean(points, axis=0)

def rotate_points(points, angle):
    return points.dot(np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]))
        
def normalize_points(points, angle=None):    
    center = np.mean(points, axis=0)
    centered = points - center
    scale = np.mean(np.linalg.norm(centered, axis=1))        
    
    normalized = centered / scale
    if angle is not None:
        normalized = normalized.dot(np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]))
        
    normalized = normalized[np.argsort(np.arctan2(normalized[:,1], normalized[:,0]))] # sort by angle. -pi to pi    
    return normalized, center, scale

def denormalize_params(params, src_center, src_scale, dst_center, dst_scale):
    """Denormalize affine parameters."""
    a, b, c, d, e, f = params.reshape(6)
    return np.array([
        a * src_scale / dst_scale,
        b * src_scale / dst_scale,
        (c * dst_scale + dst_center[0] - a * src_center[0] - b * src_center[1]) / dst_scale,
        d * src_scale / dst_scale,
        e * src_scale / dst_scale,
        (f * dst_scale + dst_center[1] - d * src_center[0] - e * src_center[1]) / dst_scale
    ])
    # a * src_scale / dst_scale, b * src_scale / dst_scale,
    # 

def affine_transform(params, points):
    """Apply affine transformation."""
    a, b, c, d, e, f = params.reshape(6)
    x, y = points[:, 0], points[:, 1]
    return np.column_stack((a*x + b*y + c, d*x + e*y + f))

def estimate_affine(src_points, dst_points):
    """Estimate affine transformation using least squares."""
    A = np.column_stack((src_points, np.ones(src_points.shape[0])))    
    B = dst_points    
    params, _, _, _ = np.linalg.lstsq(A, B, rcond=None)    
    return params

def ransac_affine(back_img:BackgroundImg, obj_img:ObjectImg, candidate_indices:np.ndarray, n_iterations=1000, mad_factor=1.4826, mad_threshold=3):
    """
    Ransac for affine transformation.
    
    MAD (Median Absolute Deviation)를 이용한 threshold 계산.
    
    scale과 rotation이 초기값으로 허용되지 않아 문제 발생.
        
    """
    # src_points: object image keypoints
    src_points = np.array([kp.pt for kp in obj_img.kp1])    
    # dst_points: background image keypoints
    dst_points = np.array([kp.pt for kp in back_img.kp1])[back_img.N_nn_indices[candidate_indices]] # shape: (len(candidate_indices) x N x 2)
    
    # 후보군에 따른 추가 계산이 필요하다면 이후에 추가
    dst_points = dst_points[0, :, :] # shape: (N x 2)                
    
    best_inliers = []
    best_params = None
    n_points = src_points.shape[0]    
    
    initial_rotates = []
    where_rotates = []
    for i, idx in enumerate(candidate_indices):
        initial_rotates.append(back_img.pca_class_list[idx].components_[0] @ obj_img.pca.components_[0]) # @: matrix multiplication => cosine theta value                           
        if back_img.is_rotated[idx]:
            where_rotates.append(i)
                
    initial_rotates = np.arccos(np.array([back_img.pca_class_list[idx].components_[0] @ obj_img.pca.components_[0] for idx in candidate_indices]))
    if where_rotates:
        where_rotates = np.array(where_rotates)    
        initial_rotates[where_rotates] = initial_rotates[where_rotates] + np.pi    
        print(f"initial_rotates: {initial_rotates}")    
        
    initial_scales = []
    for idx in candidate_indices:
        initial_scales.append(back_img.radii[idx] / obj_img.radius)
    
    # 후보군에 따른 추가 계산이 필요하다면 이후에 추가
    inital_rotate = initial_rotates[0]
    initial_scale = initial_scales[0]           
    
    # Normalize points
    src_normalized, src_center, src_scale = normalize_points(src_points, angle=inital_rotate)
    dst_normalized, dst_center, dst_scale = normalize_points(dst_points)                                
        
    cos_theta, sin_theta = np.cos(inital_rotate), np.sin(inital_rotate)
    initial_params = np.array([
        initial_scale * cos_theta, -initial_scale * sin_theta, 0,
        initial_scale * sin_theta, initial_scale * cos_theta, 0
    ])
    
    for _ in range(n_iterations):
        sample_indices = np.random.choice(n_points, 3, replace=False)
        sample_src_points = src_normalized[sample_indices]
        sample_dst_points = dst_normalized[sample_indices]
        
        params = estimate_affine(sample_src_points, sample_dst_points)
        
        transformed_points = affine_transform(params, src_normalized)
        
        residuals = np.linalg.norm(transformed_points - dst_normalized, axis=1)        
        mad = median_abs_deviation(residuals)        
        threshold = mad * mad_factor # 1.4826: MAD to STD conversion factor
                                        
        inliers = np.where(residuals < threshold)[0]        
        if len(inliers) > len(best_inliers):
            if threshold > mad_threshold:
                # 그 때 아핀 변환을 한 값을 보고싶음...
                plt.scatter(transformed_points[:, 0], transformed_points[:, 1], c='r', s=3)                                
                plt.scatter(dst_normalized[:, 0], dst_normalized[:, 1], c='b', s=3)            
                plt.legend(['transformed', 'dst'])
                plt.title(f"threshold: {threshold}, inliers: {len(inliers)}")
                plt.show()
                
                continue
            print(f"threshold: {threshold}, inliers: {len(inliers)}")
            best_inliers = inliers
            best_params = params
            best_residuals = residuals
    
    if len(best_inliers) > 3:
        print(f"best_inliers: {best_inliers}")
        # best_params = estimate_affine(src_points[best_inliers], dst_points[best_inliers])
        best_params = denormalize_params(best_params, src_center, src_scale, dst_center, dst_scale)
        print(f"best_params: {best_params}")
        affine_transformed = affine_transform(best_params, src_points)
        plt.scatter(affine_transformed[:, 0], affine_transformed[:, 1], c='r', s=3)
        plt.scatter(dst_points[:, 0], dst_points[:, 1], c='b', s=3)
        plt.legend(['src', 'dst'])
        plt.title('best_inliers')
        plt.show()
    
    return best_params        


def stable_affine(back_img:BackgroundImg, obj_img:ObjectImg, candidate_indices:np.ndarray):
    """
    iterative stable affine transformation.                        
    """
    # src_points: object image keypoints
    src_points = np.array([kp.pt for kp in obj_img.kp1])    
    # dst_points: background image keypoints
    dst_points = np.array([kp.pt for kp in back_img.kp1])[back_img.N_nn_indices[candidate_indices]] # shape: (len(candidate_indices) x N x 2)    
    # 후보군에 따른 추가 계산이 필요하다면 이후에 추가
    dst_points = dst_points[0, :, :] # shape: (N x 2)                

    # 고려해야 할 것.
    # 1. src PCA 주성분이 음수인 경우
    # 2. dst PCA 주성분이 음수인 경우
    # 3. rotate된 src PCA 주성분이 음수인 경우
    
    initial_rotates = []
    where_rotates = []
    for i, idx in enumerate(candidate_indices):        
        if back_img.is_rotated[idx]:
            where_rotates.append(i)    
                
    initial_rotates = np.arccos(np.array([back_img.pca_class_list[idx].components_[0] @ obj_img.pca.components_[0] for idx in candidate_indices]))
    if where_rotates:
        where_rotates = np.array(where_rotates)
        initial_rotates[where_rotates] = (2*np.pi - initial_rotates[where_rotates]) % (2*np.pi) # 0 to 2pi
        print(f"initial_rotates: {initial_rotates}")    
        
    initial_scales = []
    for idx in candidate_indices:
        initial_scales.append(back_img.radii[idx] / obj_img.radius)
    
    plt.imshow(back_img.img)    
    plt.scatter(dst_points[:, 0], dst_points[:, 1], c='b', s=3)    
    plt.show()
        
    # 후보군에 따른 추가 계산이 필요하다면 이후에 추가
    inital_rotate = initial_rotates[0]
    initial_scale = initial_scales[0]            
    print("initial_scale", initial_scale)    
    print("Scale: ",back_img.radii[candidate_indices[0]], obj_img.radius)
    
    # center points
    src_center = center_points(src_points)
    dst_center = center_points(dst_points)
    dst_center = dst_center[np.argsort(np.arctan2(dst_center[:,1], dst_center[:,0]))] # sort by angle. 0 to 2pi
    
    # rotate points
    src_rotated = rotate_points(src_center, inital_rotate)
    src_rotated = src_rotated[np.argsort(np.arctan2(src_rotated[:,1], src_rotated[:,0]))] # sort by angle. 0 to 2pi
    src_scaled = src_rotated * initial_scale
            
    plt.scatter(src_scaled[:, 0], src_scaled[:, 1], c='r', s=3)
    plt.scatter(dst_center[:, 0], dst_center[:, 1], c='b', s=3)
    plt.legend(['src_scaled', 'dst'])
    plt.title('scaled points')
    plt.show()
    
    src_solution = src_scaled
    dst_solution = dst_center
    params = None
    while True:
        params = estimate_affine(src_solution, dst_solution)
        transformed_points = affine_transform(params, src_solution)
        squared_residuals = np.sum((transformed_points - dst_solution)**2, axis=1)
        delta = np.mean(squared_residuals)
        probs = np.exp(-squared_residuals / (2 * delta))
        omega = np.mean(probs)
        
        inlier_indices = np.where(probs <= 1.3 * omega)[0]        
        if len(inlier_indices) == len(src_solution):
            print("No outliers found")
            break
        
        src_solution = src_solution[inlier_indices]
        dst_solution = dst_solution[inlier_indices]       
        
    plt.scatter(affine_transform(params, src_solution)[:, 0], affine_transform(params, src_solution)[:, 1], c='r', s=3)
    # plt.scatter(transformed_points[:, 0], transformed_points[:, 1], c='r', s=3)
    plt.scatter(dst_solution[:, 0], dst_solution[:, 1], c='b', s=3)
    plt.legend(['transformed_src', 'dst'])
    
    return params
    
    