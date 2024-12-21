import cv2
import io
import numpy as np
import heapq

from typing import List, Optional, Tuple
from img.best_match import BestMatch


from img.object_img import ObjectImg
from img.background_img import BackgroundImg
from img.shape_match_utils import (
    compute_similarity,
    scale_score
)

def preprocess_insert_img(img_bytes:bytes) -> ObjectImg:
    """
    img: alpha 채널을 가지고 있는 bytes
    
    이미지를 rgba 순서의 cv2 이미지로 변환.
    """
    obj = ObjectImg.from_bytes(img_bytes)    
    # 원본 이미지를 하나가 특정 shape이 되도록 리사이징.
    # 저장하기 위한 것이기에 bg_ratio에 들어갈 비율은 그대로 1.    
    obj.set_resize_scale_by_shape(shape=(1024,1024), bg_ratio=1)
    obj.set_orb(nfeatures=500, max_keypoints=100) # 성능 테스트 필요.
    obj.set_pca()
    print(f"keypoint 개수: {obj.get_kp_length()}")    
    obj.set_histogram()
    return obj

def preprocess_query_img(img_bytes:bytes) -> BackgroundImg:
    """
    img: alpha 채널을 가지고 있는 bytes
    
    이미지를 rgba 순서의 cv2 이미지로 변환.
    """
    back = BackgroundImg.from_bytes(img_bytes)    
    # 원본 이미지를 하나가 특정 shape이 되도록 리사이징.
    # 저장하기 위한 것이기에 bg_ratio에 들어갈 비율은 그대로 1.    
    back.set_orb(nfeatures=2000, max_keypoints=1000, nms_distance=3) # 성능 테스트 필요.
    back.set_pca(N=100, T=12) # N과 T는 고정. N은 상위 100개 obj keypoint. T는 division 개수.
    back.set_histograms()  
    return back
    # scores = [compute_refined_score(back_img, obj_img, m, scale_weight=0.3) for m in range(M)]    

def t_nn_score(back_img: BackgroundImg, best_matches: List[BestMatch], scale_weight=0.3) -> List[BestMatch]:
    # 이게 이미지 best_matches야.
    # initial_score_og = compute_similarity(back_img.histograms[back_idx], obj_img.histogram)
    
    # rotate: np.roll(obj_img.histogram, obj_img._num_angle_divisions//2, axis=1)
    # rotated_obj_hist = np.roll(obj_img.histogram, obj_img._num_angle_divisions//2, axis=1)
    # initial_score_rotated = compute_similarity(back_img.histograms[back_idx], rotated_obj_hist)
            
    for back_idx in range(back_img._M):
        t_nn_similarity = []        
        for nn_idx in back_img.t_nn_indices[back_idx]:
            t_nn_similarity.append(compute_similarity(back_img.histograms[nn_idx], best_matches[back_idx].histogram))
        t_nn_score = np.average(t_nn_similarity) # T개의 값이 들어감.
        # TODO: 난이도에 따라 달라질 수 있지만 눈에는 보여야 하니 background에서 scale 차이가 너무 크면 안됨.
        # if back_img.radii[back_idx] < min(back_img.img.shape[:2]) * 0.05 and back_img.radii[back_idx] > min(back_img.img.shape[:2]):
        #     best_matches[back_idx].t_nn_score = 0.0
        if (back_img.radii[back_idx] < min(back_img.img.shape[:2]) * 0.05 
            or back_img.radii[back_idx] > min(best_matches[back_idx].width, best_matches[back_idx].height) * 0.3
            or back_img.radii[back_idx] > best_matches[back_idx].radius *0.3
            ):
            best_matches[back_idx].t_nn_score = 0.0
        else:            
            best_matches[back_idx].t_nn_score = t_nn_score * best_matches[back_idx].distance + scale_weight * scale_score(back_img, back_idx, best_matches[back_idx])
            
    return best_matches

def place_non_overlapping_matches(back: BackgroundImg, best_matches: List[BestMatch]) -> List[int]:        
    """
    return: (object vectordb id, backgroundimg keypoints back_idx) list
    """
    scores = [match.t_nn_score for match in best_matches if match.t_nn_score > 0.0]
    candidates_indices = back.get_candidates_indices(scores, back._M) # M개의 t_nn_score기준 역순 정렬.        
    
    grid = np.zeros((back.img.shape[0], back.img.shape[1]), dtype=np.bool)        
    def check_overlap(bbox):
        x, y, w, h = bbox
        return np.any(grid[y:y+h, x:x+w])
            
    id_set = set()        
    candidates = [] # (back_idx). score 기준으로 정렬된 후보지.
    for idx in candidates_indices:
        back_idx = idx        
        x, y = back.kp1[back_idx].pt
        x = int(x) - 1
        y = int(y) - 1
        radius = back.radii[back_idx]
        box_size = int(radius * 2) + 2
        bbox = (int(x - box_size/2), int(y - box_size/2), box_size, box_size)
        
        if ( 
            best_matches[back_idx].id in id_set # 만약 하나씩만 넣는 경우.
            or check_overlap(bbox) # 겹치는 경우.
            ):
            continue
        
        grid[y:y+box_size, x:x+box_size] = True
        candidates.append(idx)
        id_set.add(best_matches[back_idx].id)
    
    return candidates

def localize_object(back_img:BackgroundImg, best_matches:List[BestMatch], obj_imgs:List[ObjectImg], candidate_indices:List[int]
                    ) -> bytes:
    # dst_points: background image keypoints
    dst_points = np.array([kp.pt for kp in back_img.kp1])[back_img.N_nn_indices[candidate_indices]] # shape: (len(candidate_indices) x N x 2)
    # 후보군에 따른 추가 계산이 필요하다면 이후에 추가
    # dst_points = dst_points[0, :, :] # shape: (N x 2)                

    # 고려해야 할 것.
    # 1. src PCA 주성분이 음수인 경우
    # 2. dst PCA 주성분이 음수인 경우
    # 3. rotate된 src PCA 주성분이 음수인 경우
    
    # shape: (len(candidate_indices) x N x 2)
    result_img = back_img.img.copy()
    back_kps_centers = np.array([back_img.pca_class_list[idx].mean_ for idx in candidate_indices])
    for obj_seq, idx in enumerate(candidate_indices):                
        initial_scale = back_img.radii[idx] / best_matches[idx].radius
        if initial_scale > 0.3:
            print(f"[PASS] initial_scale: {initial_scale}")
            continue
        print(f"initial_scale: {initial_scale}")
        rotation_angle = np.arccos(np.dot(back_img.pca_class_list[idx].components_[0], best_matches[idx].pca_comp))
        back_kp_center = back_kps_centers[obj_seq]
        # where_rotates = np.where(back_img.is_rotated[candidate_indices])[0] # 반 바퀴 회전한 경우 확인.
        # rotation_angles[where_rotates] = rotation_angles[where_rotates] + np.pi # 반 바퀴 회전시킴
        
        # if back_img.is_rotated[idx]:
        #     rotation_angle = rotation_angle + np.pi
        # back_img에서의 매칭 포인트의 중앙점
        M = cv2.getRotationMatrix2D((best_matches[idx].width//2, best_matches[idx].height//2), 
                                    rotation_angle * 180 / np.pi, 
                                    initial_scale)
        # back_img에서의 매칭 포인트의 중앙점과 object_img의 중앙점을 일치시키기 위한 translation 계산
        obj_center = best_matches[idx].kp_center
        shift = back_kp_center - obj_center
        M[0, 2] += shift[0]
        M[1, 2] += shift[1]
        
        # 흰색 캔버스 생성
        white_canvas = np.full((back_img.img.shape[0], back_img.img.shape[1], 3), 255, dtype=np.uint8)
        
        # 새로운 객체를 별도의 캔버스에 변환
        transformed_obj = cv2.warpAffine(obj_imgs[obj_seq].img, M, 
                                        (back_img.img.shape[1], back_img.img.shape[0]),
                                        dst=white_canvas,
                                        flags=cv2.INTER_LINEAR,
                                        borderMode=cv2.BORDER_TRANSPARENT)
        # print("중간 확인")
        # cv2.imwrite('transformed.png', cv2.cvtColor(transformed_obj, cv2.COLOR_BGR2RGB))
        
        # 흰색 마스크 생성
        white_threshold = 250
        transformed_mask = np.all(transformed_obj >= white_threshold, axis=-1)
        # 마스크를 사용하여 현재 result_img에 새 객체 합성
        result_img[~transformed_mask] = transformed_obj[~transformed_mask]
        
        # transformed_obj = cv2.warpAffine(obj_imgs[obj_seq].img, M, 
        #                                 (back_img.img.shape[1], back_img.img.shape[0]),
        #                                 dst=result_img,
        #                                 flags=cv2.INTER_LINEAR,
        #                                 borderMode=cv2.BORDER_TRANSPARENT)        
        # # 2. 흰색 배경 마스킹 (RGB 값이 모두 높은 픽셀을 찾음)
        # # 각 채널이 예를 들어 250 이상인 픽셀을 흰색으로 간주
        # print("중간 확인")
        # cv2.imwrite('transformed.png', cv2.cvtColor(transformed_obj, cv2.COLOR_BGR2RGB))
        # white_threshold = 250
        # transformed_mask = np.all(transformed_obj >= white_threshold, axis=-1)        
        # # 3. 결과 이미지 생성
        # result_img = transformed_obj.copy()
        # result_img[transformed_mask] = back_img.img[transformed_mask]            
    from datetime import datetime
    now = datetime.now().strftime("%Y%m%d%H%M%S")    
    cv2.imwrite(f'./result_imgs/{now}_results.png', cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
    
    return result_img.tobytes()    
     
        
        