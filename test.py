# 한 번 추적해가며 디버깅 해보기
from img_class.background_img import BackgroundImg
from img_class.object_img import ObjectImg
from utils.shape_match_utils import compute_refined_score, interpolate_feature_scores, visualize_results, ransac_affine, stable_affine, apply_affine_matrix, localize_object
import time
import matplotlib.pyplot as plt

DEBUG = True
# obj_img = ObjectImg("apple.jpg")
obj_img = ObjectImg("apple_removebg.png")
plt.imshow(obj_img.img)
plt.show()
back_img = BackgroundImg("forest.jpg")
obj_img.set_resize_scale(back_img)

if DEBUG:
    print("[WARNING] DEBUG MODE: YOU ARE IN DEBUG MODE. PLEASE debug=False IN PRODUCTION.")

# debug settings.
obj_img.set_method_debug("set_orb", debug=True)
obj_img.set_method_debug("set_pca", debug=DEBUG)
obj_img.set_method_debug("set_histogram", debug=DEBUG)
back_img.set_method_debug("set_orb", debug=True)
back_img.set_method_debug("set_pca", debug=DEBUG)
back_img.set_method_debug("set_histograms", debug=DEBUG)

obj_img.set_orb(nfeatures=300)
back_img.set_orb(nfeatures=1000, nms_distance=3)

N = obj_img.get_kp_length()
M = back_img.get_kp_length()
T = 8
NUM_OF_CANDIDATES = 10
if DEBUG:
    print(f"object keypoints: {N}, background keypoints: {M}, T: {T}")

obj_img.set_pca()
back_img.set_pca(N, T)

obj_img.set_histogram()
back_img.set_histograms()
if DEBUG:
    t1 = time.time()
scores = [compute_refined_score(back_img, obj_img, m, scale_weight=0.3) for m in range(M)]
if DEBUG:
    t2 = time.time()
    print(f"elapsed time in compute_refined_scores: {t2 - t1}")

print(min(scores), max(scores))

candidate_indices = back_img.get_candidates_indices(scores, NUM_OF_CANDIDATES)
print(f"candidate_indices: {candidate_indices}")
# affine_matrix = ransac_affine(back_img, obj_img, candidate_indices, 1000, mad_factor=1, mad_threshold=1)
# print(affine_matrix)
localize_object(back_img, obj_img, candidate_indices)
# affine_matrix, best_idx, scale = stable_affine(back_img, obj_img, candidate_indices)
# affine_matrix = transform_object(back_img, obj_img, candidate_indices)
# result = apply_affine_matrix(back_img, best_idx, obj_img, scale, affine_matrix)


# For score visualization
# X, Y, Z = interpolate_feature_scores(back_img, scores)
# print(Z)
# visualize_results(X, Y, Z, back_img)

