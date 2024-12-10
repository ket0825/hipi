# sweep line algorithm
# 1. heapq.heappop(max_pq) # (score, index)
# 2. 해당 후보지의 index에 대한 bbox를 가져옴.
# 3. bbox를 기준으로 x와 y에 대한 sweep line을 그림.
# 4. 만약 bbox가 겹치는 경우, 해당 후보지의 다음 점수를 가져오고, current를 1 증가시킴.
# 5. 다시 heapq.heappush(max_pq, (score, index))를 수행.
# 6  heapq가 비어있거나, current가 candidate_len를 초과할 때 까지 1~5를 반복.
import heapq
# 3개. 3개의 후보지에 따른 점수, 3개의 후보지의 bbox를 지님.
image_width = 100
image_height = 100

score_bbox_candidates = [
    {
        "scores": [1, 0.99, 0.98],
        "bbox": [(2,2,5,5), (3,3,6,6), (4,4,7,7)], # (x, y, w, h)
        "current": 0
    },
    {
        "scores": [0.9, 0.7, 0.6],
        "bbox": [(5,5,7,7), (12,12,7,7), (20,20,8,8)],
        "current": 0
    },
    {
        "scores": [0.8, 0.6, 0.5],
        "bbox": [(10,10,5,5), (5,5,8,8), (6,6,9,9)],
        "current": 0
    }
]
print(score_bbox_candidates)

candidate_len = len(score_bbox_candidates) # 3
max_pq = []
score_bbox_candidates.sort(key=lambda x: x["scores"][0], reverse=True)
# max heap
for i in range(candidate_len):
    heapq.heappush(max_pq, (-score_bbox_candidates[i]["scores"][0], i)) # i는 후보지의 index
print(max_pq)

occupied_ranges = {
    'x': [],
    'y': []
}

def check_overlap(bbox, occupied_ranges):
    x, y, w, h = bbox
    # 이진 검색으로 겹치는 구간 찾기
    x_overlap = any(start <= x+w and end >= x 
                   for start, end in occupied_ranges['x'])
    y_overlap = any(start <= y+h and end >= y 
                   for start, end in occupied_ranges['y'])
    return x_overlap and y_overlap

while max_pq:
    score, index = heapq.heappop(max_pq)
    print(f"score: {score}, index: {index}")
    cur_idx = score_bbox_candidates[index]["current"]
    
    bbox = score_bbox_candidates[index]["bbox"][cur_idx]
    x, y, w, h = bbox
    print(f"bbox: {bbox}")
    
    overlap = check_overlap(bbox, occupied_ranges)
    
    if overlap:
        score_bbox_candidates[index]["current"] += 1
        # 현재 후보지의 점수가 남아있는 경우.
        if score_bbox_candidates[index]["current"] < len(score_bbox_candidates[index]["scores"]):
            heapq.heappush(max_pq, (-score_bbox_candidates[index]["scores"][score_bbox_candidates[index]["current"]], index))
    else:
        occupied_ranges['x'].append((x, x+w))
        occupied_ranges['y'].append((y, y+h))
        print(f"occupied_ranges: {occupied_ranges}")

for item in score_bbox_candidates:
    print(f'현재 score: {item["scores"][item["current"]]}, 현재 bbox: {item["bbox"][item["current"]]}')
