# api/v1/endpoints/img.py
from fastapi import (
    APIRouter, 
    UploadFile, 
    File, 
    HTTPException,
    Response,
)

import io
from milvus.utils.minio_utils import get_client, put_object, get_img_object
from img.cv_utils import preprocess_insert_img, preprocess_query_img, t_nn_score, place_non_overlapping_matches, localize_object
from milvus.collection.obj import Obj
from img.object_img import ObjectImg
from milvus.utils.connection_pools import get_milvus_client
from img.best_match import BestMatch
import traceback
import numpy as np

minio_client = get_client() # 계속 사용할 것이므로 전역으로 선언
router = APIRouter()

@router.post("/")
def insert_image(file: UploadFile = File(...)):
    content_type = file.content_type
    contents = file.file.read()    
    try:
        if content_type in ["image/jpeg", "image/jpg"]:
            # JPEG 처리 로직
            print("JPEG!")
            # processed_contents = process_jpeg(contents)
            # save_path = f"processed/{file.filename}"
            return {"message": "jpg"}
        elif content_type == "image/png":
            # PNG는 그대로 저장
            print("PNG!")
            # 이후 전처리는 새로운 서버에서 처리할 수도 있음.        
            obj = preprocess_insert_img(contents)
            res = put_object(minio_client, "objects", file.filename, 
                            #  io.BytesIO(obj.img_origin)
                             data=obj.img
                             )
            print(f"UPLOADED {res.bucket_name}, {res.object_name}, {res.etag}")
            with get_milvus_client() as milvus_client:
                # obj_collection = Obj()                
                res = milvus_client.insert(
                    collection_name="obj", # obj_collection.name
                    data={
                        "shape_context": obj.get_histogram().astype("float16"),
                        "n_features": obj.get_kp_length(),
                        "pca_comp_x": obj.pca.components_[0][0].astype("float16"),
                        "pca_comp_y": obj.pca.components_[0][1].astype("float16"),                        
                        "keypoint_center_x": obj.get_kp_center()[0].astype("float16"), # "keypoint_center_x"
                        "keypoint_center_y": obj.get_kp_center()[1].astype("float16"), # "keypoint_center_y"
                        "width": obj.img.shape[1],
                        "height": obj.img.shape[0],
                        "name": file.filename,
                        "img_path": f"{res.bucket_name}/{res.object_name}",
                        "radius": obj.radius,
                    }
                )
                milvus_client.flush(collection_name="obj")
            print(f"INSERTED {res}")                                                                
            return {"message": "png"}
        else:
            raise HTTPException(status_code=400, detail="Unsupported image type")
    except:
        traceback.print_exc()
        e = traceback.format_exc()
        raise HTTPException(status_code=500, detail="Internal Server Error")    
        
    
@router.post("/bulk")
async def insert_bulk_images():
    return {"message": "Insert bulk images"}    

@router.post("/query")
def query_image(file: UploadFile = File(...)):
    content_type = file.content_type
    contents = file.file.read()    
    try:
        if content_type in ["image/jpeg", "image/jpg", "image/png"]:            
            print("JPEG or PNG!")                        
            # 이후 전처리는 새로운 서버에서 처리할 수도 있음.        
            back = preprocess_query_img(contents)
            histograms = back.get_histograms().astype("float16") # M * 60 (divisions * radius)
            # TODO: 뒤집은 histograms도 계샨해서 더 큰 값을 넣어줘야 함. object를 뒤집는 것은 커질수록 불가능.
            # 아니먄 이미지의 rotate, 이미지의 거울상, 거울상의 rotated까지 넣어주자.
            with get_milvus_client() as milvus_client:
                milvus_client.load_collection("obj")                                
                # 검색 파라미터 설정
                search_params = {
                    "metric_type": "COSINE",  # ObjIdx에 정의된 metric_type
                    "params": {"nprobe": 16}  # nprobe는 검색할 클러스터 수
                }

                # 검색 실행
                best_matches = milvus_client.search(
                    collection_name="obj", 
                    data=histograms,
                    anns_field="shape_context",
                    search_params=search_params,           # 검색 파라미터
                    limit=1,                       # 각 벡터당 반환할 최대 결과 수
                    output_fields=["name", "img_path", 
                                   "shape_context", "radius", 
                                   "pca_comp_x", "pca_comp_y", 
                                   "keypoint_center_x", "keypoint_center_y",
                                    "width", "height"
                                   ]  # 결과와 함께 반환할 필드들
                )              
            # 필요없는 속성 제거.
            best_matches.__delattr__("extra") # cost 값 존재.
            
            best_matches = [BestMatch(
                id=match[0]["id"],
                distance=match[0]["distance"],
                name=match[0]["entity"]["name"],
                img_path=match[0]["entity"]["img_path"],
                width=match[0]["entity"]["width"],
                height=match[0]["entity"]["height"],
                histogram=np.frombuffer(match[0]["entity"]["shape_context"], dtype=np.float16).astype(np.int8),
                pca_comp=np.array([match[0]["entity"]["pca_comp_x"], match[0]["entity"]["pca_comp_y"]], dtype=np.float16),
                kp_center=np.array([match[0]["entity"]["keypoint_center_x"], match[0]["entity"]["keypoint_center_y"]], dtype=np.float16),
                radius=match[0]["entity"]["radius"]
            ) for match in best_matches]            
                        
            print(f"[SEARCH] best_matches length: {len(best_matches)}")
            
            best_matches = t_nn_score(back, best_matches, scale_weight=0.3)
            candidates = place_non_overlapping_matches(back, best_matches)
            # 추가로, 일정 점수 이하는 추출하지 않게도 할 수 있음.
            # candidates = candidates[:2]
            obj_imgs = []
            for back_idx in candidates:                
                bucket_name, object_name = best_matches[back_idx].img_path.split("/")        
                img_byte = get_img_object(minio_client, bucket_name, object_name)
                obj_imgs.append(ObjectImg.from_bytes(img_byte))
            # 실제 background에 object를 배치하는 로직.
            result_bytes = localize_object(back, best_matches, obj_imgs, candidates)
            # TODO: 결과 이미지를 반환 필요
            # return Response(content=result_bytes, media_type="image/png")
            return {"message": "png"}
        else:
            raise HTTPException(status_code=400, detail="Unsupported image type")
    except:
        traceback.print_exc()
        e = traceback.format_exc()
        raise HTTPException(status_code=500, detail="Internal Server Error")        