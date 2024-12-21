import urllib3
from minio import Minio
from minio.helpers import ObjectWriteResult
from dotenv import load_dotenv
import os
import io
import cv2
from typing import Any

load_dotenv(".env")

def get_client() -> Minio:
    try:
        minio_host = os.getenv("MINIO_HOST")
        minio_port = os.getenv("MINIO_PORT")        
        minio_addr = f"{minio_host}:{minio_port}"
        minio_access_key = os.getenv("MINIO_ROOT_USER")
        minio_secret_key = os.getenv("MINIO_ROOT_PASSWORD")
        
        print(f"Attempting to connect to MinIO at {minio_addr}")
        
        http_client = urllib3.PoolManager(
                num_pools=1,
                timeout=urllib3.Timeout(
                    connect=10.0,
                    read=10.0
                ),                
                retries=urllib3.Retry(
                    total=3,
                    backoff_factor=0.2,
                    status_forcelist=[500, 502, 503, 504]
                )
            )
        
        client = Minio(    
            endpoint=minio_addr,
            access_key=minio_access_key,
            secret_key=minio_secret_key,
            secure=False,
            http_client=http_client
        )
        
        # Test connection
        buckets =  client.list_buckets()
        print(f"BUCKETS len: {len(buckets)}")
        for bucket in buckets:
            print(f"BUCKET: {bucket.name}")
        return client
    except Exception as e:
        print(f"Failed to connect to MinIO: {e}")
        raise e

def create_bucket(client: Minio, bucket_name: str):
    if not client.bucket_exists(bucket_name):
        client.make_bucket(bucket_name)
    else:
        print(f"Bucket {bucket_name} already exists.")

def get_bucket(client: Minio, bucket_name: str) -> str:
    for bucket in client.list_buckets():
        if bucket.name == bucket_name:
            return bucket

# get은 이미지 파일을 메모리에 가져오는 것임.
# production에서는 여기에 과부하가 걸릴 수 있음. IO이기에.
def get_img_object(client: Minio, bucket_name: str, object_name: str) -> ObjectWriteResult:
    res = client.get_object(
        bucket_name=bucket_name,
        object_name=object_name,    
    )
    return res.read()

# 메모리 저장 방식이 필요.
def put_object(client: Minio, bucket_name: str, object_name: str, data: Any) -> ObjectWriteResult:
    _, img_encoded = cv2.imencode('.jpg', data) # or '.png'
    # 바이트 배열로 변환
    img_bytes = img_encoded.tobytes()
    
    resp = client.put_object(
        bucket_name=bucket_name,
        object_name=object_name,
        data=io.BytesIO(img_bytes),
        length=len(img_bytes)
    )
    return resp

if __name__ == "__main__":    
    minio_client = get_client()
    create_bucket(minio_client, "objects")
    # put_object(minio_client, "objects", "test.jpg", "test.jpg")
    # get_object(minio_client, "objects", "test.jpg", "./zzz/test.jpg")
    
    