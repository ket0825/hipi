import urllib3
import os
from urllib3.exceptions import HTTPError
import mimetypes

def insert_images(dirname: os.PathLike, base_url: str = "http://localhost:8080"):
    """
    디렉토리 내의 모든 이미지를 서버에 업로드합니다.
    
    Args:
        dirname: 이미지가 있는 디렉토리 경로
        base_url: 서버 기본 URL
    """
    # urllib3 풀 매니저 생성
    http = urllib3.PoolManager()
    endpoint = f"{base_url}/api/v1/img/"  # 끝에 슬래시 추가
    
    success_count = 0
    fail_count = 0
    
    for file in os.listdir(dirname):
        img_path = os.path.join(dirname, file)
        
        if not os.path.isfile(img_path):
            print(f"Warning: {img_path} is not a file. Skipping...")
            continue
            
        try:
            # 파일의 MIME 타입 확인
            content_type, _ = mimetypes.guess_type(img_path)
            if not content_type or content_type not in ["image/jpeg", "image/jpg", "image/png"]:
                print(f"Warning: {file} is not a supported image file. Skipping...")
                continue
                
            # 이미지 파일 읽기
            with open(img_path, 'rb') as img:
                image_data = img.read()
                
                # multipart/form-data 형식으로 요청 보내기
                response = http.request(
                    'POST',
                    endpoint,
                    fields={
                        'file': (file, image_data, content_type)
                    },
                    timeout=30.0
                )
                
                if response.status == 200:
                    print(f"Successfully uploaded {file}")
                    success_count += 1
                else:
                    print(f"Failed to upload {file}. Status: {response.status}")
                    print(f"Response: {response.data.decode('utf-8')}")
                    fail_count += 1
                    
        except FileNotFoundError:
            print(f"Error: Could not open {img_path}")
            fail_count += 1
        except HTTPError as e:
            print(f"HTTP Error occurred while uploading {file}: {e}")
            fail_count += 1
        except Exception as e:
            print(f"Unexpected error occurred while uploading {file}: {e}")
            fail_count += 1
            
    print(f"\nUpload completed.")
    print(f"Success: {success_count}")
    print(f"Failed: {fail_count}")
    
if __name__ == '__main__':
    insert_images("./obj_rmbg")