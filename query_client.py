import urllib3
import os
from urllib3.filepost import encode_multipart_formdata
import mimetypes

def query_server(dirpath: os.PathLike, output_dir: os.PathLike):
    url = 'http://localhost:8080/api/v1/img/query'
    http = urllib3.PoolManager()
    
    # output 디렉토리가 없으면 생성
    os.makedirs(output_dir, exist_ok=True)
    
    for file in os.listdir(dirpath):
        img_path = os.path.join(dirpath, file)
        if not os.path.isfile(img_path):
            print(f"Warning: {img_path} is not a file. Skipping...")
            continue
            
        try:
            with open(img_path, 'rb') as img:
                # 파일 데이터 준비
                files = {'file': (file, img.read())}
                # multipart/form-data 인코딩
                fields, content_type = encode_multipart_formdata(files)
                
                # HTTP 요청 전송
                response = http.request(
                    'POST',
                    url,
                    body=fields,
                    headers={'Content-Type': content_type}
                )
                
                if response.status == 200:
                    print(f"Successfully queried {file}")
                    
                    # 응답 헤더에서 content-type 확인
                    response_content_type = response.headers.get('content-type', '')
                    # 파일 확장자 결정
                    ext = mimetypes.guess_extension(response_content_type) or '.jpg'
                    
                    # 원본 파일명에서 확장자를 제거하고 새 확장자 추가
                    base_name = os.path.splitext(file)[0]
                    output_filename = f"{base_name}{ext}"
                    
                    # 결과 이미지 저장
                    output_path = os.path.join(output_dir, output_filename)
                    with open(output_path, 'wb') as output:
                        output.write(response.data)
                    print(f"Saved result to {output_path}")
                else:
                    print(f"Failed to query {file}. Status: {response.status}")
                    print(f"Response: {response.data}")
                    
        except FileNotFoundError:
            print(f"Error: Could not open {img_path}")
        except Exception as e:
            print(f"Error processing {file}: {str(e)}")

if __name__ == "__main__":
    query_server("./query_input", "./query_output")