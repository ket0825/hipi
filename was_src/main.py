# main.py
import os

from fastapi import FastAPI
from api.v1.endpoints import img
from milvus.utils.connection_pools import MilvusConnectionPool
from dotenv import load_dotenv
# from init_db import init_db

load_dotenv(".env")

from contextlib import contextmanager
from fastapi import FastAPI
import os
import signal
import sys

@contextmanager
def fastapi_milvus_lifecycle():
    """
    FastAPI 앱과 Milvus 연결의 생명주기를 관리하는 context manager
    
    WARNING: vscode debug 모드인 경우 정상적으로 작동하지 않음.
    """
    def cleanup():
        print("[CLEANUP] Milvus connection pools...")
        if MilvusConnectionPool.__instance is None:
            return
        while not MilvusConnectionPool.__instance.empty():
            client = MilvusConnectionPool.__instance.get()
            client.close()

    def signal_handler(signum, frame):
        print(f"\nReceived signal {signum}")
        cleanup()
        sys.exit(0)

    def create_app() -> FastAPI:
        # Milvus 연결 설정
        addr = f"tcp://{os.environ['MILVUS_HOST']}:{os.environ['MILVUS_PORT']}"
        MilvusConnectionPool.set_pool(
            db_name=os.environ['MILVUS_DB_NAME'],
            uri=addr,
            pool_size=1
        )
        
        # FastAPI 앱 생성
        app = FastAPI()
        app.include_router(img.router, prefix="/api/v1/img")
        @app.get("/health")
        async def read_root():
            return {"Hello": "World"}
        
        return app

    try:
        # 시그널 핸들러 등록
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # FastAPI 앱 생성
        app = create_app()
        
        yield app  # 앱 인스턴스 반환

    except Exception as e:
        print(f"Error during app creation: {e}")
        raise
    finally:
        cleanup()

if __name__ == "__main__":
    import uvicorn
    with fastapi_milvus_lifecycle() as app:
        uvicorn.run(app, host="0.0.0.0", port=8080)