import os
from queue import Queue
from pymilvus import MilvusClient
from contextlib import contextmanager


class MilvusConnectionPool:
    """
    Use this class for global(data area in memory) for thread safe connection pool. 
    """
    __instance = None
    
    @classmethod
    def set_pool(cls, db_name:str, uri:str, pool_size=5):
        if MilvusConnectionPool.__instance is None:
            print(f"Creating Milvus connection pool with size: {pool_size}" )
            cls.__instance = Queue(maxsize=pool_size)
            
            for _ in range(pool_size):
                client = MilvusClient(db_name=db_name, uri=uri)
                cls.__instance.put(client)
                                        
    
    @classmethod
    def get_client(cls) -> MilvusClient:
        if cls.__instance is None:
            raise ValueError("Connection pool is not initialized.")
        return cls.__instance.get()        
        
    @classmethod
    def release_client(cls, client: MilvusClient) -> None:
        cls.__instance.put(client)        
    
    def __del__(self):
        print("Destructor called")
        if self.__instance is None:
            return
        while not self.__instance.empty():
            client = self.__instance.get()
            client.close()
        
@contextmanager
def get_milvus_client():
    client = MilvusConnectionPool.get_client()
    try:
        yield client
    finally:
        MilvusConnectionPool.release_client(client)

