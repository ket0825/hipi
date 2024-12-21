import os

from dotenv import load_dotenv
from pymilvus import MilvusClient, Collection
from milvus.utils.create_utils import create_db, create_collections, create_index
from milvus.collection.obj_ref import ObjRef
from milvus.collection.obj import Obj
from milvus.collection_schema.obj_schema import ObjSchema, ObjIdx
from milvus.collection_schema.obj_ref_schema import ObjRefSchema

load_dotenv(".env")
def init_db(client: MilvusClient) -> tuple[str, str]:
    db_name = os.getenv("MILVUS_DB_NAME")    
    create_db(client, db_name)
    obj_schema = ObjSchema()
    obj_ref_schema = ObjRefSchema()
    create_collections(client, obj_schema.name, obj_schema)
    create_collections(client, obj_ref_schema.name, obj_ref_schema)
    create_index(client, obj_schema.name, [ObjIdx()])
    client.load_collection(obj_schema.name)
    # client.load_collection(obj_ref_schema.name) # No need to load obj_ref collection. No vector search required
    return obj_schema.name, obj_ref_schema.name
        
    
if __name__ == "__main__":
    db_host = os.getenv("MILVUS_HOST")
    db_port = os.getenv("MILVUS_PORT")
    addr = f"tcp://{db_host}:{db_port}"
    
    client = MilvusClient(uri=addr)
    init_db(client)