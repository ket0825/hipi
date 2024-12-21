from dotenv import load_dotenv
from pymilvus import CollectionSchema, MilvusClient, MilvusException

load_dotenv(".env")
def create_db(client: MilvusClient, db_name: str) -> None:
    # db_host = os.getenv("MILVUS_HOST")
    # db_port = os.getenv("MILVUS_PORT")
    # connections.connect(host=db_host, port=db_port)    
    try:        
        client.create_database(db_name=db_name)
        # db.create_database(db_name=db_name)        
    except MilvusException as e:
        if e.code == 65535:
            print("DB already exists.")
        else:
            print(f"DB NAME: {db_name}, Error in db creation: {e}")
    finally:
        client.using_database(db_name)
        print(f"current DB: {db_name}")
    

def create_collections(client: MilvusClient, collection_name:str, schema:CollectionSchema) -> None:     
    if not client.has_collection(collection_name):    
        client.create_collection(
            collection_name=collection_name,
            schema=schema,                        
        )
        print(f"[CREATED] Collection *{collection_name}* created.")
    else:
        print(f"Collection *{collection_name}* already exists.")
        
    print(f"total_collections: {client.list_collections()}")
    
def create_index(client: MilvusClient, collection_name:str, index_list:list[dict]) -> None:    
    # if client.has_collection(collection_name):
        # No need to drop index
        # if client.get_load_state(collection_name) == 3:
        #     """
        #     NotExist = 0
        #     NotLoad = 1
        #     Loading = 2
        #     Loaded = 3
        #     """
        #     client.release_collection(collection_name)
        #     print(f"Collection loaded: {collection_name}")
        # # Drop all indexes
        # client.drop_index(collection_name=collection_name, index_name="all", sync=True)
        # print(f"Index dropped for collection: {collection_name}")                
        # print(f"Collection exists: {collection_name}")        
    try:
        indexes = client.list_indexes(collection_name=collection_name)
        index_params = MilvusClient.prepare_index_params()    
        for index in index_list:
            if index.field_name not in indexes:
                index_params.add_index(
                    field_name=index.field_name,
                    index_type=index.index_type,
                    metric_type=index.metric_type,
                    params=index.params
                )
        
        if not index_params._indexes:
            print(f"Index already exists for collection: {collection_name}")
            return
        
        client.create_index(
            collection_name=collection_name,            
            index_params=index_params,
            sync=True,            
        )
        print(f"Index created for collection: {collection_name}")
    except Exception as e:
        print(f"Error in creating index: {e}")    
        