from pymilvus import Collection
from milvus.collection_schema.obj_ref_schema import ObjRefSchema

class ObjRef(Collection):
    def __init__(self, schema=None, **kwargs):
        if schema is None:
            schema = ObjRefSchema()
        return super().__init__(name="obj_ref", schema=schema, **kwargs)
    

        