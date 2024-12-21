from pymilvus import Collection
from milvus.collection_schema.obj_schema import ObjSchema

class Obj(Collection):
    def __init__(self, schema=None, **kwargs):
        if schema is None:
            schema = ObjSchema()
        super().__init__(name="obj", schema=schema, **kwargs)