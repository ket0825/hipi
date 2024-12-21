from pymilvus import FieldSchema, CollectionSchema, DataType
from dataclasses import dataclass, field

class ObjSchema(CollectionSchema):
    def __init__(self, name:str = "obj", dim=60, **kwargs):
        self._name = name
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="shape_context", dtype=DataType.FLOAT16_VECTOR, dim=dim, description="histogram vector", nullable=False),
            FieldSchema(name="pca_comp_x", dtype=DataType.FLOAT, description="pca_component_[0][0]: X (for calculate rotation angle)", nullable=False),            
            FieldSchema(name="pca_comp_y", dtype=DataType.FLOAT, description="pca_component_[0][1]: Y (for calculate rotation angle)", nullable=False),
            FieldSchema(name="keypoint_center_x", dtype=DataType.FLOAT, description="Mean of keypoint x: X (for calculate translation)", nullable=False),            
            FieldSchema(name="keypoint_center_y", dtype=DataType.FLOAT, description="Mean of keypoint y : Y (for calculate translation)", nullable=False),            
            FieldSchema(name="width", dtype=DataType.INT16, nullable=False, description="img width"),              
            FieldSchema(name="height", dtype=DataType.INT16, nullable=False, description="img height"),              
            FieldSchema(name="radius", dtype=DataType.FLOAT, description="radius", nullable=False),
            FieldSchema(name="n_features", dtype=DataType.INT16, nullable=False, description="number of features"),
            FieldSchema(name="name", dtype=DataType.VARCHAR, nullable=True, description="image name", max_length=256),
            FieldSchema(name="img_path", dtype=DataType.VARCHAR, nullable=True, description="png image path in minIO", max_length=256),            
        ]
        super().__init__(fields=fields, **kwargs)        

    def add_field(self, field_name: str, data_type: DataType, **field_kwargs):
        # Dynamic field addition before collection creation
        new_field = FieldSchema(name=field_name, dtype=data_type, **field_kwargs)
        self.fields.append(new_field)
        return new_field        
    
    @property
    def name(self):
        return self._name


@dataclass
class ObjIdx:
    field_name:str ="shape_context"
    metric_type:str = "COSINE"
    index_type:str = "IVF_FLAT"
    index_name:str = "obj_index"
    params:dict = field(default_factory=lambda: { "nlist": 128 })
        
    def __repr__(self):
        return f"{self.field_name}, {self.metric_type}, {self.index_type}, {self.index_name}, {self.params}"