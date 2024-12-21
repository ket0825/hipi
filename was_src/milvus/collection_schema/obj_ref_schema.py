from pymilvus import FieldSchema, CollectionSchema, DataType

class ObjRefSchema(CollectionSchema):
    def __init__(self, name:str = "obj_ref", **kwargs):
        self._name = name
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False, description="same as obj collection id"),
            FieldSchema(name="ref", dtype=DataType.INT64, nullable=False, description="ID with reference count"),
            FieldSchema(name="temp_vector", dtype=DataType.FLOAT_VECTOR, nullable=True, dim=2, description="temp vector for nothing"),
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