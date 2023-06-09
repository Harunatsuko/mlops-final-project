from pydantic import BaseModel

class Config(BaseModel):
    src_bucket: str
    data_path: str
    output_path: str

class ObjectsList(BaseModel):
    objects_list: list = []