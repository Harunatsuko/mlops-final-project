from pydantic import BaseModel

class Config(BaseModel):
    path_A: str
    path_B: str
    output_path: str