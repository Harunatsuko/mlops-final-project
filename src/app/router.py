import os
import json
from fastapi import APIRouter
from starlette import status
from starlette.responses import Response

from app.app_models import Config, ObjectsList
from app.utils.train import train_in_process
from app.utils.load import upload_from_s3
from app.utils.save import save_last_weights_to_s3

PATH_A = 'flowers_imitation'
PATH_B = 'flowers_photo'

api_router = APIRouter()

@api_router.post("/load_data/")
def load_data(objects_list: ObjectsList):
    print(objects_list.objects_list)
    updated = upload_from_s3(objects_list.objects_list)
    if updated:
        return Response(status_code=status.HTTP_200_OK)
    else:
        return Response(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_router.get("/train_model/{num_epochs}")
def train_model(num_epochs: int = 20):
    with open('./config.json', 'r', encoding='utf8') as c:
        config = json.load(c)
    train_in_process(os.path.join(config['data_path'],PATH_A),
                os.path.join(config['data_path'],PATH_B),
                config['output_path'],
                num_epochs)
    return Response('Train process is started',
                    media_type='text/plain',
                    status_code=status.HTTP_200_OK)

@api_router.post("/set_config/")
def set_config(config: Config):
    if os.path.exists(config.data_path) and \
    os.path.exists(config.output_path):
        config_file = {'src_bucket': config.src_bucket, 
                       'data_path': config.data_path,
                       'output_path': config.output_path}
        with open('./config.json', 'w', encoding='utf8') as c:
            json.dump(config_file, c)
        res = status.HTTP_200_OK
    else:
        res = status.HTTP_400_BAD_REQUEST
    return Response(status_code=res)

@api_router.get("/save_model/")
def save_model():
    saved = save_last_weights_to_s3()
    if saved:
        return Response(status_code=status.HTTP_200_OK)
    else:
        return Response(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)
