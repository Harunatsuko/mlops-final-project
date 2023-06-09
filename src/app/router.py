import json
from fastapi import APIRouter
from starlette import status
from starlette.responses import Response

from app_models import Config
from train import train_in_process

api_router = APIRouter()

@api_router.get("/load_data/")
def load_data():
    updated = load_data(body, chat_id)
    if updated:
        return Response(status_code=status.HTTP_200_OK)
    else:
        return Response(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_router.get("/train_model/{num_epochs}")
def train_model(num_epochs: int = 20):
    with open('./config.json', 'r', encoding='utf8') as c:
        config = json.load(c)
    train_in_process(config['path_A'],
                config['path_B'],
                config['output_path'],
                num_epochs)
    return Response('Train process is started',
                    media_type='text/plain',
                    status_code=status.HTTP_200_OK)

@api_router.post("/set_config/")
def set_config(config: Config):
    if os.path.exists(config.path_A) and \
    os.path.exists(aconfigrgs.path_B) and \
    os.path.exists(config.output_path):
        config_file = {'path_A': config.path_A, 
                       'path_B': config.path_B,
                       'output_path': config.output_path}
        with open('./config.json', 'w', encoding='utf8') as c:
            json.dump(config_file)
        status = status.HTTP_200_OK
    else:
        status = status.HTTP_400_BAD_REQUEST
    return Response(status_code=status)