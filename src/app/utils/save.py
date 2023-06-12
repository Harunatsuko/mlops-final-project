import os
import json
from datetime import datetime
import boto3
from app.model.model import MODEL_FILENAME

def save_last_weights_to_s3():
    saved = True
    is_new = False
    AWS_ID_KEY = os.environ['AWS_ID_KEY']
    AWS_SECRET_KEY = os.environ['AWS_SECRET_KEY']
    try:
        session = boto3.session.Session()
        s3 = session.client(service_name='s3',
                            endpoint_url='https://storage.yandexcloud.net',
                            aws_access_key_id= AWS_ID_KEY,
                            aws_secret_access_key= AWS_SECRET_KEY)

        with open('./config.json', 'r', encoding='utf8') as c:
            config = json.load(c)
        
        model_weights_path = os.path.join(config['output_path'], MODEL_FILENAME)
        if os.path.exists(model_weights_path):
            now_date_str = datetime.now().strftime('%d-%m-%Y')
            s3.upload_file(Filename=model_weights_path,
                           Bucket=config['src_bucket'],
                           Key=MODEL_FILENAME+now_date_str)
            is_new = True
            os.remove(model_weights_path)

    except Exception as e:
        print(e)
        saved = False
    return saved, is_new