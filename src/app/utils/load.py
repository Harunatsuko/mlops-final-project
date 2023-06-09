import os
import json
import boto3

def upload_from_s3(objects_list):
    updated = True
    try:
        session = boto3.session.Session()
        s3 = session.client(service_name='s3',
                            endpoint_url='https://storage.yandexcloud.net')

        with open('./config.json', 'r', encoding='utf8') as c:
            config = json.load(c)

        for obj in objects_list:
            if '/' in obj:
                folders = '/'.join(obj.split('/')[0:-1])
                filename = obj.split('/')[-1]

            if not os.path.exists(os.path.join(config['data_path'], folders)):
                curr_folder = config['data_path']
                for folder in folders.split('/'):
                    curr_folder = os.path.join(curr_folder, folder)
                    os.mkdir(curr_folder)

            out_path = os.path.join(config['data_path'], folders, filename)
            s3.download_file(config['src_bucket'], obj, out_path)
    except Exception as e:
        print(e)
        updated = False
    return updated