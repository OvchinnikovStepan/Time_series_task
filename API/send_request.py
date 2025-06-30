from pydantic import BaseModel
import json
import requests

#Отправил

def send_model_request(payload: BaseModel):
    with open('config.json', 'r', encoding='utf-8') as f:
        config = json.load(f)
        url = config['url']

        headers = {'Content-Type': 'application/json'}
        response = requests.post(url, json=payload, headers=headers)

        return response


