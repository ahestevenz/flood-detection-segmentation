from typing import Tuple, Dict, List
import json

def load_conf(file:str,
              from_string:bool=False)->Dict:
    with open(file, 'r') as f:
        js = json.load(f)
        js = json.loads(js) if from_string else js
        return js

def save_conf(js_dict:Dict, file:str):
    dict_to_json = json.dumps(js_dict)
    with open(file, "w") as f:
        json.dump(dict_to_json, f) 