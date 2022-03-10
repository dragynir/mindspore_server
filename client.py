import numpy as np
import requests

if __name__ == '__main__':


    params = {'param0': 'param0', 'param1': 'param1'}
    arr = np.ones((1, 6))
    data = {'params': params, 'arr': arr.tolist()}
    url = 'http://127.0.0.1:9097/predict'
    response = requests.post(url, json=data)