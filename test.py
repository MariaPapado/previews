from fastapi import FastAPI
import uvicorn
import cv2
from pydantic import BaseModel
from pimsys.regions.RegionsDb import RegionsDb
import os
import torch
import cv2
import json
import base64
import numpy as np
from shapely import geometry
import sys
sys.path.append('./src/')
import myUNF
import rasterio
import requests
from shapely import wkt

config_db = {
        "host": "sar-ccd-db.orbitaleye.nl",
        "port": 5433,
        "user": "postgres",
        "password": "sarccd-db",
        "database": "sarccd2"
    }


addr = 'http://0.0.0.0:8002'
test_url = addr + '/predict'
#test_url = 'http://cloud-detection.stg.orbitaleye-dev.nl/predict'


customer = 'PTT-2024'
domain = 'ptt.orbitaleye.nl'
corridor_width = 1000  # meters

date_start = '2025-01-10'
date_end = '2025-03-30'

response = requests.post(test_url, json={'customer':customer, 'domain': domain, 'corridor_width': corridor_width, 'date_start': date_start, 'date_end':date_end})


if response.ok:
    print('ok')
    response_result = json.loads(response.text)
    cloud_polygon = wkt.loads(response_result['result']['noncloudy_polygon']) 
    print(cloud_polygon)
#    response_result_data = base64.b64decode(response_result['result'])
#    result = np.frombuffer(response_result_data,dtype=np.uint8)
#    print('rrr', result.shape, np.unique(result))
#    res = result.reshape(image_before.shape[:2])
else:
    print('no')

def save_geojson(filename, target_poly):
    geojson = {'type': 'FeatureCollection', 'name': 'Shopping_cart', 'csr': {'type': 'name', 'properties': {'name': 'urn:ogc:def:crs:OGC:1.3:CRS84'}}, 'features': []}
    tmp_i = {'type': 'Feature', 'properties': {'Name': str(1)},
            'geometry': geometry.mapping(target_poly)}
    geojson['features'].append(tmp_i)
    json.dump(geojson, open(filename, 'w'))

save_geojson('{}.json'.format('check'), cloud_polygon)

#cv2.imwrite('res.png', res)
