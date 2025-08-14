import orbital_vault as ov
from CustomerDatabase import CustomerDatabase
from pimsys.regions.RegionsDb import RegionsDb
import pyproj
from shapely import ops, geometry, wkb
import numpy as np
import cosmic_eye_client
from datetime import datetime, timedelta
import psycopg2 as sqlsystem
from shapely.wkt import loads
import matplotlib.pyplot as plt
import os
import rasterio
import torch
import sys
sys.path.append('./src/')
from src import myUNF
from src import cloudy_class
from src import funcs
from unet_model import *
from unet_parts import *
import requests
import cv2
from PIL import Image

def get_image(image_url):
    resp = requests.get(image_url)
    try:
        image = np.asarray(bytearray(resp.content))
        #image = image.astype(np.int16)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        image = image[:, :, [2, 1, 0]] 

    except:
        print(resp)
        
    return image


def extract_clouds(customer, domain, corridor_width, project_start, project_end):
    
    config = ov.get_sarccdb_credentials()
    customer_db = CustomerDatabase(ov.get_customerdb_credentials()['username'], ov.get_customerdb_credentials()['password'])
    project = customer_db.get_project_by_name(customer)


#    with RegionsDb(config) as database:
#        database_customer = database.get_regions_by_customer(customer)

    # get corridor
    ce_login = ov.get_project_server_credentials(project['name'])

    client = cosmic_eye_client.connect(domain)
    client.login(ce_login['user'], ce_login['password'])
    pipe_contours = client.getAllPipelines()

    corridor_all = []
    for pipeline in pipe_contours:
        points = wkb.loads(str(pipeline[3]), hex=True).coords
        pipe_line = geometry.LineString(np.array(points.xy).T)
        epsg_utm = funcs.convert_wgs_to_utm(pipe_line.centroid.x, pipe_line.centroid.y)
        corridor_utm = funcs.convert_epsg_geometry(pipe_line, epsg_utm).buffer(corridor_width)
        corridor = funcs.convert_epsg_geometry_inverse(corridor_utm, epsg_utm)
        corridor_all.append(corridor)

    valid_corridor_all = [p for p in corridor_all if isinstance(p, geometry.Polygon)]
    corridor_poly = geometry.MultiPolygon(valid_corridor_all).buffer(0.0)


    # get the previews

    previews = funcs.get_skywatch_archive_results_from_db(corridor_poly.envelope, project_start, project_end)
    # previews = previews.sort(key=lambda x: x['start_time'])


    print('len', len(previews))


    ####################################################################################################################################
    creds = ov.get_sarccdb_credentials()


    
    for preview in previews:
    #if preview['db_id']==187389:
        print(preview['db_id'])
        image = get_image(preview['preview_uri'])    
        print(image.shape)
        image = image.astype(np.uint8)
        image = Image.fromarray(image)
        image.save('./DATASET_previews_TEST/gni/{}.png'.format(preview['db_id']))




if __name__ == "__main__":
    # Setup
    customer = 'GNI-2024'
    domain = 'gni.orbitaleye.nl'
    corridor_width = 1000  # meters

    date_start = '2025-05-01'
    date_end = '2025-08-13'

#    project_start = datetime.strptime(project['reporting_intervals'][26][0], '%Y-%m-%d')  # datetime.strptime(project['first_report_date'], '%Y-%m-%d')
#    project_end = datetime.strptime(project['reporting_intervals'][26][1], '%Y-%m-%d')  

    project_start = datetime.strptime(date_start, '%Y-%m-%d')  # datetime.strptime(project['first_report_date'], '%Y-%m-%d')
    project_end = datetime.strptime(date_end, '%Y-%m-%d') 

    extract_clouds(customer, domain, corridor_width, project_start, project_end)



#import json
#geojson = {'type': 'FeatureCollection', 'name': 'Shopping_cart', 'csr': {'type': 'name', 'properties': {'name': 'urn:ogc:def:crs:OGC:1.3:CRS84'}}, 'features': []}
#tmp_i = {'type': 'Feature', 'properties': {'Name': str(1)},
#        'geometry': geometry.mapping(shopping_cart)}
#geojson['features'].append(tmp_i)
#json.dump(geojson, open('./check_shopcart/shopcart.geojson', 'w'))
###################################################################################################################################################



    
