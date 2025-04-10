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
#sys.path.append('./src/')
#from src import myUNF
#from src import cloudy_class
from PIL import Image
import requests
import cv2
from tqdm import tqdm

def get_utc_timestamp(x: datetime):
    return int((x - datetime(1970, 1, 1)).total_seconds())


def convert_wgs_to_utm(lon: float, lat: float):
    """Based on lat and lng, return best utm epsg-code"""
    utm_band = str((np.floor((lon + 180) / 6) % 60) + 1)
    if len(utm_band) == 1:
        utm_band = "0" + utm_band
    if lat >= 0:
        epsg_code = "326" + utm_band
    else:
        epsg_code = "327" + utm_band
    return int(float(epsg_code))


def convert_epsg_geometry(geom, epsg):
    proj = pyproj.Transformer.from_crs(4326, epsg, always_xy=True)
    return ops.transform(proj.transform, geom)


def convert_epsg_geometry_inverse(geom, epsg):
    proj = pyproj.Transformer.from_crs(epsg, 4326, always_xy=True)
    return ops.transform(proj.transform, geom)


def get_skywatch_archive_results_from_db(location:geometry.Polygon, start_date:datetime, end_date:datetime):
    
    config = ov.get_sarccdb_credentials()
    schema = 'optical_ingestion'
    connection_string = f"host={config['host']} port={config['port']} user='{config['user']}' password='{config['password']}' dbname={config['database']} options='-c search_path={schema}'"
    with sqlsystem.connect(connection_string) as db:
        # Get a cursor object
        cursor = db.cursor()
        # Search archive images
        cursor.execute("SELECT * FROM products WHERE ((%(start_date)s <= start_time AND start_time <= %(end_date)s) OR (%(start_date)s <= end_time AND end_time <= %(end_date)s)) and source = %(source)s",
                       {'start_date': get_utc_timestamp(start_date),
                        'end_date': get_utc_timestamp(end_date),
                        'bounds': sqlsystem.Binary(location.wkb),
                        'source': 'SkySat'})
        output = cursor.fetchall()

    output_sorted = []

    keys_i = ['db_id', 'product_id', 'bounds', 'source', 'product_name', 'resolution', 'start_time', 'end_time',
              'preview_uri', 'thumbnail_uri', 'location_coverage_percentage', 'area_sq_km', 'cost',
              'result_cloud_cover_percentage', 'ingestion_time', 'ordered']
    for i in output:
        tmp_dict = {keys_i[j]: i[j] for j in range(len(i))}
        tmp_dict['bounds'] = wkb.loads(tmp_dict['bounds'], hex=True)
        if tmp_dict['bounds'].intersects(location):
            output_sorted.append(tmp_dict)

    return output_sorted



# Setup
customer = 'LSNed-2025'
corridor_width = 1000  # meters

customer_db = CustomerDatabase(ov.get_customerdb_credentials()['username'], ov.get_customerdb_credentials()['password'])
project = customer_db.get_project_by_name(customer)

config = ov.get_sarccdb_credentials()
with RegionsDb(config) as database:
    database_customer = database.get_regions_by_customer(customer)

# get corridor
ce_login = ov.get_project_server_credentials(project['name'])

client = cosmic_eye_client.connect('lsned.orbitaleye.nl')
client.login(ce_login['user'], ce_login['password'])
pipe_contours = client.getAllPipelines()

corridor_all = []
for pipeline in pipe_contours:
    points = wkb.loads(str(pipeline[3]), hex=True).coords
    pipe_line = geometry.LineString(np.array(points.xy).T)
    epsg_utm = convert_wgs_to_utm(pipe_line.centroid.x, pipe_line.centroid.y)
    corridor_utm = convert_epsg_geometry(pipe_line, epsg_utm).buffer(corridor_width)
    corridor = convert_epsg_geometry_inverse(corridor_utm, epsg_utm)
    corridor_all.append(corridor)

valid_corridor_all = [p for p in corridor_all if isinstance(p, geometry.Polygon)]
corridor_poly = geometry.MultiPolygon(valid_corridor_all).buffer(0.0)



def check_intersections(target_poly, poly_list):

    for poly in poly_list:
        if target_poly.intersects(poly):
            intersection = target_poly.intersection(poly)
            percent = (intersection.area / target_poly.area) * 100
            #print('percent', percent)
            if percent>10:
                return 2
    return 1

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


reporting_intervals = project['reporting_intervals']
keep_reporting_intervals = []

months = ['10', '11', '12', '01', '02', '03']
for r_int in reporting_intervals:
    middles = [date.split('-')[1] for date in r_int]
    for m in months:
        if any(m in s for s in middles):
            keep_reporting_intervals.append(r_int)


print(keep_reporting_intervals)



# get the previewsc
#print('project', project['reporting_intervals'][2])
#print('project', project['reporting_intervals'][8])


previews_bounds_list = []
keep_previews = []
for keep_date in keep_reporting_intervals:

    project_start = datetime.strptime(keep_date[0], '%Y-%m-%d')  # datetime.strptime(project['first_report_date'], '%Y-%m-%d')
    project_end = datetime.strptime(keep_date[1], '%Y-%m-%d')  
    previews = get_skywatch_archive_results_from_db(corridor_poly.envelope, project_start, project_end)
    # previews = previews.sort(key=lambda x: x['start_time'])

    shopping_cart = geometry.MultiPolygon()

    print('len', len(previews))

    ####################################################################################################################################
    creds = ov.get_sarccdb_credentials()

    for preview in previews:
    #    print(preview['bounds'])    
        if not previews_bounds_list:
            previews_bounds_list.append(preview['bounds']) 
        else:
            result = check_intersections(preview['bounds'], previews_bounds_list)
            if result==1:
                keep_previews.append(preview)


for _,preview in enumerate(tqdm(keep_previews)):
    image = get_image(preview['preview_uri'])
    image = Image.fromarray(image)
    image.save('./DATASET_previews/PWN/{}.png'.format(preview['db_id']))

print('finall', len(keep_previews))

