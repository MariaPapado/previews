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
from PIL import Image
import requests
import cv2
from tqdm import tqdm
import random
from geopy.distance import geodesic
import shutil

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

#config = ov.get_sarccdb_credentials()
#with RegionsDb(config) as database:
#    database_customer = database.get_regions_by_customer(customer)

# get corridor

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

def load_image(image,bounds=None):

    region_width = geodesic((bounds[1], bounds[0]), (bounds[1], bounds[2])).meters
    region_height = geodesic((bounds[1], bounds[0]), (bounds[3], bounds[0])).meters
    height, width = image.shape[:2]
    rx, ry = region_height / height, region_width / width # how many meters per pixel
    #mx, my = rx/target_res, ry/target_res

    #nx, ny = int(np.round(mx*height)), int(np.round(my*width))
    #image_save = image.astype(np.uint8)
    #image_save = Image.fromarray(image_save)
    #image_save.save('./previews/{}_{}_{}.png'.format(pid, ny, nx))
    #image = cv2.resize(image, (ny,nx), interpolation=cv2.INTER_NEAREST)            

    return rx, ry

def find_corridor_poly(customer, domain, corridor_width, customer_db):
    project = customer_db.get_project_by_name(customer)
    ce_login = ov.get_project_server_credentials(project['name'])

    client = cosmic_eye_client.connect(domain)  ##################################################################################
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

    return corridor_poly

def find_reporting_intervals(customer):
    project = customer_db.get_project_by_name(customer)
    print('proj', project)
    reporting_intervals = project['reporting_intervals']
    keep_reporting_intervals = []

#    months = ['09', '10', '11', '12']


    months = ['09', '10', '11', '12', '01', '02', '03', '04']
#    months = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
    for r_int in reporting_intervals:
        print(r_int)
        middles = [date.split('-')[1] for date in r_int]
        for m in months:
            if any(m in s for s in middles):
                keep_reporting_intervals.append(r_int)

    return keep_reporting_intervals

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


customers = ['PTT-2024', 'ENAGAS-2025-processing', 'BHE-2024', 'GNI-2024', 'Slovnaft-ext-2024', 'NGC-2023',
             'Waternet-2024', 'TC-Energy-2025-processing']
domains = ['ptt.orbitaleye.nl', 'enagas.orbitaleye.nl', 'bhegts.orbitaleye.nl', 'gni.orbitaleye.nl',
           'slovnaft.orbitaleye.nl', 'ngc.orbitaleye.nl', 'waternet.orbitaleye.nl', 'tcenergy.orbitaleye.nl']

customer = 'PTT-2024'    #########################################################################################################
domain = 'ptt.orbitaleye.nl'
keep_reporting_intervals = []
cnt_limit = 50
#1756


#customer = 'BHE-2024'  
#domain = 'bhegts.orbitaleye.nl'
#keep_reporting_intervals = []
#cnt_limit = 20
#494

#customer = 'GNI-2024'  
#domain = 'gni.orbitaleye.nl'
#keep_reporting_intervals = []
#cnt_limit = 20
#115

#customer = 'Slovnaft-ext-2024'  
#domain = 'slovnaft.orbitaleye.nl'
#keep_reporting_intervals = []
#cnt_limit = 20
#566

#customer = 'NGC-2023'  
#domain = 'ngc.orbitaleye.nl'
#keep_reporting_intervals = []
#cnt_limit = 30
#1100

#customer = 'ENAGAS-2025-processing'  
#domain = 'enagas.orbitaleye.nl'
#keep_reporting_intervals = [['2025-01-01', '2025-03-30']]
#cnt_limit = 20
#0 ????

#customer = 'TC-Energy-2025-processing'
#domain = 'tcenergy.orbitaleye.nl'
#keep_reporting_intervals = [['2025-01-01', '2025-02-01']]
#cnt_limit=50

#customer = 'Waternet-2024'  
#domain = 'waternet.orbitaleye.nl'
#keep_reporting_intervals = []
#cnt_limit = 55
#227



save_folder = './DATASET_previews/{}'.format(customer)
if os.path.exists(save_folder):
    shutil.rmtree(save_folder)
os.mkdir(save_folder)

#customer = 'TC-Energy-2024'
#domain = 'tcenergy.orbitaleye.nl'
#not working ??


##tha prepei na kanw adjust kai to corridor width???????????

corridor_width = 9435  # meters
customer_db = CustomerDatabase(ov.get_customerdb_credentials()['username'], ov.get_customerdb_credentials()['password'])

corridor_poly = find_corridor_poly(customer, domain, corridor_width,  customer_db)

if not keep_reporting_intervals:
    keep_reporting_intervals = find_reporting_intervals(customer)


# get the previewsc
#print('project', project['reporting_intervals'][2])
#print('project', project['reporting_intervals'][8])




def get_final_previews(keep_reporting_intervals, corridor_poly):


    previews_bounds_list = []
    keep_previews = []
    for _, keep_date in enumerate(tqdm(keep_reporting_intervals)):

        project_start = datetime.strptime(keep_date[0], '%Y-%m-%d')  # datetime.strptime(project['first_report_date'], '%Y-%m-%d')
        project_end = datetime.strptime(keep_date[1], '%Y-%m-%d')  
        previews = get_skywatch_archive_results_from_db(corridor_poly.envelope, project_start, project_end)
        # previews = previews.sort(key=lambda x: x['start_time'])

    # shopping_cart = geometry.MultiPolygon()

        #print('len', len(previews))

        ####################################################################################################################################
        #creds = ov.get_sarccdb_credentials()

        for preview in previews:
        #    print(preview['bounds'])    
            if not previews_bounds_list:
                previews_bounds_list.append(preview['bounds']) 
            else:
                result = check_intersections(preview['bounds'], previews_bounds_list)    #####################
                if result==1:
                    keep_previews.append(preview)

    return keep_previews


final_previews = get_final_previews(keep_reporting_intervals, corridor_poly)

print('finall', len(final_previews))


random.shuffle(final_previews)

#final_previews = final_previews[:100]

scales = [0.,10.,20.,30.,40.,50., 60.]

cnts = [0, 0, 0, 0, 0, 0]

scales_keep_previews = []


for _,preview in enumerate(tqdm(final_previews)):

    try:
        image = get_image(preview['preview_uri'])

        rx, ry = load_image(image, preview['bounds'].bounds)


        for i in range(0, len(scales)):
            if scales[i]<=rx<=scales[i+1] and cnts[i]<cnt_limit:
                cnts[i] = cnts[i] + 1
                preview['scale'] = round(rx)
                preview['image'] = image
                scales_keep_previews.append(preview)
    except:
        pass

print('cn', cnts)



for _,preview in enumerate(tqdm(scales_keep_previews)):
#    image = get_image(preview['preview_uri'])
#    rx, ry = load_image(image, preview['bounds'].bounds)

    image = Image.fromarray(preview['image'])
    image.save('./DATASET_previews/{}/{}_{}.png'.format(customer, preview['db_id'], round(preview['scale'])))


