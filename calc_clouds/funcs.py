from datetime import datetime, timedelta
import pyproj
from shapely import ops, geometry, wkb
import orbital_vault as ov
import psycopg2 as sqlsystem
import numpy as np
import json

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


def get_ingestion_results_from_db(target_timestamp: int, customer: str, location: geometry.Polygon):

    config = ov.get_sarccdb_credentials()
    schema = 'public'
    connection_string = f"host={config['host']} port={config['port']} user='{config['user']}' password='{config['password']}' dbname={config['database']} options='-c search_path={schema}'"
    with sqlsystem.connect(connection_string) as db:
        # Get a cursor object
        cursor = db.cursor()
        # Search archive images
        cursor.execute("SELECT id,aoi_id,target_date,bounds FROM skywatch_aoi_request WHERE customer = %(customer)s and region_id = -3 and target_date = %(target_date)s",
                       {'target_date': datetime(1970, 1, 1) + timedelta(seconds=target_timestamp),
                        'customer': customer})
        output = cursor.fetchall()

    output_sorted = []

    keys_i = ['id', 'aoi_id', 'target_date', 'bounds']
    for i in output:
        tmp_dict = {keys_i[j]: i[j] for j in range(len(i))}
        tmp_dict['bounds'] = wkb.loads(tmp_dict['bounds'], hex=True)
        if tmp_dict['bounds'].intersects(location):
            output_sorted.append(tmp_dict)

    return output_sorted

def save_geojson(filename, target_poly):
    geojson = {'type': 'FeatureCollection', 'name': 'Shopping_cart', 'csr': {'type': 'name', 'properties': {'name': 'urn:ogc:def:crs:OGC:1.3:CRS84'}}, 'features': []}
    tmp_i = {'type': 'Feature', 'properties': {'Name': str(1)},
            'geometry': geometry.mapping(target_poly)}
    geojson['features'].append(tmp_i)
    json.dump(geojson, open(filename, 'w'))
