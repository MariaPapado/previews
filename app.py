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
import os
import rasterio
import torch
import sys
sys.path.append('./src/')
from src import myUNF
from src import cloudy_class
from src import funcs
from tqdm import tqdm

import json
import base64
from pydantic import BaseModel

from fastapi import FastAPI, HTTPException, Request, status
import uvicorn

from fastapi.responses import JSONResponse
import traceback



app = FastAPI()


class HealthCheck(BaseModel):
    """Response model to validate and return when performing a health check."""

    status: str = "OK"


@app.get(
    "/healthcheck",
    tags=["healthcheck"],
    summary="Perform a Health Check",
    response_description="Return HTTP Status Code 200 (OK)",
    status_code=status.HTTP_200_OK,
    response_model=HealthCheck,
)
def get_health() -> HealthCheck:
    return HealthCheck(status="OK")

def extract_clouds(customer, domain, corridor_width, date_start, date_end):

    project_start = datetime.strptime(date_start, '%Y-%m-%d')  # datetime.strptime(project['first_report_date'], '%Y-%m-%d')
    project_end = datetime.strptime(date_end, '%Y-%m-%d') 
    
    ov.configure_secrets_backend(use_secretsmanager=True)

#    config = ov.get_sarccdb_credentials()
#    config['database'] = config.pop('dbname')
    #regions_db_cred = ov.get_sarccdb_credentials()

#    settings_db = {
#                        "host": regions_db_cred.get("host"),
#                        "port": regions_db_cred.get("port"),
#                        "user": regions_db_cred.get("user"),
#                        "password": regions_db_cred.get("password"),
#                        "database": regions_db_cred.get("dbname"),
#                        "schema": "tpi_dashboard,public",
#                    }
    

#    print('aaaaaaaaaaaaaaa', ov.get_customerdb_credentials()['username'], ov.get_customerdb_credentials()['password'])
#    customer_db = CustomerDatabase(ov.get_customerdb_credentials()['username'], ov.get_customerdb_credentials()['password'])

#    project = customer_db.get_project_by_name(customer)


#    with RegionsDb(config) as database:
#        database_customer = database.get_regions_by_customer(customer)

    # get corridor
    #ce_login = ov.get_project_server_credentials(project['name'])
    ce_login = {'host': 'ptt.orbitaleye.nl', 'password': 's6ziEl{7wqtx8?>~oGe$hsj`c', 'port': '9996', 'user': 'Orbital Eye'}

    client = cosmic_eye_client.connect(domain)
    client.login(ce_login['user'], ce_login['password'])
    pipe_contours = client.getAllPipelines()

    corridor_all = []
    for _, pipeline in enumerate(tqdm(pipe_contours)):
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

    shopping_cart = geometry.MultiPolygon()

    ####################################################################################################################################
    creds = ov.get_sarccdb_credentials()

    ###################################################################################################################################################
    #### Set device
#    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device='cpu'
    #### Load trained model
    model_pth = 'net_30.pt'
    trained_model = myUNF.UNetFormer(num_classes=4) #.to(device)

    trained_model.load_state_dict(torch.load(model_pth, weights_only=True, map_location=device))
    trained_model = trained_model.eval()
    ###################################################################################################################################################

    cloudy = cloudy_class.Cloudy(creds, customer, trained_model, corridor_poly, device, min_polygon_size=0.0000001)

    vals = []

    print('len', len(previews))
    results_list = []
    #previews = previews[:1]
    print('ok')
    for _, preview in enumerate(tqdm(previews)):
      if preview['db_id']==274287:
        print(preview)    
        bounds_corridor = preview['bounds'].intersection(corridor_poly).buffer(0.)
#        ingested_manually = funcs.get_ingestion_results_from_db(preview['start_time'], customer, bounds_corridor)

#        ingested_manually_poly = geometry.MultiPolygon([x['bounds'].buffer(0.) for x in ingested_manually]).buffer(0.)

    ###################################################################################################################################################

        cloud_polygons = cloudy.get_clouds(preview)

        if cloud_polygons is not None:

            cloud_mask_preview = geometry.MultiPolygon(cloud_polygons)

            shapely_polys = [geometry.shape(g) for g in cloud_polygons if geometry.shape(g).is_valid and not geometry.shape(g).is_empty]

            # Merge into MultiPolygon
            multi = geometry.MultiPolygon([p for p in shapely_polys if isinstance(p, geometry.Polygon)])
            #funcs.save_geojson('./POLS/{}.json'.format(preview['db_id']), multi)

        else:
            cloud_mask_preview = geometry.MultiPolygon()

        # See part that intersect with the corridor
        non_cloudy_part_preview = bounds_corridor.difference(cloud_mask_preview).buffer(0.)
        preview_corridor = non_cloudy_part_preview.intersection(bounds_corridor).buffer(0.).difference(shopping_cart).buffer(0.)

        current_dict = {
            "preview_id": preview['db_id'],
#            "noncloudy_polygon": preview_corridor.wkt
            "noncloudy_polygon": cloud_mask_preview.wkt

        }
        #results_list.append(current_dict)

    return current_dict


##########################################################################################################
#def validate_image(encoded_image: str, width: int, height: int) -> np.ndarray:
#    #    decoded_image = np.fromstring(base64.b64decode(encoded_image), dtype=np.float32)
#    decoded_image = np.frombuffer(base64.b64decode(encoded_image), dtype=np.float32)

#    decoded_image = decoded_image.reshape(width, height, 3)
#    decoded_image = np.ascontiguousarray(decoded_image)
#    return decoded_image


class request_body(BaseModel):
    customer: str
    domain: str
    corridor_width: int
    date_start: str
    date_end: str


@app.post("/predict")
def predict(data: request_body):
    date_start, date_end = data.date_start, data.date_end
    customer, domain, corridor_width = data.customer, data.domain, data.corridor_width
    #image = validate_image(data.imageData, w, h)
    resultData = extract_clouds(customer, domain, corridor_width, date_start, date_end)
    #resultData = np.ascontiguousarray(resultData)
    #result_base64 = base64.b64encode(resultData.tobytes()).decode("utf-8")
    return {"result": resultData}


# Custom exception handler to show detailed errors
@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    error_message = traceback.format_exc()
    print(error_message)  # This ensures the full error is logged in Docker logs
    return JSONResponse(
        status_code=500, content={"error": str(exc), "traceback": error_message}
    )
######################################################################################################################
