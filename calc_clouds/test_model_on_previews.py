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


def extract_clouds(customer, corridor_width, config, project, project_start, project_end):
    
    with RegionsDb(config) as database:
        database_customer = database.get_regions_by_customer(customer)

    # get corridor
    ce_login = ov.get_project_server_credentials(project['name'])

    client = cosmic_eye_client.connect('ptt.orbitaleye.nl')
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

    shopping_cart = geometry.MultiPolygon()

    print('len', len(previews))


    ####################################################################################################################################
    creds = ov.get_sarccdb_credentials()

    ###################################################################################################################################################
    #### Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('device', device)
    #### Load trained model
    model_pth = '/cephfs/installers/ml_models/high-res-cloud-rgb-v1/net_30.pt'
    trained_model = myUNF.UNetFormer(num_classes=4) #.to(device)

    trained_model.load_state_dict(torch.load(model_pth, weights_only=True, map_location=device))
    trained_model = trained_model.eval()
    ###################################################################################################################################################

    cloudy = cloudy_class.Cloudy(creds, customer, trained_model, corridor_poly, "cuda", min_polygon_size=0.0000001)

    vals = []

    print('len', len(previews))
    
    for preview in previews:
    #if preview['db_id']==187389:
        print(preview)    
        bounds_corridor = preview['bounds'].intersection(corridor_poly).buffer(0.)
        ingested_manually = funcs.get_ingestion_results_from_db(preview['start_time'], customer, bounds_corridor)

        ingested_manually_poly = geometry.MultiPolygon([x['bounds'].buffer(0.) for x in ingested_manually]).buffer(0.)

        # TODO: @Can then you compare this area to what you computed
        #print(preview['preview_uri'])
        #print(preview['db_id'])

    ###################################################################################################################################################

        cloud_polygons = cloudy.get_clouds(preview)



        if cloud_polygons is not None:

            cloud_mask_preview = geometry.MultiPolygon(cloud_polygons)

            shapely_polys = [geometry.shape(g) for g in cloud_polygons if geometry.shape(g).is_valid and not geometry.shape(g).is_empty]

            # Merge into MultiPolygon
            multi = geometry.MultiPolygon([p for p in shapely_polys if isinstance(p, geometry.Polygon)])
            funcs.save_geojson('./POLS/{}.json'.format(preview['db_id']), multi)

        else:
            cloud_mask_preview = geometry.MultiPolygon()

        # See part that intersect with the corridor
        non_cloudy_part_preview = bounds_corridor.difference(cloud_mask_preview).buffer(0.)
        preview_corridor = non_cloudy_part_preview.intersection(bounds_corridor).buffer(0.).difference(shopping_cart).buffer(0.)
        # Add to shopping cart
        shopping_cart = shopping_cart.union(preview_corridor).buffer(0.)
        # Compare to what was ordered
        difference_with_analyst = preview_corridor.difference(ingested_manually_poly).buffer(0.)
        if not difference_with_analyst.is_empty:
            epsg_utm = funcs.convert_wgs_to_utm(difference_with_analyst.centroid.x, difference_with_analyst.centroid.y)
            difference_with_analyst_utm = funcs.convert_epsg_geometry(difference_with_analyst, epsg_utm)
            print('diff', difference_with_analyst_utm.area / 1e6)
            vals.append(difference_with_analyst_utm.area / 1e6)


    #overit = 0
    #underit = 0
    #for val in vals:
    #    if val<10:
    #        underit = underit +1
    #    else:
    #        overit = overit + 1

    #obs = np.arange(1, len(vals)+1)
    #threshold = 10

    ## Set bar colors based on the threshold
    #colors = ['blue' if value <= threshold else 'red' for value in vals]

    ## Create the bar graph with the specified colors
    #plt.bar(obs, vals, color=colors)

    ## Add labels and title
    #plt.xlabel('Categories')
    #plt.ylabel('Values')
    #plt.title('{} under 10, {} over 10'.format(underit, overit))
    #plt.savefig('plt.png')



if __name__ == "__main__":
    # Setup
    customer = 'PTT-2024'
    corridor_width = 1000  # meters

    customer_db = CustomerDatabase(ov.get_customerdb_credentials()['username'], ov.get_customerdb_credentials()['password'])
    project = customer_db.get_project_by_name(customer)

    config = ov.get_sarccdb_credentials()

    project_start = datetime.strptime(project['reporting_intervals'][26][0], '%Y-%m-%d')  # datetime.strptime(project['first_report_date'], '%Y-%m-%d')
    project_end = datetime.strptime(project['reporting_intervals'][26][1], '%Y-%m-%d')  


    extract_clouds(customer, corridor_width, config, project, project_start, project_end)



#import json
#geojson = {'type': 'FeatureCollection', 'name': 'Shopping_cart', 'csr': {'type': 'name', 'properties': {'name': 'urn:ogc:def:crs:OGC:1.3:CRS84'}}, 'features': []}
#tmp_i = {'type': 'Feature', 'properties': {'Name': str(1)},
#        'geometry': geometry.mapping(shopping_cart)}
#geojson['features'].append(tmp_i)
#json.dump(geojson, open('./check_shopcart/shopcart.geojson', 'w'))
###################################################################################################################################################



    
