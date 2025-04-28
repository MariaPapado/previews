from geopy.distance import geodesic
import cv2
import torch
import numpy as np
from shapely.validation import make_valid
import rasterio
import rasterio.features
from shapely.geometry import Polygon, MultiPolygon, shape, box
import geopandas as gpd
import requests
import os
import orbital_vault as ov
import psycopg
from shapely import ops, geometry, wkb
from PIL import Image
import json
from rasterio.features import shapes

class Cloudy:
    def __init__(self, creds, customer, trained_model, corridor_poly, device, min_polygon_size=0):
        self.trained_model = trained_model
        self.device = device
        self.min_polygon_size = min_polygon_size
        self.creds = creds
        self.corridor_poly = corridor_poly
        self.customer = customer


    def get_image(self, image_url):
        resp = requests.get(image_url)
        try:
            image = np.asarray(bytearray(resp.content))
            #image = image.astype(np.int16)
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
            image = image[:, :, [2, 1, 0]] 

        except:
            print(resp)
            
        return image

    def load_image(self, image, pid, resize=False, target_res=10, bounds=None):

        if resize and bounds:
            region_width = geodesic((bounds[1], bounds[0]), (bounds[1], bounds[2])).meters
            region_height = geodesic((bounds[1], bounds[0]), (bounds[3], bounds[0])).meters
            height, width = image.shape[:2]
            rx, ry = region_height / height, region_width / width # how many meters per pixel
            mx, my = rx/target_res, ry/target_res

            nx, ny = int(np.round(mx*height)), int(np.round(my*width))
            #image_save = image.astype(np.uint8)
            #image_save = Image.fromarray(image_save)
            #image_save.save('./previews/{}_{}_{}.png'.format(pid, ny, nx))
            image = cv2.resize(image, (ny,nx), interpolation=cv2.INTER_NEAREST)            

            return image

    def stretch_8bit(self, band, lower_percent=2, higher_percent=98):
        a = 0
        b = 255
        real_values = band.flatten()
        real_values = real_values[real_values > 0]
        c = np.percentile(real_values, lower_percent)
        d = np.percentile(real_values, higher_percent)

        t = a + (band - c) * (b - a) / float(d - c)
        t[t<a] = a
        t[t>b] = b
        return t/255.


    def preprocess_image(self, image):

        image = self.stretch_8bit(image)

        return image


    def pad_left(self, arr, n=256): 
        deficit_x = (n - arr.shape[1] % n) 
        deficit_y = (n - arr.shape[2] % n) 
        if not (arr.shape[1] % n): 
            deficit_x = 0 
        if not (arr.shape[2] % n): 
            deficit_y = 0 
        arr = np.pad(arr, ((0, 0), (deficit_x, 0), (deficit_y, 0)), mode='constant', constant_values=0) 
        return arr, deficit_x, deficit_y


    def postprocess_mask(self, image):
        # Find contours
        contours_first, hierarchy = cv2.findContours(
            image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        contours = []
        for con in contours_first:
            area = cv2.contourArea(con)
            if area > 5:
                contours.append(con)

        output = cv2.drawContours(
            np.zeros((image.shape[0], image.shape[1], 3)),
            contours,
            -1,
            (255, 255, 255),
            thickness=cv2.FILLED,
        )

        # Smooth the mask
    #    blurred_mask = cv2.GaussianBlur(output, (25, 25), 0)

        # Threshold back to binary
    #    _, smoothed_mask = cv2.threshold(blurred_mask, 127, 255, cv2.THRESH_BINARY)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        smoothed_mask = cv2.morphologyEx(output, cv2.MORPH_CLOSE, kernel)

        return smoothed_mask

    def generate_mask(self, image, or_x, or_y):

        self.trained_model.to(self.device).eval()
        image = np.transpose(image, (2,0,1))
        image_pad, dx, dy = self.pad_left(image, n=256)
        image_tensor = torch.from_numpy(image_pad).float().to(self.device)
        with torch.no_grad():
            if image_tensor.ndim == 3:
                img = image_tensor.unsqueeze(0).to(self.device)

            pred_mask = self.trained_model(img)
            pred_mask = pred_mask[:, :, dx:, dy:]

            pred_mask = torch.argmax(pred_mask, dim=1).squeeze(0).cpu().numpy()  # Get predicted mask
            idx = np.where(pred_mask!=1)
            pred_mask[idx]=0

            pred_mask = cv2.resize(pred_mask, (or_y,or_x), interpolation=cv2.INTER_NEAREST)
            pred_mask = self.postprocess_mask(pred_mask.astype(np.uint8))

            # Convert image back to numpy for visualization
            #img_np = img.squeeze().cpu().numpy().transpose(1, 2, 0)  # Convert back to H, W, C
            #image_min, image_max = img_np.min(), img_np.max()
            #img_np = 255 * (img_np - image_min) / (image_max - image_min)  # Stretch to [0, 255]
            #img_np = img_np.astype(np.uint8)

            return pred_mask

    def get_polygons_from_single_change_mask(self, mask, geotiff_transform, connectivity=4):
        # Is mask give, then apply, else continue without

        # Convert to polygons
        result = []
#        print('hereeeeeeeeeeee')
#        print(np.unique(mask))
        for shape, value in rasterio.features.shapes(mask, connectivity=connectivity, transform=geotiff_transform):
#            print('aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa')
            if value == 255:
                result.append(Polygon(shape['coordinates'][0]))

        return result

    def filter_polygons(self, polygons):
        """
        Convert polygons to valid Shapely geometries and filter by area.
        """
        min_polygon_size=0
        valid_polygons = [make_valid(shape(poly)) for poly in polygons if poly['type'] == 'Polygon']
        return [poly for poly in valid_polygons if poly.area > min_polygon_size]


    def create_geodataframe(self, polygons, crs="EPSG:4326"):
        """
        Create a GeoDataFrame from polygons.
        """
        if not polygons:
            print("No valid polygons to process.")
            return None
        return gpd.GeoDataFrame(geometry=polygons, crs=crs)


    def save_tif_coregistered(self, filename, image, poly, channels=1, factor=1):
        height, width = image.shape[0], image.shape[1]
        geotiff_transform = rasterio.transform.from_bounds(poly.bounds[0], poly.bounds[1],
                                                        poly.bounds[2], poly.bounds[3],
                                                        width/factor, height/factor)

        new_dataset = rasterio.open(filename, 'w', driver='GTiff',
                                    height=height/factor, width=width/factor,
                                    count=channels, dtype='uint8',
                                    crs='+proj=latlong',
                                    transform=geotiff_transform)

        # Write bands
        if channels>1:
            for ch in range(0, image.shape[2]):
                new_dataset.write(image[:,:,ch], ch+1)
        else:
            new_dataset.write(image, 1)
        new_dataset.close()

        return True


    def get_clouds(self, preview):

        #oneimg, poly_ingested_manually_tasked, tasking_bounds = self.retrieve_data(preview['db_id'])
        bounds = preview['bounds'].bounds

        original_image_array = self.get_image(preview['preview_uri'])
        or_x, or_y = original_image_array.shape[0], original_image_array.shape[1]

        resize, target_res = True, 10
        image = self.load_image(original_image_array, preview['db_id'], resize, target_res, bounds)
        image = self.preprocess_image(image)  # Ensure input is on GPU
        mask = self.generate_mask(image, or_x, or_y)
        mask[(mask==2) | (mask==3)] = 0

        #cv2.imwrite('./PROBS/{}.png'.format(preview['db_id']), mask*255)


        idx_n0 = np.where(mask!=0)
        print('idx_n0', idx_n0)
        
        if len(idx_n0[0]!=0):
            polygon=True
        else:
            polygon=False
        if polygon==True:

            p_bounds = preview['bounds'].bounds
            transform = rasterio.transform.from_bounds(*p_bounds, image.shape[1], image.shape[0])
            cloud_polygons = self.get_polygons_from_single_change_mask(mask[:,:,0].astype(np.uint8), transform, 4)
#            filtered_polygons = self.filter_polygons(cloud_polygons)
            #gdf_cloud = create_geodataframe(filtered_polygons)

            return cloud_polygons
        else:
            return None

