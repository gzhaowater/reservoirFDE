#!/usr/bin/env python
# coding: utf-8

import ee
ee.Authenticate()
ee.Initialize()

import pandas as pd
import time


lakes = ee.FeatureCollection("projects/zgeternity/assets/Reservoirs/GDW_fde")
print(lakes.size().getInfo()) 

jrc_occr = ee.ImageCollection("projects/zgeternity/assets/Reservoirs/water_occurrence_1000").mosaic()

srtm_nasa = ee.Image("NASA/NASADEM_HGT/001").select(['elevation'], ['srtm'])

alos2_2 = ee.Image("JAXA/ALOS/AW3D30/V2_2").select(['AVE_DSM'], ['alos2_2'])

alos3_2_col = ee.ImageCollection('JAXA/ALOS/AW3D30/V3_2').select(['DSM'], ['alos3_2'])
proj3_2 = alos3_2_col.first().select(0).projection()
alos3_2 = alos3_2_col.mosaic().setDefaultProjection(proj3_2)

aster = ee.Image("projects/sat-io/open-datasets/ASTER/GDEM").select(['b1'], ['aster'])

glo30_col = ee.ImageCollection("COPERNICUS/DEM/GLO30").select(['DEM'], ['glo30'])
proj_glo30 = glo30_col.first().select(0).projection()
glo30 = glo30_col.mosaic().setDefaultProjection(proj_glo30)

def elev_extract(lake):

    occr_area = ee.Image.pixelArea().addBands(jrc_occr)
    occr_area_values = occr_area.reduceRegion(
        reducer=ee.Reducer.sum().group(1),    geometry=lake.geometry(), scale=30, maxPixels=1e10).get('groups')

    occr_srtm_nasa = srtm_nasa.addBands(jrc_occr)
    occr_srtm_nasa_values = occr_srtm_nasa.reduceRegion(
        reducer=ee.Reducer.intervalMean(25, 75).group(1), geometry=lake.geometry(), scale=30, maxPixels=1e10).get('groups')

    occr_alos2_2 = alos2_2.addBands(jrc_occr)
    occr_alos2_2_values = occr_alos2_2.reduceRegion(
        reducer=ee.Reducer.intervalMean(25, 75).group(1), geometry=lake.geometry(), scale=30, maxPixels=1e10).get('groups')

    occr_alos3_2 = alos3_2.addBands(jrc_occr)
    occr_alos3_2_values = occr_alos3_2.reduceRegion(
        reducer=ee.Reducer.intervalMean(25, 75).group(1), geometry=lake.geometry(), scale=30, maxPixels=1e10).get('groups')

    occr_aster = aster.addBands(jrc_occr)
    occr_aster_values = occr_aster.reduceRegion(
        reducer=ee.Reducer.intervalMean(25, 75).group(1), geometry=lake.geometry(), scale=30, maxPixels=1e10).get('groups')
    
    occr_glo30 = glo30.addBands(jrc_occr)
    occr_glo30_values = occr_glo30.reduceRegion(
        reducer=ee.Reducer.intervalMean(25, 75).group(1), geometry=lake.geometry(), scale=30, maxPixels=1e10).get('groups')

    ring = ee.Geometry.MultiPoint(ee.Array(lake.geometry().geometries() \
                            .map(lambda item: ee.Geometry(item).coordinates()).flatten()) \
                                  .reshape([-1, 2]).toList())

    ring_elevs = srtm_nasa.addBands(alos2_2).addBands(alos3_2) \
                     .addBands(aster).addBands(glo30) \
                     .reduceRegion(reducer=ee.Reducer.intervalMean(25, 75), geometry=ring, scale=30, maxPixels=1e10)

    return ee.Feature(None).set('GDW_ID', lake.get('GDW_ID')).set(ring_elevs) \
                           .set('occr_area', occr_area_values) \
                           .set('occr_srtm_nasa', occr_srtm_nasa_values) \
                           .set('occr_alos2_2', occr_alos2_2_values) \
                           .set('occr_alos3_2', occr_alos3_2_values) \
                           .set('occr_aster', occr_aster_values) \
                           .set('occr_glo30', occr_glo30_values)

group = ee.Number(1000)

n_grp = lakes.size().divide(group).ceil().toInt()
print(n_grp.getInfo())

for i in range(0, n_grp.getInfo(), 1):
    i_ee = ee.Number(i)

    lakes_grp = ee.FeatureCollection(lakes.toList(group, group.multiply(i_ee)))

    output_grp = lakes_grp.map(elev_extract)

    outn = 'GDW_dem_' + str(int(i))

    task = ee.batch.Export.table.toDrive(
                        collection=output_grp,
                        folder='Others',
                        description=outn,
                        fileFormat='GeoJSON')
    task.start()
    print('Submitted ' + str(i))

