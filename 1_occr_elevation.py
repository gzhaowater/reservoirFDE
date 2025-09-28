#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import glob
import json
import zipfile
from datetime import datetime
import geopandas as gpd

from scipy import stats

import os
import psutil
process = psutil.Process(os.getpid())
print(process.memory_info().rss/1e6)
from sys import getsizeof

output = []
with zipfile.ZipFile("out_0_gee_occr_dem.zip", "r") as zipf:
    for fname in zipf.namelist():
        if 'geojson' in fname:
            with zipf.open(fname) as myZip:
                mdata = pd.read_json(myZip)
                print(fname, process.memory_info().rss/1e6, datetime.now())
                
                for index, row in mdata.iterrows():
                    prp = row['features'].get('properties')
                    fid = prp.get('GDW_ID')
                    
                    for mdict in ['occr_alos2_2', 'occr_alos3_2', 'occr_aster', 'occr_glo30', 'occr_srtm_nasa']:
                        ring_dict = mdict.replace('occr_', '').replace('nasa', 'new')
                        tmp = {'GDW_ID': fid, 'source': mdict.replace('occr_', ''), 'ring': prp.get(ring_dict)}
                        
                        if len(prp.get(mdict)) > 0:
                            for tmp1 in prp.get(mdict):
                                if 'mean' in tmp1:
                                    tmp[tmp1['group']] = tmp1['mean']
                            output.append(tmp)
                    
                    mdict = 'occr_area'
                    tmp = {'GDW_ID': fid, 'source': mdict}
                    for tmp1 in prp.get(mdict):
                        tmp[tmp1['group']] = tmp1['sum']
                    output.append(tmp)

output_df = pd.DataFrame.from_dict(output)

output_df1 = output_df.drop(columns=['ring']).set_index(['GDW_ID', 'source']).stack().reset_index()

output_df2 = output_df1.pivot(index=['GDW_ID', 'level_2'], columns='source', values=0).reset_index() \
                .rename({'level_2': 'group', 'occr_area': 'area_m2'}, axis=1)

output_df2['group'] = output_df2['group'].apply(lambda x: round(x/10))

output_df2 = output_df2.groupby(['GDW_ID', 'group']).agg({'alos2_2': 'mean', 'alos3_2': 'mean', 'aster': 'mean', 
                                                          'glo30': 'mean', 'srtm_nasa': 'mean', 'area_m2': 'sum'}).reset_index()

N = 10
cols = ['srtm_nasa', 'alos3_2', 'aster', 'glo30']

def df_process(df_group):
    
    glb_g_min = df_group['group'].min()
    glb_g_max = df_group['group'].max()
    df_all = df_group[df_group['area_m2'] > 900*10]
    
    g_min_sel = 0
    default = 0
    for col in cols:
        
        df_tmp = df_all[['group', col]].copy()
        
        df_tmp = df_tmp.dropna()
        
        if len(df_tmp) > 0:
            y_padded = np.pad(df_tmp[col].values, (N//2, N-1-N//2), mode='edge')
            df_tmp[col] = np.convolve(y_padded, np.ones((N,))/N, 'valid')

            g_max, v_max = df_tmp[df_tmp[col] == df_tmp[col].max()].iloc[0][['group', col]]
            g_min, v_min = df_tmp[df_tmp[col] == df_tmp[col].min()].iloc[0][['group', col]]
            
            ### check bottom flat line ###
            vmin1 = v_min + 0.5
            df_tmp_bottom = df_tmp[(df_tmp[col] <= vmin1) & (df_tmp['group'] <= g_min)]
            if len(df_tmp_bottom) >=5:
                df_tmp_min = df_tmp[(df_tmp[col] >= vmin1) & (df_tmp['group'] <= g_min)]
                if len(df_tmp_min) >= 3:
                    g_min, v_min = df_tmp_min.iloc[-1][['group', col]]

            if g_min > g_max:
                df_tmp_valid = df_tmp[df_tmp['group'] <= g_min]
                v_part1 = np.interp(np.arange(glb_g_min, g_min+1, 1), df_tmp_valid['group'], df_tmp_valid[col])

                ############# linear regression for the last part ##############
                
                
                ############# make sure the slope is negative ##############
                
                
            
    if g_min_sel == 0:
        f_col = 'glo30'
        default = 1
        df_tmp = df_group[['group', f_col]].copy()
        glb_g_min = df_tmp['group'].min()
        glb_g_max = df_tmp['group'].max()
        
        df_tmp = df_tmp.dropna()
        if len(df_tmp) > 0:
            y_padded = np.pad(df_tmp[f_col].values, (N//2, N-1-N//2), mode='edge')
            df_tmp[f_col] = np.convolve(y_padded, np.ones((N,))/N, 'valid')
        
            final_g = np.arange(glb_g_min, glb_g_max+1, 1) 
            final_v = np.interp(final_g, df_tmp['group'], df_tmp[f_col])
            f_g_max, f_v_max, f_g_min, f_v_min = np.max(final_g), np.max(final_v), np.min(final_g), np.min(final_v)
        else:
            final_g = np.array([])
            final_v = np.array([])
            f_g_max, f_v_max, f_g_min, f_v_min = 0, 0, 0, 0
    
    return (final_g, final_v, [f_g_max, f_g_min], [f_v_max, f_v_min], f_col, default)


all_out = []

output_df = output_df2.dropna().copy()

for hylak_id, df_group in output_df.groupby('GDW_ID'):

    final_g, final_v, [f_g_max, f_g_min], [f_v_max, f_v_min], f_col, default = df_process(df_group)
    all_out.append([hylak_id, final_g, final_v, f_col, default, f_g_max, f_g_min, f_v_max, f_v_min])

    if len(all_out) % 1000 == 0:
        print(len(all_out), datetime.now())

all_out_df = pd.DataFrame(all_out, columns=['GDW_ID', 'final_g', 'final_v', 'source', 'default', 
                                            'f_g_max', 'f_g_min', 'f_v_max', 'f_v_min'])

display(all_out_df)
