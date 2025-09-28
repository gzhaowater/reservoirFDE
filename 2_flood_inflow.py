#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from scipy.stats import pearson3
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

import os
import psutil
process = psutil.Process(os.getpid())
print(process.memory_info().rss/1e6)

def generate_unit_hydrograph(daily_flows):
    """
    通过历史洪水事件综合生成设计洪水过程线。

    参数:
    daily_flows (pd.DataFrame): 包含'flow'列和datetime索引的日流量数据。

    返回:
    duh_flow: 单位洪水过程量
    dimless_time_axis: 单位洪水过程时间
    representative_time_to_peak: 代表性洪峰时间
    """
    
    # 1. 识别洪水事件
    # 定义一个阈值来识别潜在的洪水（例如，流量超过50%分位数）
    threshold = daily_flows['flow'].quantile(0.50)
    above_threshold_dates = daily_flows.index[daily_flows['flow'] > threshold]
    
    if above_threshold_dates.empty:
        print("警告：未能在数据中识别出显著的洪水事件。无法生成过程线。")
        return None
        
    # 识别独立的洪水事件（如果两个洪峰相隔小于7天，则视为一个事件）
    events = []
    if len(above_threshold_dates) > 0:
        current_event_start = above_threshold_dates[0]
        for i in range(1, len(above_threshold_dates)):
            if (above_threshold_dates[i] - above_threshold_dates[i-1]).days > 7:
                # 确定上一个事件的峰值日期
                event_period = daily_flows.loc[current_event_start:above_threshold_dates[i-1]]
                if not event_period.empty:
                    peak_date = event_period['flow'].idxmax()
                    events.append(peak_date)
                current_event_start = above_threshold_dates[i]
        # 处理最后一个事件
        last_event_period = daily_flows.loc[current_event_start:above_threshold_dates[-1]]
        if not last_event_period.empty:
            events.append(last_event_period['flow'].idxmax())

    # 去除重复的事件峰值日期
    unique_peak_dates = sorted(list(set(events)))
    
    # 2. 提取、对齐和标准化洪水过程线
    normalized_hydrographs = []
    total_durations = []
    time_to_peaks = []

    # 准备一个标准的无量纲时间轴用于插值

    # print(f"从数据中识别出 {len(unique_peak_dates)} 个独立的洪水事件用于过程线综合。")
    
    for peak_time in unique_peak_dates:
        # 定义一个围绕洪峰的窗口来提取完整的洪水过程
        window_start = peak_time - pd.Timedelta(days=7)
        window_end = peak_time + pd.Timedelta(days=14)
        event_hydro = daily_flows.loc[window_start:window_end]['flow']
        
        if event_hydro.empty or event_hydro.max() == 0:
            continue

        # 确保洪峰是窗口内的最大值
        if event_hydro.idxmax() != peak_time:
            continue

        peak_flow = event_hydro.max()
        
        # 寻找洪水起涨点
        rising_limb = event_hydro.loc[:peak_time]

        if pd.isna(hydro_start_time):
            hydro_start_time = rising_limb.index.min()

        # 计算特征时间：洪峰时间 (Time to Peak)
        ttp_hours = (peak_time - hydro_start_time).total_seconds() / 3600.0
        if ttp_hours <= 0: continue # 忽略无效事件
        
        time_to_peaks.append(ttp_hours)

        # 时间轴无量纲化 (相对于洪峰时间，并用洪峰时间缩放)
        time_hours = (event_hydro.index - peak_time).total_seconds() / 3600.0
        dimless_time = time_hours / ttp_hours

        # 流量轴无量纲化
        dimless_flow = event_hydro.values / peak_flow
        
        # 插值到标准时间轴
        interp_flow = np.interp(dimless_time_axis, dimless_time, dimless_flow, left=0, right=0)
        normalized_hydrographs.append(interp_flow)

    if not normalized_hydrographs:
        print("警告：处理后无有效的历史洪水过程线。无法继续。")
        return None

    # 3. 平均得到无量纲单位线 (DUH)
    duh_flow = np.mean(normalized_hydrographs, axis=0)

    # 使用历史洪峰时间的平均值作为代表性的时间尺度
    representative_time_to_peak = np.mean(time_to_peaks)
    
    return duh_flow, dimless_time_axis, representative_time_to_peak


def calculate_design_peak(daily_flows, return_period=100):
    """
    通过对数皮尔逊III型分布进行洪水频率分析，计算指定重现期的设计洪峰。

    参数:
    daily_flows (pd.DataFrame): 包含'flow'列和datetime索引的日流量数据。
    return_period (int): 需要计算设计洪峰的重现期，默认为100。

    返回:
    float: 设计洪峰流量 (Q_T)。
    tuple: 频率分析的相关参数，用于绘图。
    """
    # 1. 提取年最大值序列 (AMS)
    ams = daily_flows['flow'].resample('YE').max()
    ams = ams.dropna()

    # 2. 对数转换 (base 10)
    log_ams = np.log10(ams)

    # 3. 计算矩法参数
    mu_y = log_ams.mean()
    sigma_y = log_ams.std(ddof=1) # 使用样本标准差
    cs = log_ams.skew()

    # 4. 计算非超越概率
    p = 1.0 - 1.0 / return_period
    
    # 5. 使用PPF函数计算对数洪水值
    # 参数映射: skew=cs, loc=mu_y, scale=sigma_y
    log_q_t = pearson3.ppf(p, skew=cs, loc=mu_y, scale=sigma_y)
    
    # 6. 反向转换得到设计洪峰
    q_t = 10**log_q_t
        
    return q_t, (ams, log_ams, mu_y, sigma_y, cs)


mdata = pd.read_parquet('./out_1_resevoir_zflows.parquet')

dates = pd.to_datetime(pd.date_range(start='1984-01-01', end='2018-12-31', freq='D'))

duh_data = []
peak_data = []
for index, row in mdata.iterrows():

    lake_id = row['GDW_ID']
    df_daily = pd.DataFrame({'flow': row['Inflow']}, index=dates)
    try:
        duh_flow, dimless_time_axis, representative_time_to_peak = generate_unit_hydrograph(df_daily)
    except:
        print(lake_id, end='|')
        continue

    duh_data.append([lake_id, dimless_time_axis - dimless_time_axis.min(), duh_flow, representative_time_to_peak])
    
    for RETURN_PERIOD in [10, 25, 50, 100]:
        design_peak, ffa_params = calculate_design_peak(df_daily, return_period=RETURN_PERIOD)
        peak_data.append([lake_id, RETURN_PERIOD, design_peak])
    