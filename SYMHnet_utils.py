'''
 (c) Copyright 2023
 All rights reserved
 Programs written by Yasser Abduallah
 Department of Computer Science
 New Jersey Institute of Technology
 University Heights, Newark, NJ 07102, USA

 Permission to use, copy, modify, and distribute this
 software and its documentation for any purpose and without
 fee is hereby granted, provided that this copyright
 notice appears in all copies. Programmer(s) makes no
 representations about the suitability of this
 software for any purpose.  It is provided "as is" without
 express or implied warranty.

 @author: Yasser Abduallah
'''

from __future__ import print_function
import warnings
warnings.filterwarnings("ignore")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

try:
    import tensorflow as tf
    tf.get_logger().setLevel('INFO')
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
except Exception as e:
    print('')
import tensorflow
import os 
import numpy as np 
import pandas as pd 
import sys
import csv

from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as md
import matplotlib.ticker as mticker 
import matplotlib
import pickle

import seaborn as sns

import argparse
import time
from time import sleep
from tensorflow.keras.utils import plot_model
from sklearn.metrics import mean_squared_error
import math
from math import sqrt
from scipy.stats import pearsonr
import random
from tensorflow import keras
import tensorflow.compat.v2 as tf
tf.enable_v2_behavior()
from tensorflow.keras import layers, models
import tensorflow_probability as tfp
from sklearn.model_selection import train_test_split
verbose = False 

tfd = tfp.distributions

col_name = str('SYM_H').upper()
col_name = col_name.replace('_','')
col_name = col_name[:-1] +'_' + col_name[-1]

numberOfSamples=100
is_intense = True
uncertainty_margin=1
uncertainty_ep_margin=1

supported_cols = ['SYM_H','SYMH']
supported_test_storms = [i for i in range(26,43)]
columns_names=['Field_magnitude_average','BX_GSE_GSM','BY_GSE','BZ_GSE','BY_GSM','BZ_GSM','Speed','Proton_Density','Flow_pressure','Electric_field',col_name]    
features = columns_names
features1 = ['Scalar_B', 'BZ_GSE', 'BZ_GSM', 'SW_Plasma_Temperature', 'SW_Plasma_Speed', 'Flow_pressure', 'E_elecrtric_field']
sym_col = 'SYM_H'
fill_values = [999.9, 999.9, 999.9, 999.9, 999.9, 999.9, 999.9, 9999999, 999.9, 9999, 999.9, 999.9, 99.99, 999.99]
fill_values = [999.9, 999.9, 999.9, 999.9, 999.9, 999.9, 9999999, 999.9, 9999, 999.9, 999.9, 99.99, 999.99,99.99]

fill_values =[9999.99,9999.99,9999.99,9999.99,9999.99,9999.99,99999.9,999.99,9999999,99.99,999.99]

c_date = datetime.now()

t_window = ''
d_type = ''
data_dir = 'data'
log_handler = None
interval_type = 'hourly'

log_file = None

def create_log_file(dir_name='logs'):
    global log_handler
    global log_file
    try:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name, True)
        log_file = dir_name + '/run_' + str(c_date.month) + '-' + str(c_date.day) + '-' + str(c_date.year) + '.log'
    except Exception as e:
        print('creating default logging file..')
        log_file = 'logs/run_' + str(c_date.month) + '-' + str(c_date.day) + '-' + str(c_date.year) + '.log'
    log_handler = open(log_file, 'a')
    sys.stdout = Logger(log_handler)  


def set_logging(dir_name='logs'):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name, True)    
    log_file = dir_name + os.sep + 'kp_run.log'
    global log_handler
    if os.path.exists(log_file):
        l_stats = os.stat(log_file)
        # print(l_stats)
        l_size = l_stats.st_size
        # print('l_size:', l_size)
            
        if l_size >= 1024 * 1024 * 50:
            files_list = os.listdir('logs')
            files = []
            for f in files_list:
                if 'solarmonitor_html_parser_' in f:
                    files.append(int(f.replace('logs', '').replace('/', '').replace('kp_run_', '').replace('.log', '')))
            files.sort()
            # print(files)
            if len(files) == 0:
                files.append(0)
            os.rename(log_file, log_file.replace('.log', '_' + str(files[len(files) - 1] + 1) + '.log'))
            log_handler = open(log_file, 'w')
        else:
            log_handler = open(log_file, 'a')
    else:
        log_handler = open(log_file, 'w')
    # print('log_handler:', log_handler)

    
class Logger(object):

    def __init__(self, logger):
        self.terminal = sys.stdout
        self.log = logger

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass  





def get_d_str(t):
    y = str(t.year)
    m = str(t.month)
    if len(m) == 1:
        m = '0' + m
    d = str(t.day)
    if len(d) == 1:
        d = '0' + d
    return str(t.year) + '-' + m + '-' + d 


def truncate_float(number, digits=4) -> float:
    try:
        if math.isnan(number):
            return 0.0
        stepper = 10.0 ** digits
        return math.trunc(stepper * number) / stepper
    except Exception as e:
        return number


def set_verbose(b):
    global verbose
    verbose = b

    
def log(*message, verbose=False, end=' '):
    global log_handler
    if True:
        if verbose:
            print('[' + str(datetime.now().replace(microsecond=0)) + '] ', end='')
        log_handler.write('[' + str(datetime.now().replace(microsecond=0)) + '] ')
        for m in message:
            if verbose:
                print(m, end=end)
            log_handler.write(str(m) + ' ')
        if verbose:
            print('')
        log_handler.write('\n')
        
    log_handler.flush()

def print_summary_to_file(s):
    global log_file
    with open(log_file,'a') as f:
        print(s, file=f)
        
    


def get_data (t, dataset_name='training', d_type='hr'):
    file = data_dir + os.sep + file_prefix + str(dataset_name) + '_' + str(t) + str(d_type) + '.csv'
    log('Loading:', dataset_name , 'from:', file)
    data = pd.read_csv(file) 
    return data


def clean_filled_values(data,columns_names):
    for i in range(len(columns_names)):
        c = columns_names[i]
        data = data.loc[~data[c].isin(fill_values)]
    return data


def drop_columns(data, cols=[]):
    for c in cols:
        if c in data.columns:
            log('dropping column:', c)
            data = data.drop(c, axis=1)
    return data


def group_data_series_len(X_train, y_train, series_len):
    X_train_series = []
    y_train_series = []
    print(len(X_train))
    for k in range(len(X_train) - series_len):
        group_data = []
        kp_data = None
        for g in range(series_len):
            group_data.append(X_train[k + g])
            kp_data = int(float((y_train[k + g])))
        X_train_series.append(group_data) 
        y_train_series.append(kp_data)
    # print(len(X_train_series), len(y_train_series))
    # print(X_train_series[0], y_train_series[0])
    X_train_series = np.array(X_train_series)
    print('X_train_series.shape:', X_train_series.shape)
    X_train_series = X_train_series.reshape(X_train_series.shape[0], X_train_series.shape[1], X_train_series.shape[2])
    print('X_train_series.shape:', X_train_series.shape)
    return [np.array(X_train_series), np.array(y_train_series)]

def get_data_from_file(file_name, verbose=False):
    # log('Loading data from file:', file_name, verbose=verbose)
    data = pd.read_csv(file_name)
    return data

def get_symh_data_from_file(num_hours,
                            data_type='train',
                            symh_data_dir='solar-wind-data-symh',
                            columns_names =['Field_magnitude_average','BX_GSE_GSM','BY_GSE','BZ_GSE','BY_GSM','BZ_GSM','Speed','Proton_Density','Proton_Temperature','Flow_pressure','Electric_field'],
                            scale_down=False,
                            scale_up=False,
                            sample_size=None,
                            test_sample=None,
                            normalize_data =False,
                            skip_data=False,
                            random_selection=False,
                            data_filter=None,
                            ):
    # symh_data_dir='solar-wind-data-symh'
    interval_type='hourly'
    s = interval_type[0]
    num_hours = str(num_hours) 
    data_file_full = 'solar_wind_parameters_data_' + str(num_hours) + '_' + interval_type + '_' + str(data_type)+'.csv'    
    day_dir = interval_type[0]
    all_file =  data_dir + os.sep + symh_data_dir + os.sep + num_hours + day_dir + os.sep + data_file_full
    print('Loading from file:', all_file)
    all_data = pd.read_csv(all_file)
    if data_filter is not None:
        print('Appling filter YEAR is in:', data_filter)
        all_data = all_data.loc[all_data['YEAR'].isin(data_filter)].reset_index()   
    if sample_size is not None:
        # all_data = all_data.sample(sample_size)
        print('Loading sample sample:' ,sample_size)
        all_data = all_data[:sample_size].reset_index()
    data_years = list(set(all_data['YEAR'].values))
    data_years.sort()
    print('data_years:', data_years)      
    return all_data

def get_dates_from_data(data, date_col='Timestamp'):
    x_dates = []        
    # x_dates = list(test_data_all['YEAR'].values)
    for i in range (len(data)):
        x_dates.append(get_date_from_days_year_split(data['DOY'][0], data['YEAR'][0]))
    return x_dates

def get_good_symh_data_new_from_file(num_hours, interval_type,symh_data_dir, sym_col ='SYM_H',
                                     columns_names =['Field_magnitude_average','BX_GSE_GSM','BY_GSE','BZ_GSE','BY_GSM','BZ_GSM','Speed','Proton_Density','Proton_Temperature','Flow_pressure','Electric_field'],
                                     scale_down=False,
                                     scale_up=False,
                                     sample_size=None,
                                     test_sample=None,
                                     normalize_data =False,
                                     skip_data=False,
                                     random_selection=False,
                                     data_filter=None,
                                     train_data_filter=None
                                     ):
    s = interval_type[0]
    train_file_name = 'solar_wind_parameters_data_' + str(num_hours) + '_' + interval_type + '_train.csv'
    test_file_name = 'solar_wind_parameters_data_' + str(num_hours) + '_' + interval_type + '_test.csv'
    data_file_full = 'solar_wind_parameters_data_' + str(num_hours) + '_' + interval_type + '_all.csv'    
    num_hours = str(num_hours) 
    # print('Working on', interval_type[0],'#:', num_hours)
    day_dir = interval_type[0]

    tr_file = data_dir + os.sep + symh_data_dir + os.sep + num_hours + day_dir + os.sep + train_file_name
    ts_file = data_dir + os.sep + symh_data_dir + os.sep + num_hours + day_dir + os.sep + test_file_name
    all_file =  data_dir + os.sep + symh_data_dir + os.sep + num_hours + day_dir + os.sep + data_file_full
    print('Loading from file:', tr_file)
    all_data = pd.read_csv(tr_file)
    print('')
    
    # all_data[sym_col] = [k//10 for k in all_data[sym_col].values]
    # print('all_data len before:', len(all_data)) 
    # print('all_data.columns:', all_data.columns)

    # all_data = all_data.loc[all_data['YEAR'] != 2022].reset_index()

    
    print('Loading test from file:', ts_file)
    test_data_all = pd.read_csv(ts_file, dtype=None)
    if test_sample is not None:
        test_data_all = test_data_all[len(test_data_all) - test_sample:].reset_index()
    # if sample_size is not None:
    #     # all_data = all_data.sample(sample_size)
    #     all_data = all_data[len(all_data) - sample_size:].reset_index()
    if random_selection:
        print('Loading the entire full data for random selection from file:', data_file_full)
        full_data = pd.read_csv(all_file)
        
        full_data[sym_col] = full_data[sym_col].shift(-1 * (int(num_hours)*5))
        full_data = full_data.dropna()
        index = full_data[sym_col].index[full_data[sym_col].apply(np.isnan)]
        df_index = full_data.index.values.tolist()
        print('nans:', [df_index.index(i) for i in index])
        full_data = full_data.rename(columns={'YYYY':'YEAR'})
        print(list(full_data[sym_col].values)[1])
        print('data_filter:', data_filter)
        a_years=list(set(list((full_data['YEAR'].values))))
        a_years.sort()
        print('all years:', a_years)
        if data_filter is not None:
            full_data = full_data.loc[full_data['YEAR'].isin(data_filter)].reset_index()
        all_data, test_data_all = train_test_split(full_data, test_size=0.01, random_state=None)
        print('Columns:', full_data.columns)
        t_years=list(set(list((all_data['YEAR'].values))))
        t_years.sort() 
        te_years=  list(set(list((test_data_all['YEAR'].values))))
        te_years.sort()
        print('train years:', len(all_data) , t_years)
        print('test  years:', len(test_data_all), te_years)
        
        
    all_data = all_data[:]
    if train_data_filter is not None:
        print('Applying filter to training data YEAR is in:', train_data_filter)
        all_data = all_data.loc[all_data['YEAR'].isin(train_data_filter)].reset_index()
    if 'index' in all_data.columns:
        all_data = all_data.drop('index', axis=1)     
    if sample_size is not None:
        # all_data = all_data.sample(sample_size)
        all_data = all_data[:sample_size].reset_index()        
    print('len(test_data_all):',len(test_data_all))
    test_years = list(set(list( test_data_all['YEAR'].values)))
    train_years = list(set(list(all_data['YEAR'].values)))
    
    test_years.sort() 
    train_years.sort()
    print('Train years:', train_years) 
    print('Test years:' , test_years)    
    if scale_down:
        test_data_all[sym_col] = [float(v)/10 for v in test_data_all[sym_col]]
        all_data[sym_col] = [float(v)/10 for v in all_data[sym_col]]
    print('train min:',np.array(all_data[sym_col].values).min(), 'max:', np.array(all_data[sym_col].values).max())
    print('test  min:',np.array(test_data_all[sym_col].values).min(), 'max:', np.array(test_data_all[sym_col].values).max())        
    if scale_up:
        min_val_test = np.array(test_data_all[sym_col].values).min()
        min_val_train = np.array(all_data[sym_col].values).min()
        print('min_val test:', min_val_test)
        test_data_all[sym_col] = (test_data_all[sym_col] - min_val_train)//10
        
        min_val = np.array(all_data[sym_col].values).min()
        print('min_val train:', min_val_train)
        all_data[sym_col] = (all_data[sym_col] - min_val_train)//10
    if normalize_data:
        test_data_all[sym_col] = (test_data_all[sym_col] - np.array(test_data_all[sym_col]).min())/ (np.array(test_data_all[sym_col]).max() - np.array(test_data_all[sym_col]).min()) 
        all_data[sym_col] = (all_data[sym_col] - np.array(all_data[sym_col]).min())/ (np.array(all_data[sym_col]).max() - np.array(all_data[sym_col]).min()) 
        
    print('train min:',np.array(all_data[sym_col].values).min(), 'max:', np.array(all_data[sym_col].values).max())
    print('test  min:',np.array(test_data_all[sym_col].values).min(), 'max:', np.array(test_data_all[sym_col].values).max())
    # test_data_all[sym_col] = [k//10 for k in test_data_all[sym_col].values]
    # print('total size:'   , (len(test_data_all) + len(all_data)))
    # print('test_data_all[Timestamp][0]', test_data_all['Timestamp'][0])
    # print('test_data_all[Timestamp][last]', test_data_all['Timestamp'][len(test_data_all)-1])

    # test_data = test_data_all.loc[test_data_all['Timestamp'].str.contains('|'.join(['2021-7','2021-8','2021-9']))].reset_index()
    # test_filter = ['2021-10-' + str(i) + '-' for i in range(1, 32)]
    # test_filter.extend(['2021-11-' + str(i) + '-' for i in range(1, 31)])
    # test_filter =['2022-']
    # test_data = test_data_all.loc[test_data_all['Timestamp'].str.contains('|'.join(test_filter))].reset_index()
    test_data = test_data_all.reset_index()
    all_data = all_data.reset_index()
    if verbose:
        log('test_data.max:', np.array(test_data[sym_col].values).max())
        log('test_data.min:', np.array(test_data[sym_col].values).min())
        log('1 test_data[Timestamp][0]', test_data['Timestamp'][0])
        log('1 test_data[Timestamp][last]', test_data['Timestamp'][len(test_data) - 1])    
    orig_y_test = test_data[sym_col].values
    # data_2021 = test_data_all.loc[test_data_all['Timestamp'].str.contains('|'.join(['2021-' + str(i) for i in range(1,9)]))]
    # print('data_2021[Timestamp][0]', data_2021['Timestamp'][0])
    # print('data_2021[Timestamp][last]', data_2021['Timestamp'][len(data_2021)-1])
    if verbose:
        log('all_data.columns:', all_data.columns)
    # data_2021 = test_data_all.loc[~test_data_all['Timestamp'].isin(test_filter)]
    # all_data = pd.concat([all_data, data_2021])
    # all_data.sort_values(by=['Timestamp'])
    # print('all_data.size:', len(data_2021))
    # all_data  = all_data.reset_index()
        print('all_data[Timestamp][0]', all_data['Timestamp'][0])
    # print('all_data[Timestamp][last]', all_data['Timestamp'][len(all_data)-1])  
        
    cols = all_data.columns 
    # features = ['B_IMF', 'B_GSE', 'B_GSM', 'SW_Temp', 'SW_Speed', 'P_Pressure', 'E_Field']
    # columns_names  =['Scalar_B',  'BZ_GSE', 'SW_Plasma_Temperature',  'SW_Proton_Density','SW_Plasma_Speed', 'Flow_pressure', 'E_elecrtric_field']

    features = columns_names
    f_index = sym_col
    # print(features, sym_col)    
    
    norm_data = all_data[f_index]
    fig_optional_name = ''
    
    # train_percent = int(float(80. / 100. * len(all_data))) 
    # test_val_precent = int((len(all_data) - train_percent) / 2) - 50
    # print('train_precent:', train_percent, 'validate:', test_val_precent, 'test:', test_val_precent)
    
    train_data = all_data[:]
    
    train_data = clean_filled_values(train_data,columns_names)
    test_data = clean_filled_values(test_data,columns_names)
    train_percent = int(float(70. / 100. * len(all_data))) 
    test_val_precent = int((len(all_data) - train_percent) / 2) - 50    
        
#     print(train_data)
    valid_data = all_data[train_percent:-test_val_precent]
    
    # print('size of the test_data_all:', len(test_data_all))
    # print('len(train_data):', len(train_data), 'len(valid_data):', len(valid_data), 'len(test_data):', 
          # len(test_data),
          # 'len(orig_y_test):', len(orig_y_test))

    X_train = train_data[features].values
    X_train = reshape_x_data(X_train)
    # print('X_train.shape:', X_train.shape)
    y_train = reshape_y_data(train_data[f_index][:].values)
    
    X_valid = valid_data[features].values
    X_valid = reshape_x_data(X_valid)
    y_valid = reshape_y_data(norm_data[train_percent:-test_val_precent])
    
    X_test = test_data[features].values
    print('len(X_test):', len(X_test))
    X_test = reshape_x_data(X_test)

    # print('X_test size:', len(X_test))
#     y_test = reshape_y_data(norm_data[train_percent + test_val_precent:])
    y_test = reshape_y_data(test_data[f_index])
    # print('y_test len:', len(y_test))
    orig_y_test = reshape_y_data(orig_y_test)
    y = test_data['YEAR'][0]
    d = test_data['DOY'][0]
    h = test_data['HR'][0]
                
    y1 = test_data['YEAR'][len(test_data) - 1]
    d1 = test_data['DOY'][len(test_data) - 1]
    h1 = test_data['HR'][len(test_data) - 1]
    d = get_date_from_days_year_split(d, y)
    # print('d:', d)
    x_dates = []        
    # x_dates = list(test_data_all['YEAR'].values)
    for i in range (len(test_data)):
        x_dates.append(get_date_from_days_year_split(test_data['DOY'][0], test_data['YEAR'][0]))
    # x_dates=list(set(x_dates))
    # print('x_dates:', x_dates)
    if skip_data:
        l = len(X_train)
        a = []
        for i in range(0,l,10):
            a.append(i)
        print('len:', l, 'len(a):', len(a)) 
        X_train = np.array(X_train[a])
        y_train = np.array(y_train[a])
        # for b in a:
        #     print(b)
    return [ X_train, y_train, X_test, y_test, X_valid, y_valid, x_dates]




def get_date_from_days_year(d, y):
    return datetime.strptime('{} {}'.format(d, y), '%j %Y')


def get_date_from_days_year_split(d, y):
    date = get_date_from_days_year(d, y)
    return [date.year, date.month, date.day]

    
def reshape_x_data(data):
    data = [ np.array(c).reshape(len(c), 1) for c in data]
    data = np.array(data)
    data = data.reshape(data.shape[0], 1, data.shape[1])
    return data


def reshape_y_data(data):
    data = [ np.array(c) for c in data]
    data = np.array(data)
    data = data.reshape(data.shape[0], 1)
    return data



def plot_figure(storm_to_test, 
                resolution_minutes,
                view_type,
                x, 
                y_test, 
                y_preds_mean, 
                y_preds_var, 
                num_hours, 
                label='',
                file_name=None, 
                block=True, 
                do_sdv=True, 
                process_y_test=False,
                show_fig=False,
                return_fig=False,
                figsize=None,
                interval='h', 
                denormalize=False, 
                norm_max=1, 
                norm_min=0, 
                boxing=True, 
                wider_size=False,
                observation_color='#FF5050', 
                uncertainty_label='Aleatoric Uncertainty',
                fill_graph=False,
                uncertainty_margin=1,
                uncertainty_color='blue',
                x_labels=True,
                x_label='',
                scale_down=False,
                ylimit_min=None,
                ylimit_max=None,
                verbose=True,
                prediction_color='yellow',
                y_preds_var1=0,
                uncertainty_color1='black',
                yticks=None,
                xticksformat=None,
                xticksvalues=None,
                yticklabels=None,
                uncertainty_ep_margin=1,
                xticks=None,
                dateformat=None,
                prediction_errors=None,
                prediction_errors_color='blue',
                add_grid=True,
                is_legend=False):
    linewidth = 1
    markersize = 1
    marker = None
    linestyle = 'solid'
    prediction_linestyle='dashed'
    uncertainty_margin, uncertainty_ep_margin = uc_margins(storm_to_test, num_hours,resolution_minutes,view_type)    
    if wider_size:
        figsize = (8.4, 4.8)
        
    fig, ax = plt.subplots(figsize=figsize)
    if process_y_test:
        y_test = list(np.array((list(y_test)))[0,:, 0])
    if is_legend:
        ax.plot(x, y_test,
                label='Observed SYM-H',
                linewidth=linewidth,
                markersize=markersize,
                marker=marker,
                linestyle=linestyle,
                color=observation_color
                )  
        ax.plot(x, y_preds_mean ,
                label='Predicted SYM-H',
                linewidth=linewidth,
                markersize=markersize,
                marker=marker,
                linestyle=prediction_linestyle,
                color=prediction_color)
                        
        if prediction_errors is not None:
            
            ax2 = ax.twinx()
            ax2.set_ylim(ylimit_min, ylimit_max)
            if num_hours in [2,4,6]:
                print('')
            else:
                ax2.yaxis.set_ticks([])        
            ax.plot(x, prediction_errors,
                    linewidth=0.2,
                    markersize=markersize,
                    marker=marker,
                    linestyle=linestyle,
                    color=prediction_errors_color,
                    label='Prediction Error'
                    )
            if add_grid:
                ax2.grid(which='both',linewidth=0.3,linestyle='dashed')
                ax.grid(which='both',linewidth=0.3,linestyle='solid') 
    else:
        if prediction_errors is not None:
            ax2 = ax.twinx()
            ax2.set_ylim(ylimit_min, ylimit_max)
            if num_hours in [2,4,6]:
                print('')
            else:
                ax2.yaxis.set_ticks([])        
            ax.plot(x, prediction_errors,
                    linewidth=0.2,
                    markersize=markersize,
                    marker=marker,
                    linestyle=linestyle,
                    color=prediction_errors_color,
                    label='Prediction Error'
                    )
            if add_grid:
                ax2.grid(which='both',linewidth=0.3,linestyle='dashed')
                ax.grid(which='both',linewidth=0.3,linestyle='solid') 
        
        ax.plot(x, y_preds_mean ,
                label='Predicted SYM-H',
                linewidth=linewidth,
                markersize=markersize,
                marker=marker,
                linestyle=prediction_linestyle,
                color=prediction_color)
        ax.plot(x, y_test,
                label='Observed SYM-H',
                linewidth=linewidth,
                markersize=markersize,
                marker=marker,
                linestyle=linestyle,
                color=observation_color
                )
    ax.set_ylabel(label)
    
    if fill_graph:
        plt.fill_between(x, ((y_test ) - y_preds_var * uncertainty_ep_margin),
                             (y_preds_mean + y_preds_var * uncertainty_ep_margin),
                             color=uncertainty_color, alpha=0.2,label='Epistemic Uncertainty')

        plt.fill_between(x, (y_preds_mean - y_preds_var1 * uncertainty_margin),
                             (y_preds_mean + y_preds_var1 * uncertainty_margin),
                             color=uncertainty_color1, alpha=0.2, label='Aleatoric Uncertainty')        
        
    if scale_down:
        ax.set_ylim(-5, 10)
    log('x_label is:', x_label)
    plt.xlabel(x_label)
    
    if ylimit_min is not None and ylimit_max is not None:
        log('Setting ylimits...:', ylimit_min, ylimit_max)
        ax.set_ylim(ylimit_min, ylimit_max)
    label_y = label
    if label_y.startswith('F'):
        label_y = 'F10.7'
    plt.title(str(num_hours) + '' + interval + ' ahead prediction', fontsize=12, fontweight='bold')
    if not prediction_errors:
        if add_grid:
            ax2 = ax.twinx()
            ax2.set_ylim(ylimit_min, ylimit_max)
            
            ax2.grid(which='both',linewidth=0.3,linestyle='dashed')
            
            ax.grid(which='both',linewidth=0.3,linestyle='solid')
            xt = []
            if num_hours in [2,4,6]:
                xt =[]
            else:
                ax2.yaxis.set_ticks([])             
       
    else: 
        print('')
    if not  boxing:
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.xaxis.set_tick_params(direction='in')
    ax.yaxis.set_ticks_position('left')
    ax.yaxis.set_tick_params(direction='in')
    if len(x) <= 6:
        log('Setting x ticks to x:', x)
        ax.xaxis.set_ticks(x)

    if yticks is not None:
        ax.yaxis.set_ticks(yticks)
        ax.axes.get_yaxis().set_visible(False)

    if yticklabels is not None:
        log('setting the yticklabels:', yticklabels)
        if len(yticks) > 0:
            ax.set_yticklabels(yticklabels)
                
    ax.xaxis.set_major_locator(plt.MaxNLocator(len(xticksvalues)))
    log('xticksvalues in utils:', xticksvalues)
    log('xticks in utils1:', xticks)
    if xticks is not None:
        xticks= [str(xt) for xt in xticks]
        ax.set_xticks(xticks)
    ax.set_xticklabels(xticksvalues) 
    if is_legend:
        legend=ax.legend(ncol=4,fontsize="10", loc='upper center', bbox_to_anchor=(0.5, -0.05)) 
        for legend in legend.get_lines():
            legend.set_linewidth(6.0)
    if file_name is not None:
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        log('Saving figure to file:', file_name,verbose=True)
        plt.savefig(file_name, bbox_inches='tight')
        if str(file_name.strip()).endswith('.pdf'):
            plt.savefig(str(file_name).strip().replace('.pdf','.png'), bbox_inches='tight')
        if is_legend:
            export_legend(legend, filename=str(file_name).strip().replace('.pdf','_legend.pdf'), expand=[-2,-2,2,2])
    if return_fig:
        return plt
    if show_fig:
        plt.show(block=block)



def uc_margins(storm_to_test, h,resolution_minutes,view_type):
    log(storm_to_test, h,resolution_minutes,view_type)
    if storm_to_test == 37:
        uncertainty_ep_margin=0.7
        uncertainty_margin=4   
    if storm_to_test == 36:
        uncertainty_ep_margin=1
        uncertainty_margin=5                                                             
    if storm_to_test in [28,31,33,40,42]:
        uncertainty_margin=8
        uncertainty_ep_margin=0.9
        if storm_to_test == 31:
            uncertainty_margin=4
            uncertainty_ep_margin=0.5                                         
        if storm_to_test == 33:
            uncertainty_margin=2.5
            uncertainty_ep_margin=0.5                       


    return uncertainty_margin, uncertainty_ep_margin

def get_colors_from_cmap(cmap_name,n=3):
    from mycolorpy import colorlist as mcp
    import numpy as np
    colors=mcp.gen_color(cmap=cmap_name,n=n)
    return colors

def pad_progress_bar(n,d):
    n = str(n)
    d = str(d) 
    a = n + '/' + d 
    t = d + '/' + d 
    r = n + '/' + d 
    for c in range(len(t) - len(a)):
        r = ' ' + r
    return r 
    if  len(str(n)) == 5:
        return n + '  '
    if len(str(n)) == 6:
        return n + ' '
    return n

def uncertainty(model, X_test,col_index, N=100, metric='avg', verbose=0, scale_down=False):
    # predict stochastic dropout model T times
    p_hat = []
    print('Uncertainty Quantification, please wait while in progress..')
    aleatoric=[]
    epistemic = []
    for t in range(N):
        preds= model(X_test, training = True)
        preds = preds[:, 0, col_index]
        if scale_down:
            for k in range(len(preds)):
                preds[k] = round(preds[k],3)
        p_hat.append(preds)            
        if verbose == 1 or verbose == True:
            print(pad_progress_bar(str(t+1), str(N)) ,
                  ' [===== Uncertainty Quantification ======]  - ',
                  pad_progress_bar(int(float(((t+1)/N)*100)),N), '%')
    p_hat = np.array(p_hat)
    preds = p_hat 
    # mean prediction
    prediction = np.mean(p_hat, axis=0)

    max_p=np.max(preds)
    min_p=np.min(preds)
    preds_min_p= preds-min_p
    max_p_preds = max_p - preds
    multiplication_term = (preds_min_p) * (max_p_preds)
    multiplication_term_mean = np.mean(multiplication_term, axis=0)
    aleatoric=(np.sqrt(multiplication_term_mean)) 
    epistemic=(np.mean(preds ** 2, axis=0) - np.mean(preds, axis=0) ** 2)
    epistemic = np.sqrt(epistemic) 
    return np.squeeze(prediction), np.std(np.squeeze(aleatoric)) *1, np.std(np.squeeze(epistemic)) *1, p_hat


def get_range(d1,d2):
    tokens1=d1.split('-')
    if '/' in d1:
        tokens1 = d1.split('/')
        
    tokens2=d2.split('-')
    if '/' in d2:
        tokens2 = d2.split('/')
    year1=int(tokens1[0])
    month1=int(tokens1[1])
    day1=int(tokens1[2])
    
    year2=int(tokens1[0])
    month2=int(tokens2[1])
    day2=int(tokens2[2])
    
    t1 = datetime(year1,month1,day1,0,0,0)
    t2 = datetime(year2,month2,day2,23,59,59)
    a=[]
    c_time = t1 
    while c_time <= t2:
        t = str(c_time.year) + '-' + str(c_time.month) + '-'  + str(c_time.day) +'-'
        # print(t)
        a.append(t)
        c_time = c_time + timedelta(days=1)
    return a



def format_date_filter(d):
    d = str(d)
    tokens = str(d).split('-')
    if '/' in d:
        tokens = str(d).split('/')
    a = []
    for t in tokens:
        if t.startswith('0'):
            t = t[1:]
        a.append(t)
    # print('-'.join(a))
    return '-'.join(a)

def get_num(n):
    n = str(n) 
    if n.startswith('0'):
        return n[1:]
    return n

def get_range_from_data(data):
    rng = []
    for r in range(len(data)):
        s_d = data['Start date'][r]
        e_d = data['End date'][r] 
        t1 = format_date_filter(s_d)
        t2 =  format_date_filter(e_d)
        rng.extend(get_range(t1,t2))        
        # print('training range:', value)
    rng = list(set(rng))
    rng.sort()
    return rng

def boolean(b):
    if b == None:
        return False
    b = str(b).strip().lower()
    if b in ['y','yes','ye','1','t','tr','tru','true']:
        return True 
    return False

create_log_file()

