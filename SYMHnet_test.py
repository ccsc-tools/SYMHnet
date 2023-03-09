import pandas as pd
import numpy as np
import os
import typing
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
try:
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
except Exception as e:
    print('')
    
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math
from math import sqrt
from datetime import datetime, timedelta
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
from tensorflow import keras
from tensorflow.keras import layers

from SYMHnet_utils import *
# from SYMHnet_model import SYMHnetModel
from SYMHnet_dataset import * 

num_hours = 1
interval_type = 'hourly'
epochs = 100
prev_weights = None
w_dir=None
supported_cols = ['SYM_H','SYMH']
stats_only=False
rmse_all = -1000
verbose  = True
model_verbose = 2
figsize=(7,2.5)
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

def get_r2(y_true, y_pred):
    mean_val = np.array(y_true).mean()
    s_true = 0
    s_predict = 0
    for i in range(len(y_true)):
        s_predict  = s_predict + math.pow( (y_true[i] - y_pred[i]),2)
        s_true = s_true + math.pow((y_true[i]  - mean_val),2)
    return round(1 - (s_predict/s_true),4)

def get_rmse(y_true, y_pred):
    s = 0
    si = 0
    for i in range(len(y_true)):
        s = s + math.pow( (y_true[i] - y_pred[i]),2)
    s = math.sqrt(s/len(y_true))
    return round(s,4)  
def relative_root_mean_squared_error(true, pred):
    num = np.sum(np.square(true - pred))
    den = np.sum(np.square(pred))
    squared_error = num/den
    rrmse_loss = np.sqrt(squared_error)
    return rrmse_loss

def test(start_hour, end_hour, res=''):
    col_name='SYM_H'
    if res == '':
        print('1-Minute Resolution: Starting the training for start_hour:', start_hour)
    else:
        print('5-Minute Resolution: Starting the training for start_hour:', start_hour)
    columns_names=['Field_magnitude_average','BX_GSE_GSM','BY_GSE','BZ_GSE','BY_GSM','BZ_GSM','Speed','Proton_Density','Flow_pressure','Electric_field',col_name]    
    storms_data = pd.read_csv('data'+os.sep  +'storms_list.csv')
    storm_years = list(storms_data['Start date'].values) 
    storm_years.extend(list(storms_data['End date'].values))
    storm_years = [int(format_date_filter(s).split('-')[0]) for s in storm_years]
    storm_years = list(set(storm_years))
    storm_years.sort() 

    train_storms = storms_data.loc[storms_data['Storm no.'] <= 20]
    if 'index' in train_storms.columns:
        train_storms = train_storms.drop('index',axis=1)
    train_storm_years = list(train_storms['Start date'].values) 
    train_storm_years.extend(list(train_storms['End date'].values))
    train_storm_years = [int(format_date_filter(s).split('-')[0]) for s in train_storm_years]
    train_storm_years = list(set(train_storm_years))
    train_storm_years.sort()
    training_range = get_range_from_data(train_storms)
    
    
    validation_storms = storms_data.loc[storms_data['Storm no.'].isin([s for s in range(21,26)])].reset_index()
    val_storm_years = list(validation_storms['Start date'].values) 
    val_storm_years.extend(list(validation_storms['End date'].values))
    val_storm_years = [int(format_date_filter(s).split('-')[0]) for s in val_storm_years]
    val_storm_years = list(set(val_storm_years))
    val_storm_years.sort()
    validation_range = get_range_from_data(validation_storms)
    test_storms = storms_data.loc[storms_data['Storm no.']>= 26].reset_index()
    test_range = get_range_from_data(test_storms)
    
    test_storm_years = list(test_storms['Start date'].values) 
    test_storm_years.extend(list(test_storms['End date'].values))
    test_storm_years = [int(format_date_filter(s).split('-')[0]) for s in test_storm_years]
    test_storm_years = list(set(test_storm_years))
    test_storm_years.sort()
        
    for k in range(start_hour,end_hour):
        postfix = res 
        if res == '':
            postfix = '_1min'
        else:
            postfix = '_5min'
        data_dir_to_save = 'data' + os.sep + 'solar_wind_symh' + postfix + os.sep + str(k) + 'h'
        os.makedirs(data_dir_to_save, exist_ok=True)
        epochs = 3
        print('Running testing for h =', k, 'hour ahead')
        num_hours = k
        dataset = SYMHnetDataset(num_hours=num_hours, columns_names=columns_names)
        rows, cols = (len(columns_names), len(columns_names))
        arr = [[1 for i in range(cols)] for j in range(rows)]
        route_distances = np.array(arr)

                    
        data_dir_to_save = 'data' + os.sep + 'solar_wind_symh' + postfix + os.sep + str(k) + 'h'
        # training_range.extend( validation_range)
        training_file = data_dir_to_save + os.sep + 'training_data_' + str(k) + 'h.csv'
        train_data =  get_data_from_file(training_file, verbose=verbose)
        storms_file = data_dir_to_save + os.sep + 'storms_data_' + str(k) + 'h.csv'
        storms_data =  get_data_from_file(storms_file, verbose=verbose)
        d = [train_data, storms_data]
        validation_file = data_dir_to_save + os.sep + 'validation_data_' + str(k) + 'h.csv'
        validation_data = get_data_from_file(validation_file, verbose=verbose)
        
        train_array = train_data[columns_names].to_numpy()
        validation_array = validation_data[columns_names].to_numpy()
        
        train_dataset, val_dataset = (
            dataset.create_tf_dataset(data_array)
            for data_array in [train_array, validation_array]
        )
        if not 'level_0' in  test_storms.columns:
            test_storms = test_storms.reset_index()
            
        for itr in range(1):
            if res == '':
                res = '1min'
            else:
                res = '5min'
            w_dir = 'models_storms_' + res  +os.sep  + str(num_hours) + 'h_' + str(col_name).replace('_','').lower()
            model = keras.models.load_model(w_dir + os.sep + 'model_weights_full')

            ylimit_min=None 
            ylimit_max = None            
            for i in range(len(test_storms)):
                storm_num = (i+26)
                if res == '':
                    res = '1min'
                result_data_set_file = 'results_datasets' + os.sep + 'storm_num_' + str(storm_num) + os.sep + res 
                os.makedirs(result_data_set_file, exist_ok=True) 
                result_data_set_file = result_data_set_file + os.sep + str(start_hour) + 'h_uq_training.csv'
                if  storm_num in [36]:
                        ylimit_min=-170
                        ylimit_max=70  
                        xticksvalues=['1/18/2004' ,'1/20/2004', '1/22/2004', '1/24/2004' ,'1/26/2004']
                        if storm_num in [37]:
                            ylimit_min=None
                            ylimit_max=None
                else:
                    continue                                     
                storm_start = test_storms['Start date'][i]
                storm_end = test_storms['End date'][i]
                t1 = format_date_filter(storm_start)
                t2 =  format_date_filter(storm_end)
                log(storm_start, t1, storm_end,t2)
                value = get_range(t1,t2)
                s_data = storms_data.loc[storms_data['Timestamp'].str.contains('|'.join(value)) ]
                if 'level_0' in s_data.columns:
                    s_data = s_data.drop('level_0',axis=1)
                
                s_data = s_data.reset_index()
                x_dates = get_dates_from_data(s_data)
                s_data = s_data[columns_names].to_numpy()
                
                ax_dates = []
                start_date = x_dates[0]
                end_date = x_dates[-1]
                start_date_ts = datetime(int(start_date[0]), int(start_date[1]), int(start_date[2]),0,0,0)
                end_date_ts = datetime(int(end_date[0]), int(end_date[1]), int(end_date[2]),0,0,0)
                
                log('start_date_ts:', start_date_ts)
                test_data = s_data
                current_date = start_date_ts
                minutes = 5
                if res == '':
                    minutes = 1
                for c in range(len(test_data)-1):
                    ax_dates.append(current_date)
                    current_date = current_date + timedelta(minutes=minutes)
                end_date_ts = ax_dates[-1]
                log('end_date_ts:', end_date_ts)
            
                test_array = s_data 
                test_dataset = dataset.create_tf_dataset(
                    test_array,
                    batch_size=test_array.shape[0]
                )
            
                x_test, y = next(test_dataset.as_numpy_iterator())
                col_index = len(columns_names)-1
                for l in range(1):
                    print('Running prediction..')
                    y_pred = model.predict(x_test)
            
                    y_preds = y_pred[:, 0, col_index]
                    y_test = y[:, 0, col_index]        
                    predictions_ft = uncertainty(model, x_test,col_index, N=100, verbose=False)
                    
                    y_preds = predictions_ft[0]
            
                    corr, _ = pearsonr( y_preds, y_test)
                    corr = round(corr,4)
                    rmse = round(sqrt(mean_squared_error(y_test, y_preds)),4)
                    r2 = round(r2_score(y_test, y_preds),4)
                    r2_calc = round(get_r2(y_test,y_preds),4)
                    mae= round(mean_absolute_error(y_test,y_preds),4) 
                    RMSE = round(rmse,4)
                    rrmse = round(relative_root_mean_squared_error(y_test,y_preds),4)
                    r = np.corrcoef(y_test, y_preds)
                    w_dir = 'models_storms_' + res  +os.sep  + str(num_hours) + 'h_' + str(col_name).replace('_','').lower() 
                    handler = open(result_data_set_file,'w')
                    handler.write('Date,Actual,Prediction,Aleatoric,Epistemic\n')
                    for z in range(len(y_test)):
                        if z == 0:
                            handler.write(str(ax_dates[z]) + ',' +  str(y_test[z] ) + ',' + str(y_preds[z]) + ',' + str(predictions_ft[1]) + ',' + str(predictions_ft[2]) + '\n')
                        else:
                            handler.write(str(ax_dates[z]) + ','+ str(y_test[z] ) + ',' + str(y_preds[z]) + '\n')
                    handler.flush()
                    handler.close() 
                file_name_aleatoric = 'figures' + os.sep +'storm_' +str(res).replace('_','') + '_' + str(storm_num) +'_' + str(num_hours) + interval_type[0] +'_' + str(col_name).replace('_','').lower()  + '_aleatoric.pdf'
                file_name_epistemic ='figures' + os.sep + 'storm_' + str(res).replace('_','') + '_' +str(storm_num) +'_' + str(num_hours) + interval_type[0] +'_' + str(col_name).replace('_','').lower()  + '_epistemic.pdf'
                file_name_uq ='figures' + os.sep + 'storm_' +str(res).replace('_','') + '_' + str(storm_num) +'_' + str(num_hours) + interval_type[0] +'_' + str(col_name).replace('_','').lower()  + '_uq.pdf'
                              
                plot_figure(ax_dates,y_test,predictions_ft[0],predictions_ft[1],num_hours,label=str(col_name).replace('_','-') + ' index',
                            file_name=file_name_uq ,wider_size=False,figsize = figsize,
                            interval=interval_type[0],uncertainty_label='Aleatoric Uncertainty', fill_graph=True,
                            # uncertainty_color='#aabbff',
                            x_labels=True,
                            x_label='',
                            scale_down=False,
                            uncertainty_color='black',
                            prediction_color='yellow',
                            observation_color='#FF5050',
                            xticksvalues=xticksvalues,
                            y_preds_var1=predictions_ft[2],
                            ylimit_min=ylimit_min,
                            ylimit_max=ylimit_max,
                            uncertainty_margin=4.5,
                            uncertainty_ep_margin=1.2                                                 
                            )
    
if __name__ == '__main__':
    
    col_name = str('SYM_H').upper()
    col_name = col_name.replace('_','')
    col_name = col_name[:-1] +'_' + col_name[-1]
    if not col_name in supported_cols:
        print('Invalid column name:', col_name,'must be one of:', supported_cols)
        exit()
    res = ''
    if float(int(sys.argv[1])) == 1:
        res = ''
    else:
        res = '_5min'
    starting_hour = 1
    ending_hour = 7
    hours_list = []
    hours_list = list(sys.argv[2].split(','))
    hours_list = [int(h) for h in hours_list]
    for h in hours_list:
        starting_hour = h
        ending_hour = h + 1
        # print('Starting hour:', starting_hour, 'ending hour:', ending_hour-1)
        test(starting_hour, ending_hour,col_name,res=res) 
    plt.show()   
