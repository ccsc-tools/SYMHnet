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
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
try:
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
except Exception as e:
    print('')
    
import math
from datetime import datetime, timedelta
from tensorflow import keras

from SYMHnet_utils import *
from SYMHnet_dataset import * 

num_hours = 1
interval_type = 'hourly'
epochs = 100
w_dir=None
is_small=True
verbose  = False
figsize=(7,3)
resolution_minutes=1
view_type = ''

def check_test_params(storm_to_test,start_hour,resolution_minutes,view_type):
    if not resolution_minutes in [1,5]:
        print('Invalid resolution:', resolution_minutes)
        print('Supported resolution: 1 or 5')
        exit()
    if not storm_to_test in supported_test_storms:
        print('Invalid or unsupported test storm:', storm_to_test)
        print('Supported storms are:', supported_test_storms)
        exit()
    
    if not str(view_type).strip().lower() in ['' , '_lv']:
        print('Invalid localization view type:', view_type) 
        print('Supported localization view type: _lv or blank')
        exit()  
              
def test_storm(storm_to_test,
         start_hour,
         end_hour, 
         resolution_minutes=1, 
         do_pred_error=False,
         view_type = ''):
    test(storm_to_test,
            start_hour,
            end_hour, 
            resolution_minutes=resolution_minutes, 
            do_pred_error=do_pred_error,
            view_type = view_type,
            jupyter_enabled=True)    
def test(storm_to_test,
         start_hour,
         end_hour, 
         resolution_minutes=1, 
         do_pred_error=False,
         view_type = '',
         jupyter_enabled=False):
    figure_names = []
    fig_ax = None
    check_test_params(storm_to_test,start_hour,resolution_minutes,view_type)
    global figsize
    res = ''
    if resolution_minutes == 1:
        res = ''
    else:
        res = '_5min'

    if is_small:
        figsize=(5,2.2)
    
    log('Starting the training for start_hour:', start_hour, 'end_hour:', end_hour )
    storms_data = pd.read_csv('data'+os.sep  +'storms_list.csv')
    storm_years = list(storms_data['Start date'].values) 
    storm_years.extend(list(storms_data['End date'].values))
    storm_years = [int(format_date_filter(s).split('-')[0]) for s in storm_years]
    storm_years = list(set(storm_years))
    storm_years.sort() 

    test_storms = storms_data.loc[storms_data['Storm no.']>= 26].reset_index()
    test_range = get_range_from_data(test_storms)
    log('test_range:', test_range)
    
    test_storm_years = list(test_storms['Start date'].values) 
    test_storm_years.extend(list(test_storms['End date'].values))
    test_storm_years = [int(format_date_filter(s).split('-')[0]) for s in test_storm_years]
    test_storm_years = list(set(test_storm_years))
    test_storm_years.sort()
    log('test_years:', test_storm_years)
    fig = None 
    ax = None 
    for k in range(start_hour,end_hour):
        if jupyter_enabled and  k == 1:
            fig, ax = plt.subplots(1, 2,figsize=figsize) 
            # fig_ax = [fig,ax]       
        postfix = '_' + str(resolution_minutes) + 'min'
        data_dir_to_save = 'data' + os.sep + 'solar_wind_symh_' + str(resolution_minutes) + 'min' + os.sep + str(k) + 'h'
        os.makedirs(data_dir_to_save, exist_ok=True)
        epochs = 3
        print('Running testing for storm #' +str(storm_to_test) + ' for ', str(k) +'-hour ahead for', str(resolution_minutes)  +'-minute resolution')
        num_hours = k
        dataset = SYMHnetDataset(num_hours=num_hours, columns_names=columns_names)
        rows, cols = (len(columns_names), len(columns_names))
        arr = [[1 for i in range(cols)] for j in range(rows)]
        route_distances = np.array(arr)

                    
        data_dir_to_save = 'data' + os.sep + 'solar_wind_symh' + postfix + os.sep + str(k) + 'h' +  os.sep + str(storm_to_test) 
        training_file = data_dir_to_save + os.sep + 'training_data_' + str(k) + 'h.csv'
        train_data =  get_data_from_file(training_file, verbose=verbose)
        storms_file = data_dir_to_save + os.sep + 'storms_data_' + str(k) + 'h.csv'
        storms_data =  get_data_from_file(storms_file, verbose=verbose)
        d = [train_data, storms_data]
        validation_file = data_dir_to_save + os.sep + 'validation_data_' + str(k) + 'h.csv'
        validation_data = get_data_from_file(validation_file, verbose=verbose)

        log('len(train_data):', len(train_data), verbose=verbose)
        
        train_array = train_data[columns_names].to_numpy()
        validation_array = validation_data[columns_names].to_numpy()
        log('len(train_array):', len(train_array), verbose=verbose)
        
        train_dataset, val_dataset = (
            dataset.create_tf_dataset(data_array)
            for data_array in [train_array, validation_array]
        )
        if k == 1:
            test_storms = test_storms.reset_index()
            
        res = str(resolution_minutes) + 'min'
        w_dir = 'models_storms_' + res  +os.sep  + str(num_hours) + 'h_' + str(col_name).replace('_','').lower()
        model = keras.models.load_model(w_dir + os.sep + 'model_weights_full')
        ylimit_min=None 
        ylimit_max = None            
        storm_num = storm_to_test
        i = storm_to_test - 26
        res = str(resolution_minutes) + 'min'
        result_data_set_file = 'results_datasets' + os.sep + 'storm_num_' + str(storm_num) + os.sep + res 
        os.makedirs(result_data_set_file, exist_ok=True) 
        result_data_set_file = result_data_set_file + os.sep + str(start_hour) + 'h_uq_training.csv'
        storm_start = test_storms['Start date'][i]
        storm_end = test_storms['End date'][i]
        

        t1 = format_date_filter(storm_start)
        t2 =  format_date_filter(storm_end)
        log('t1', t1)
        log('values-->', storm_start, t1, storm_end,t2)
        value = get_range(t1,t2)
        log('value:', value)
        if '_lv' in view_type :
            if storm_to_test == 36:
                if is_intense:
                    value = ['2004-1-22-' + str(n) for n in range(9,21)]
                else:
                    value = ['2004-1-18-' + str(n) for n in range(0,7)]
            if storm_to_test == 37:
                if is_intense:
                    value = ['2004-11-7-' + str(n) for n in range(12,24)]
                    for n in range(0,15):
                        value.append('2004-11-8-' + str(n))
                else:
                    value = ['2004-11-7-' + str(n) for n in range(0,3)]                      
        log('value 2:', value)
        s_data = storms_data.loc[storms_data['Timestamp'].str.contains('|'.join(value)) ]

        log(storm_num,'storms_data size:', len(storms_data), 's_data size:', len(s_data))
        log(storm_num,'min:',np.array(s_data['SYM_H']).min(),'max:', np.array(s_data['SYM_H']).max())
        max_val = np.array(s_data['SYM_H']).max()
        min_val = np.array(s_data['SYM_H']).min()
        log(storm_num,'range:', value)
        # if jupyter_enabled:
        #     s_data = s_data[-1000:]
        #     s_data = s_data[:-1000]
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
        
        test_data = s_data
        current_date = start_date_ts
        minutes = resolution_minutes

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
        y_pred = model.predict(x_test)

        y_preds = y_pred[:, 0, col_index]
        y_test = y[:, 0, col_index]        
        predictions_ft = uncertainty(model, x_test,col_index, N=numberOfSamples, verbose=verbose)
        
        y_preds = predictions_ft[0]
        w_dir = 'models_storms_' + str(resolution_minutes) +'min'  +os.sep  + str(num_hours) + 'h_' + str(col_name).replace('_','').lower() 
        handler = open(result_data_set_file,'w')
        log('saving to:', result_data_set_file)
        handler.write('Date,Actual,Prediction,Aleatoric,Epistemic\n')
        for z in range(len(y_test)):
            if z == 0:
                handler.write(str(ax_dates[z]) + ',' +  str(y_test[z] ) + ',' + str(y_preds[z]) + ',' + str(predictions_ft[1]) + ',' + str(predictions_ft[2]) + '\n')
            else:
                handler.write(str(ax_dates[z]) + ','+ str(y_test[z] ) + ',' + str(y_preds[z]) + '\n')
        handler.flush()
        handler.close() 
        file_name_uq ='figures' + os.sep + 'storm_' +str(res).replace('_','') + '_' + str(storm_num) +'_' + str(num_hours) + interval_type[0] +'_' + str(col_name).replace('_','').lower()  + '_uq' + view_type +'.pdf'
        
        l = len(ax_dates)
        log('dates-->', ax_dates[0],ax_dates[(l//4)], ax_dates[(l//4)*2], ax_dates[-1] )
        y_label = str(col_name).replace('_','-') + ' (nT)'
        yticks = None
        if is_small and num_hours in [2,4,6]:
            y_label = ''
            yticks = []
        pred_errors = None
        fill_graph=True
        if do_pred_error:
            fill_graph=False
            pred_errors= []
            for n in range(len(y_test)):  
                pred_errors.append(math.floor(((y_test[n] - predictions_ft[0][n]))))
            if len(pred_errors) > 0:
                file_name_uq = file_name_uq.replace(".pdf","_pe.pdf")
            log('Preds stats -->', storm_to_test,res, k, np.array(pred_errors).max() , np.array(pred_errors).min())
        xticks = [ ax_dates[0],ax_dates[(l//3)], ax_dates[(l//3)*2], ax_dates[-1]]
        if storm_num in [28, 31, 33,40,42]:
            ylimit_min=-550
            ylimit_max=150
            xticksvalues = [str(t1) for t1 in xticks]
            if storm_num == 33:
                ylimit_min=-550
                ylimi_max = 150
                xticksvalues=['03/26/01' ,'03/29/01', '04/01/01', '04/04/01']
            if storm_num == 28:
                ylimit_min = int(min_val) - 50
                ylimit_max = int(max_val) + 50 
                xticksvalues=['01/09/99' ,'09/07/99', '09/13/99', '09/19/99'] 
            if storm_num == 31:
                ylimit_min = int(min_val) - 50
                ylimit_max = int(max_val) + 50 
                xticksvalues=['04/02/00' ,'04/05/00', '04/08/00', '04/11/00']  
            if storm_num == 40:
                ylimit_min = int(min_val) - 50
                ylimit_max = int(max_val) + 50 
                xticksvalues=['06/26/13' ,'06/29/13', '07/02/13', '07/05/13']  
            if storm_num == 42:
                ylimit_min = -275
                ylimit_max = int(max_val) + 50 
                xticksvalues=['08/22/18' ,'08/26/18', '08/30/18', '09/03/18']  
            else:
                ylimit_min = int(min_val) - 50
                ylimit_max = int(max_val) + 50          
        elif  storm_num in [36]:
                ylimit_min=-170
                ylimit_max=70  
                xticksvalues=['01/18/04' ,'01/21/04', '01/24/04', '01/27/04']                       
                if '_lv' in view_type :
                    if is_intense:
                        xticksvalues=['01/22/04 09:00' ,'01/22/04 12:00', '01/22/04 15:00', '01/22/04 18:00']
                    else:
                        xticksvalues=['01/18/04 0:00' ,'01/18/04 02:00', '01/18/04 04:00', '01/18/04 06:00']

        elif  storm_num in [37]:
                ylimit_min=-170
                ylimit_max=70  
                xticksvalues=['11/4/2004' ,'11/7/2004', '11/11/2004', '11/14/2004']   
                xticksvalues=['11/04/04' ,'11/07/04', '11/10/04', '11/13/04'] 
                if '_lv' in view_type :
                    if is_intense:
                        xticksvalues=['11/07/04 21:00' ,'11/08/04 03:00', '11/08/04 09:00', '11/08/04 15:00']                                
                    else:
                        xticksvalues=['11/07/04 00:00' ,'11/07/04 02:00', '11/07/04 04:00', '11/07/04 06:00']                                          
                if storm_num in [37]:
                    ylimit_min=-450
                    ylimit_max=150                
        log('xticks in test:', xticks)
        ylimit_min = -450
        ylimit_max = 150
        ytickvalues = [-400, -300, -200, -100, 0, 100]                
        figure_names.append(file_name_uq)
        plot_figure(storm_to_test, resolution_minutes,view_type,
                    ax_dates,y_test,y_preds,
                    predictions_ft[1],
                    num_hours,
                    label=y_label,
                    file_name=file_name_uq ,
                    figsize = figsize,
                    fill_graph=fill_graph,
                    xticksvalues=xticksvalues,
                    xticks = xticks,
                    y_preds_var1=predictions_ft[2],
                    ylimit_min=ylimit_min,
                    ylimit_max=ylimit_max,
                    yticks=yticks,
                    prediction_errors=pred_errors,
                    jupyter_enabled=jupyter_enabled
                    )
    
if __name__ == '__main__':
    if len(sys.argv) < 5:
        print('You must provide all required input parameters\nUsage:')
        print('SYMHnet_test <storm to test: 26 to 42>  <resolution type: 1|5 > <prediction error True|False> <local view True|False>')
        print('For example to test storm 37 for 5-minute resolution with prediction error for local view:')
        print('SYMHnet_test 37 5 True True')
        exit()
    storm_to_test = int(float(int(sys.argv[1])))
    if not storm_to_test in supported_test_storms:
        print('Invalid or unsupported test storm:', storm_to_test)
        print('Supported storms are:', supported_test_storms)
        exit()
    resolution_minutes = int(float(int(sys.argv[2])))
    if not resolution_minutes in [1,5]:
        print('Invalid resolution:', resolution_minutes)
        print('Supported resolution: 1 or 5')
        exit()
        
    do_pred_error= boolean(sys.argv[3])
    view_type_bool= boolean(str(sys.argv[4]).strip().lower())
    if not view_type_bool :
        view_type = '';
    else:
        view_type = '_lv';
    res = ''
    if resolution_minutes == 1:
        res = ''
    else:
        res = '_5min'
        resolution_minutes = 5
    starting_hour = 1
    ending_hour = 2
    hours_list = [1,2]
    print('Start testing hour:', starting_hour, 'and hour:', ending_hour)
    for h in hours_list:
        starting_hour = h
        ending_hour = h + 2     
        test(storm_to_test, 
             starting_hour, 
             ending_hour,
             resolution_minutes=resolution_minutes, 
             do_pred_error=do_pred_error,
             view_type = view_type) 
        break        
    plt.show()   

