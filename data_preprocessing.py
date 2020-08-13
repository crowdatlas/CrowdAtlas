import pandas as pd
import multiprocessing as mp
import datetime
import time
import os
import sys


def station_transform(o_station):
    n_station = o_station

    if o_station == 'DT32':
        n_station = 'EW2'
    if o_station == 'CC9':
        n_station = 'EW8'
    if o_station == 'DT14':
        n_station = 'EW12'
    if o_station == 'NE3':
        n_station = 'EW16'
    if o_station == 'CC22':
        n_station = 'EW21'
    if o_station == 'CC15':
        n_station = 'NS17'
    if o_station == 'DT11':
        n_station = 'NS21'
    if o_station == 'CC1':
        n_station = 'NS24'
    if o_station == 'CE2':
        n_station = 'NS27'
    if o_station == 'EW24':
        n_station = 'NS1&EW24'
    if o_station == 'EW13':
        n_station = 'NS25&EW13'
    if o_station == 'EW14':
        n_station = 'NS26&EW14'

    return n_station


n_stations = 52
df_tvl = pd.read_csv('travel_time.csv', index_col=0)
df_trans_tvl = pd.read_csv('EW_NS_trans_travel_time.csv', index_col=0)
df_interchange = pd.read_csv('interchange_stations.csv')
station_list = ['NS1&EW24', 'NS25&EW13', 'NS26&EW14'] + ['EW'+str(q) for q in range(1,13)] \
               + ['EW'+str(q) for q in range(15,24)] + ['EW'+str(q) for q in range(25,30)] \
               + ['NS'+str(q) for q in range(2,6)] + ['NS'+str(q) for q in range(7,12)] \
                + ['NS'+str(q) for q in range(13,25)] + ['NS'+str(q) for q in range(27,29)]

tvl_time_array = [[0 for j in range(n_stations)] for i in range(n_stations)]
for i in range(0, n_stations):
    for j in range(0, n_stations):
        tvl_time_array[i][j] = int(df_trans_tvl.loc[station_list[i], station_list[j]])


def divide_by_date_station(date):
    date_str = datetime.datetime.strftime(date, '%Y-%m-%d')
    next_day=datetime.datetime(date.year,date.month,date.day+1).strftime('%Y-%m-%d')
    df_remove = pd.DataFrame(columns=['start_station', 'start_time', 'end_station', 'end_time', 'start_date', 'end_date'])
    n_remove = 0
    n_same_od = 0
    n_err_tvl_time = 0
    n_long_stay = 0
    n_date = 0

    for k in range(0, 52):
        df = pd.read_csv('EZ_Link/' + station_list[k] + '_start.csv')
        df_opt1 = df[(df['Ride_end_date'] == date_str) & (df['Ride_end_time'] > '05:00:00')]
        df_opt2 = df[(df['Ride_end_date'] == next_day) & (df['Ride_end_time'] < '01:00:00')]
        df_opt = df_opt1.append(df_opt2, ignore_index=True)

        df_record = pd.DataFrame(columns=['origin_station', 'start_time', 'destination_station', 'end_time', 'start_date', 'end_date'])
        n_stn = 0

        for i in range(0, df_opt.shape[0]):
            start_date = df_opt.iloc[i, 3]
            start_time = df_opt.iloc[i, 4]
            end_date = df_opt.iloc[i, 6]
            end_time = df_opt.iloc[i, 7]
            start_station = station_transform(df_opt.iloc[i, 2])
            end_station = station_transform(df_opt.iloc[i, 5])
            if ('EW' in start_station or 'NS' in start_station) and ('EW' in end_station or 'NS' in end_station):
                travel_time_border = int(df_trans_tvl.loc[start_station, end_station])
                travel_time = int(time.mktime(time.strptime(end_time, '%H:%M:%S')) \
                        - time.mktime(time.strptime(start_time, '%H:%M:%S')))
                if start_station == end_station or travel_time <= travel_time_border * 60 \
                        or (df_opt.iloc[i, 3] == date_str and df_opt.iloc[i, 4] < '05:00:00'):
                    df_remove.loc[n_remove]=[start_station, start_time, end_station, end_time, start_date, end_date]
                    n_remove=n_remove+1
                    if start_station == end_station:
                        n_same_od += 1
                    else:
                        if travel_time <= travel_time_border:
                            n_err_tvl_time += 1
                        else:
                            n_long_stay += 1
                else:
                    df_record.loc[n_stn] = [start_station, start_time, end_station, end_time, start_date, end_date]
                    n_stn = n_stn + 1

        n_date += n_stn

        df_record_output = df_record.sort_values(by='start_time')
        df_record_output.to_csv('dataset/' + date_str + '/' + station_list[k] + '.csv', index=None)

    df_remove_output=df_remove.sort_values(by=['start_station','end_station'])
    df_remove_output.to_csv('dataset/' + date_str + '/EW_NS_removed_records.csv', index=None)

    print('Date: ' + str(date) + ', total num: ' + str(n_date))
    print('Same OD num: ' + str(n_same_od) + ', error travel time num: ' + str(n_err_tvl_time) + ', long-time stay num: ' + str(n_long_stay))


if __name__ == '__main__':
    argv = sys.argv
    start_date = argv[1]    # Format: '2016-01-01', '1/1/2016' or '20160101'
    end_date = argv[2]  # Format: '2016-01-31', '1/31/2016' or '20160131'
    dates = pd.bdate_range(start_date, end_date)

    if not os.path.exists('dataset'):
        os.mkdir('dataset')
    for i in range(0, len(dates)):
        date_str = datetime.datetime.strftime(dates[i], '%Y-%m-%d')
        if not os.path.exists('dataset/' + date_str):
            os.mkdir('dataset/' + date_str)

    st = time.time()
    pool = mp.Pool(processes=6)
    pool.map(divide_by_date_station, [dates[i] for i in range(len(dates))])
    pool.close()
    end = time.time()
    print('Process Time cost:', int(end - st), 'seconds')