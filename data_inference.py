import numpy as np
import pandas as pd
import datetime
import sys
from sklearn.utils import shuffle


station_list = ['NS1&EW24', 'NS25&EW13', 'NS26&EW14'] + ['EW'+str(q) for q in range(1,13)] \
               + ['EW'+str(q) for q in range(15,24)] + ['EW'+str(q) for q in range(25,30)] \
               + ['NS'+str(q) for q in range(2,6)] + ['NS'+str(q) for q in range(7,12)] \
                + ['NS'+str(q) for q in range(13,25)] + ['NS'+str(q) for q in range(27,29)]

df_trans_tvl = pd.read_csv('travel_time.csv', index_col=0)
df_trans_stn = pd.read_csv('interchange_stations.csv', index_col=0)
stn_num = 52
tvl_time_array = [[0 for j in range(stn_num)] for i in range(stn_num)]
trans_stn_array = [['*' for q in range(stn_num)] for p in range(stn_num)]
for i in range(0, stn_num):
    for j in range(0, stn_num):
        tvl_time_array[i][j] = int(df_trans_tvl.loc[station_list[i], station_list[j]])
        trans_stn_array[i][j] = df_trans_stn.loc[station_list[i], station_list[j]]


def get_stn_index(station):
    if station == 'EW24' or station == 'NS1':
        station = 'NS1&EW24'
    if station == 'EW13' or station == 'NS25':
        station = 'NS25&EW13'
    if station == 'EW14' or station == 'NS26':
        station = 'NS26&EW14'

    if station == 'NS1&EW24':
        return 0
    if station == 'NS25&EW13':
        return 1
    if station == 'NS26&EW14':
        return 2

    ret = -1
    line = station[:2]
    num = int(station[2:])
    if line == 'EW':
        if num in range(1,13):
            ret = num + 2
        if num in range(15,24):
            ret = num
        if num in range(25,30):
            ret = num - 1
    else:
        if num in range(2,6):
            ret = num + 27
        if num in range(7,12):
            ret = num + 26
        if num in range(13,25):
            ret = num + 25
        if num in range(27,29):
            ret = num + 23

    return ret


def get_midway_time(line, start_station, end_station, rel_end_time):
    record_list = []
    gen_rel_end_time = 0
    end_stn_index = get_stn_index(line + str(end_station))
    if end_station > start_station:
        j = end_station - 1
        if line == 'NS' and j in [6, 12]:
            j -= 1
        while j >= start_station:
            midway_stn_index = get_stn_index(line + str(j))
            tvl_time = tvl_time_array[midway_stn_index][end_stn_index]
            gen_rel_end_time = rel_end_time - tvl_time
            if gen_rel_end_time > 0:
                record_list.append((midway_stn_index, gen_rel_end_time))
            j -= 1
            if line == 'NS' and j in [6, 12]:
                j -= 1
    else:
        j = end_station + 1
        if line == 'NS' and j in [6, 12]:
            j += 1
        while j <= start_station:
            midway_stn_index = get_stn_index(line + str(j))
            tvl_time = tvl_time_array[midway_stn_index][end_stn_index]
            gen_rel_end_time = rel_end_time - tvl_time
            if gen_rel_end_time > 0:
                record_list.append((midway_stn_index, gen_rel_end_time))
            j += 1
            if line == 'NS' and j in [6, 12]:
                j += 1

    return record_list, gen_rel_end_time


def data_inference(date, station, hour):
    df = pd.read_csv('dataset/' + date + '/' + station + '.csv')

    record_list = []
    for i in range(0, df.shape[0]):
        cur_start_time = datetime.datetime.strptime(df.iloc[i, 1], '%H:%M:%S')
        if cur_start_time.hour != hour:
            continue
        if cur_start_time.hour < 5:
            df.iloc[i, 1] = (24 + cur_start_time.hour) * 60 + int(cur_start_time.minute)
        else:
            df.iloc[i, 1] = cur_start_time.hour * 60 + int(cur_start_time.minute)
        cur_end_time = datetime.datetime.strptime(df.iloc[i, 3], '%H:%M:%S')
        if cur_end_time.hour < 5:
            df.iloc[i, 3] = (24 + cur_end_time.hour) * 60 + cur_end_time.minute
        else:
            df.iloc[i, 3] = cur_end_time.hour * 60 + cur_end_time.minute

        start_time = int(df.iloc[i, 1])
        rel_end_time = int(df.iloc[i, 3]) - start_time

        if rel_end_time <= 80:
            cur_record = [0] * (60 + 80 + 52 + 52)
            cur_record[(start_time - hour * 60)] = 1
            cur_record[60 + rel_end_time - 1] = 1
            cur_start_station = get_stn_index(station)
            cur_record[60 + 80 + cur_start_station] = 1
            cur_end_station = get_stn_index(df.iloc[i, 2])
            cur_record[60 + 80 + 52 + cur_end_station] = 1
            record_list.append(cur_record)

            start_stn = 0
            end_stn = 0
            midway_time_list = []
            if cur_start_station in range(0, 3) and cur_end_station in range(0, 3):
                stn_start_temp = df.iloc[i, 0]
                slash = stn_start_temp.index('&')
                start_stn = int(stn_start_temp[slash + 3:])
                stn_end_temp = df.iloc[i, 2]
                slash = stn_end_temp.index('&')
                end_stn = int(stn_end_temp[slash + 3:])
                line = 'EW'
                midway_time_list, _ = get_midway_time(line, start_stn, end_stn, rel_end_time)

            if cur_start_station in range(0, 3) and cur_end_station in range(3, 29):
                stn_start_temp = df.iloc[i, 0]
                slash = stn_start_temp.index('&')
                start_stn = int(stn_start_temp[slash + 3:])
                stn_end_temp = df.iloc[i, 2]
                end_stn = int(stn_end_temp[2:])
                line = 'EW'
                midway_time_list, _ = get_midway_time(line, start_stn, end_stn, rel_end_time)

            if cur_start_station in range(0, 3) and cur_end_station in range(29, 52):
                stn_start_temp = df.iloc[i, 0]
                slash = stn_start_temp.index('&')
                start_stn = int(stn_start_temp[2:slash])
                stn_end_temp = df.iloc[i, 2]
                end_stn = int(stn_end_temp[2:])
                line = 'NS'
                midway_time_list, _ = get_midway_time(line, start_stn, end_stn, rel_end_time)

            if cur_start_station in range(3, 29) and cur_end_station in range(0, 3):
                stn_start_temp = df.iloc[i, 0]
                start_stn = int(stn_start_temp[2:])
                stn_end_temp = df.iloc[i, 2]
                slash = stn_end_temp.index('&')
                end_stn = int(stn_end_temp[slash + 3:])
                line = 'EW'
                midway_time_list, _ = get_midway_time(line, start_stn, end_stn, rel_end_time)

            if cur_start_station in range(29, 52) and cur_end_station in range(0, 3):
                stn_start_temp = df.iloc[i, 0]
                start_stn = int(stn_start_temp[2:])
                stn_end_temp = df.iloc[i, 2]
                slash = stn_end_temp.index('&')
                end_stn = int(stn_end_temp[2:slash])
                line = 'NS'
                midway_time_list, _ = get_midway_time(line, start_stn, end_stn, rel_end_time)

            if cur_start_station in range(3, 29) and cur_end_station in range(3, 29):
                stn_start_temp = df.iloc[i, 0]
                start_stn = int(stn_start_temp[2:])
                stn_end_temp = df.iloc[i, 2]
                end_stn = int(stn_end_temp[2:])
                line = 'EW'
                midway_time_list, _ = get_midway_time(line, start_stn, end_stn, rel_end_time)

            if cur_start_station in range(29, 52) and cur_end_station in range(29, 52):
                stn_start_temp = df.iloc[i, 0]
                start_stn = int(stn_start_temp[2:])
                stn_end_temp = df.iloc[i, 2]
                end_stn = int(stn_end_temp[2:])
                line = 'NS'
                midway_time_list, _ = get_midway_time(line, start_stn, end_stn, rel_end_time)

            if cur_start_station in range(3, 29) and cur_end_station in range(29, 52):
                stn_start_temp = df.iloc[i, 0]
                start_stn = int(stn_start_temp[2:])
                stn_end_temp = df.iloc[i, 2]
                end_stn = int(stn_end_temp[2:])
                trans_stn = trans_stn_array[cur_start_station][cur_end_station]
                slash = trans_stn.index('&')
                line = 'NS'
                sel_trans_stn = int(trans_stn[2:slash])
                midway_time_list, new_rel_end_time = get_midway_time(line, sel_trans_stn, end_stn, rel_end_time)
                line = 'EW'
                sel_trans_stn = int(trans_stn[slash + 3:])
                midway_time_list2, _ = get_midway_time(line, start_stn, sel_trans_stn, new_rel_end_time)
                midway_time_list.extend(midway_time_list2)

            if cur_start_station in range(29, 52) and cur_end_station in range(3, 29):
                stn_start_temp = df.iloc[i, 0]
                start_stn = int(stn_start_temp[2:])
                stn_end_temp = df.iloc[i, 2]
                end_stn = int(stn_end_temp[2:])
                trans_stn = trans_stn_array[cur_start_station][cur_end_station]
                slash = trans_stn.index('&')
                line = 'EW'
                sel_trans_stn = int(trans_stn[slash + 3:])
                midway_time_list, new_rel_end_time = get_midway_time(line, sel_trans_stn, end_stn, rel_end_time)
                line = 'NS'
                sel_trans_stn = int(trans_stn[2:slash])
                midway_time_list2, _ = get_midway_time(line, start_stn, sel_trans_stn, new_rel_end_time)
                midway_time_list.extend(midway_time_list2)

            stab_time = rel_end_time
            stab_stn_index = cur_end_station

            for k in range(len(midway_time_list)):
                arv_stn_index = midway_time_list[k][0]
                arv_rel_time = midway_time_list[k][1]
                cur_record = [0] * (60 + 80 + 52 + 52)
                cur_record[(start_time - hour * 60)] = 1
                cur_record[60 + arv_rel_time - 1] = 1
                cur_record[60 + 80 + cur_start_station] = 1
                cur_record[60 + 80 + 52 + arv_stn_index] = 1
                record_list.append(cur_record)

                for u in range(arv_rel_time + 1, stab_time):
                    cur_record = [0] * (60 + 80 + 52 + 52)
                    cur_record[(start_time - hour * 60)] = 1
                    cur_record[60 + u - 1] = 1
                    cur_record[60 + 80 + cur_start_station] = 1
                    cur_record[60 + 80 + 52 + stab_stn_index] = 1
                    record_list.append(cur_record)

                stab_time = arv_rel_time
                stab_stn_index = arv_stn_index

            for u in range(1, stab_time):
                cur_record = [0] * (60 + 80 + 52 + 52)
                cur_record[(start_time - hour * 60)] = 1
                cur_record[60 + u - 1] = 1
                cur_record[60 + 80 + cur_start_station] = 1
                cur_record[60 + 80 + 52 + cur_start_station] = 1
                record_list.append(cur_record)

    record_list=shuffle(record_list)
    records = np.array(record_list)
    x_list = records[:, :192]
    y_list = records[:, 192:]
    return x_list, y_list


if __name__ == '__main__':
    argv = sys.argv
    date = argv[1]
    station = argv[2]
    hour = int(argv[3])
    data_inference(date, station, hour)