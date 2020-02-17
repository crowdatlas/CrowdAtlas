import numpy as np
import pandas as pd
from selenium import webdriver

bs = webdriver.Chrome()
travel_time_matrix=np.zeros((157,157),dtype=int)
df_tvl=pd.DataFrame(columns=['start_station','end_station','travel_time'])
df_interchange=pd.DataFrame(columns=['inter_station','station1','station2', 'station3'])
n_inter = 0
station_list=[]
stations=[]
interchange_stations=[]
bs.get('https://www.transitlink.com.sg/eservice/eguide/rail_idx.php')
for k in range(0,157):
    station = bs.find_element_by_xpath(".//*[@name='mrtcode_start']/option["+str(k+2)+"]").text
    station_list.append(station)
    stations.append(station[station.index('[')+1:station.index(']')])
    if '/' in station:
        interchange_stations.append(station[station.index('[')+1:station.index(']')])
        inter_station = station[station.index('[')+1:station.index(']')]
        including_stations = inter_station.split('/')
        if len(including_stations) < 3:
            including_stations.append('')
        df_interchange.loc[n_inter] = [inter_station, including_stations[0], including_stations[1], including_stations[2]]
        n_inter += 1
print(interchange_stations)

df_interchange.to_csv('interchange_stations.csv', index=None)

df_tvl=pd.DataFrame(np.zeros((157,157)),index=stations, columns=stations, dtype=int)

for i in range(0,157):
    for j in range(0,157):
        bs.get('https://www.transitlink.com.sg/eservice/eguide/rail_idx.php')
        start = bs.find_element_by_name('mrtcode_start')
        start.send_keys(station_list[i])
        end = bs.find_element_by_name('mrtcode_end')
        end.send_keys(station_list[j])
        submit = bs.find_element_by_name('submit')
        submit.submit()
        df_tvl.loc[stations[i],stations[j]] = int(bs.find_element_by_xpath(".//*[@class='eguide-table']/table[2]/tbody/tr[2]/td[4]").text)
        #df_tvl.loc[station_list[i], station_list[j]] = int('9000')

bs.close()
df_tvl.to_csv('travel_time.csv')




