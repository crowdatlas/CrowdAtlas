# CrowdAtlas

The project estimates the crowd distribution within the urban rail transit system based only on the entrance and exit records of all the rail riders. Specifically, we study Singapore MRT (Mass Rapid Transit) as a vehicle and leverage the tap-in and tap-out records of the EZ-Link transit cards to estimate the crowd distribution. A machine learning based solution is designed and implemented to capture the passenger transitions among stations and across time within the rail system and based on that perform accurate estimation of the crowd distribution.

### Dataset
We collect one-year EZ-Link data of Singapore MRT trips, and extract a dataset of ride records within the two major lines EW and NS to evaluate the performance of CrowdAtlas. The name and location (latitude, longitude) of the stations are listed in the file ***rail_stations.csv***. While there are other data fields included in each EZ-Link record, we only retain 6 fields -- [*origin station*, *start date & time*, *destination station*, *end date & time*] in our two-line dataset for CrowdAtlas training and testing.

Due to a non-disclosure agreement (NDA) with Singapore LTA, we cannot disclose the whole dataset. Instead, we generate a sample dataset by minimizing each original record in size as well as mixing noises. Specifically, the sample dataset includes ride records among all EW-line and NS-line stations on weekdays of one month, separated by date and origin station, which has been preprocessed and can be used for both training and testing (***sample_data.zip***). Each record contains four relevant fields [*orgin station*, *start time*, *destination station*, *end time*] (the original start and end date can be omitted). We have verified that the added noise brings little to no impact on the effectiveness of CrowdAtlas.


### Implementation
The major source code files are illustrated as follows.

#### 1. data_preprocessing.py

Data preprocessing is conducted to divide records by date and origin station, extracting ride records within the EW and NS lines from the EZ-Link data. Three types of dirty data -- *same O \& D stations*, *unreasonable ride time*, *untimely ride start* are removed during this process. Preprocessing of data records on different dates can be conducted in parallel to reduce the total execution time through adopting a multiprocess program. There are two parameters when executing this file, *start date* and *end_date* (in the format of "2020-01-01", "20200101" or "1/1/2020"), that indicate a succession of weekdays for preprocessing. The command to run the file is `python3 data_preprocessing.py [start_date] [end_date]`.

#### 2. correlation_learning.py

With the processed data, we train a neural network model to learn the flow correlation (\ie transition probabilities among stations and across time), which is conducted in batches by the hour of start time and separately for different small groups of orgin stations' records. To reduce the overall training time, This process is executed for each hour and station group in parallel by setting two parameters *hour of start time* (6 to 22) and *station group number* (from 1 to 13, we divide stations into 13 groups). Besides, parameters *start date* and *end_date* is also needed to reflect the dates of training data. The command format should be `python3 correlation_learning.py [hour of start time] [station group number] [start_date] [end_date]`. Processes with different station group and hour parameters can be run in parallel by running a shell file ***batch.sh***.

#### 3. data_inference.py

Before training, a data inference process is invoked to transform each raw ride record into a sequence of trajectory records that involve all midway stations. We process ride records in batch by setting parameters *date*, *station* and *hour of start time*.

#### 4. crowd_estimation.py

We obtain the transition probabilities from the trained neural network model, and thus estimate the passenger number at each station by aggregating the passengers transitioned from all origin stations and from all past time beings. There are three parameters here, namely *hour of start time*, *duration in hours* and *testing date*. The command format is `python3 crowd_estimation.py [hour of start time] [duration in hours] [testing_date]`.

#### 5. MA_estimation.py

For performance comparison, we implement the MA (Moving Average) approach as a baseline. The parameters are the same as above, and thus the command format is `python3 MA_estimation.py [hour of start time] [duration in hours] [testing_date]`.

#### 6. get_travel_time.py

To acquire the travel time between any two stations, this process is implemented to crawl the ***TransitLink E-Guide*** web page and generate a file including travel time among all stations ***travel_time.csv***. Meanwhile, it also extracts all the interchange stations and lists them in another file ***inter_stations.csv***. The two files are necessary in all above processes and are both available here.

**Note**: the above steps (**1**, **2**, and **4**) should be executed by sequence.


### Output Results

After executing the above estimation process for a testing date, we can obtain a group of output files that respectively record the estimated and ground-truth passenger numbers at 52 stations in EW and NS lines during the MRT service hours (***[station_name]_num_comp.csv***). We then merge all these output files into one, which records passenger numbers of all the stations of every 5 minutes in the service hours. The files ***merged_num_comp_[1 or 2].csv*** shows the sample results of two typical testing days respectively. In addition, MAPEs of the above estimations for all stations (as well as their regions) during each hour are specifically recorded in the two files ***MAPE_across_time_station_[1 or 2].csv***.
