TSCALE = 4
import numpy
FSCALE = numpy.int64(24 / TSCALE)

# -------------------------------------------------------------------------------

from datetime import datetime, timedelta, date
str2date2 = lambda x: datetime.strptime(x, '%d/%m/%Y') + timedelta(days=6.0/24.0)
str2date3 = lambda x: datetime.strptime(x, '%m/%d/%Y %H:%M:%S')

# ../data/vtemp/pet2017_temperature_v2/pet2017_temp_hourly.csv
temp = numpy.genfromtxt("data/vtemp/pet2017_temp_hourly.csv",delimiter=',',dtype=None,skip_header=1,converters={1:str2date3}, encoding='utf=8')
temp24 = {
    'Date':numpy.array([d[1] for d in temp]),
    'Tmean':numpy.array([d[2] for d in temp])
}
temp = {
    'Date':numpy.array([d[1] for d in temp]),
    'Tmean':numpy.array([d[2] for d in temp])
}
print("Env. data check:",all([a.seconds==3600 for a in (temp['Date'][1:]-temp['Date'][:-1])]))

temp['Date'] = temp['Date'][::FSCALE] # quarterly
temp['Tmean'] = numpy.mean(temp['Tmean'].reshape((numpy.int(temp['Tmean'].shape[0]/FSCALE),FSCALE)),axis=1)
print("Quantized:",temp['Date'].shape[0]==temp['Tmean'].shape[0])

# -------------------------------------------------------------------------------

def daylength(lat,day):
    """
    Translated from the daylength function of the geosphere package of R
    lat: latitude in degree decimal (float)
    day: datetime.date or day of the year (integer)
    """
    if isinstance(day,date):
        day = day.timetuple().tm_yday
    pi180 = numpy.pi / 180.0
    P = numpy.arcsin(0.39795 * numpy.cos(0.2163108 + 2 * numpy.arctan(0.9671396 * numpy.tan(0.0086 * (day - 186)))))
    a = (numpy.sin(0.8333 * pi180) + numpy.sin(lat * pi180) * numpy.sin(P))/(numpy.cos(lat * pi180) * numpy.cos(P))
    a = numpy.min([numpy.max([a, -1]), 1])
    DL = 24 - (24.0/numpy.pi) * numpy.arccos(a)
    return(DL)

lon = 19.8516968
lat = 45.2461012

photo = numpy.array([daylength(lat,d.timetuple().tm_yday+(d.timetuple().tm_hour/24.0)) for d in temp['Date']])

# -------------------------------------------------------------------------------

import pandas
dat = pandas.read_csv("data/vtemp/data_vtemp.csv")
obs = {}
for index, row in dat.iterrows():
    if not (row[0] in obs):
        obs[row[0]] = {
            'Date':[],
            'E':[],
            'L':[],
            'P':[],
            'A':[],
            'type':'AN0sP',
            'compare':['L','P','A']
        }
    obs[row[0]]['Date'].append(str2date2(row[1]))
    obs[row[0]]['E'].append(numpy.nan if all([numpy.isnan(row[a]) for a in [2,3,4]]) else numpy.nansum(row[2:5]))
    obs[row[0]]['L'].append(numpy.nan if all([numpy.isnan(row[a]) for a in [5,6,7,8]]) else numpy.nansum(row[5:9]))
    obs[row[0]]['P'].append(row[9])
    obs[row[0]]['A'].append(row[10])
for r in obs:
    obs[r]['Date'] = numpy.array(obs[r]['Date'])
    obs[r]['E'] = numpy.array(obs[r]['E'])
    obs[r]['L'] = numpy.array(obs[r]['L'])
    obs[r]['P'] = numpy.array(obs[r]['P'])
    obs[r]['A'] = numpy.array(obs[r]['A'])
    obs[r]['days'] = numpy.array([(d-obs[r]['Date'][0]).days for d in obs[r]['Date']])
    obs[r]['temp'] = temp['Tmean'][(temp['Date']>=obs[r]['Date'][0]) & (temp['Date']<=obs[r]['Date'][-1])]
    obs[r]['photo'] = photo[(temp['Date']>=obs[r]['Date'][0]) & (temp['Date']<=obs[r]['Date'][-1])]

# Simulation time extended one month following the last data collection
obs_extend = {}
for r in obs:
    obs_extend[r] = {}
    obs_extend[r]['Date'] = numpy.array(obs[r]['Date']).copy()
    obs_extend[r]['E'] = numpy.array(obs[r]['E']).copy()
    obs_extend[r]['L'] = numpy.array(obs[r]['L']).copy()
    obs_extend[r]['P'] = numpy.array(obs[r]['P']).copy()
    obs_extend[r]['A'] = numpy.array(obs[r]['A']).copy()
    obs_extend[r]['type'] = 'AN0sP'
    obs_extend[r]['compare'] = ['L','P','A']
    #
    for i in range(30):
        obs_extend[r]['Date'] = numpy.hstack([obs_extend[r]['Date'], obs_extend[r]['Date'][-1]+timedelta(days=1)])
        obs_extend[r]['E'] = numpy.hstack([obs_extend[r]['E'], numpy.nan])
        obs_extend[r]['L'] = numpy.hstack([obs_extend[r]['L'], numpy.nan])
        obs_extend[r]['P'] = numpy.hstack([obs_extend[r]['P'], numpy.nan])
        obs_extend[r]['A'] = numpy.hstack([obs_extend[r]['A'], numpy.nan])
    #
    obs_extend[r]['days'] = numpy.array([(d-obs_extend[r]['Date'][0]).days for d in obs_extend[r]['Date']])
    obs_extend[r]['temp'] = temp['Tmean'][(temp['Date']>=obs_extend[r]['Date'][0]) & (temp['Date']<=obs_extend[r]['Date'][-1])]
    obs_extend[r]['photo'] = photo[(temp['Date']>=obs_extend[r]['Date'][0]) & (temp['Date']<=obs_extend[r]['Date'][-1])]
