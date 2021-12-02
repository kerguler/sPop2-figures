TSCALE = 4
import numpy

# -------------------------------------------------------------------------------

from datetime import datetime, timedelta, date
str2date2 = lambda x: datetime.strptime(x, '%d/%m/%Y') + timedelta(days=0.5)

import pandas
# ../data/ftemp/data_ftemp.csv
dat = pandas.read_csv("data/ftemp/data_ftemp.csv")
obs = {}
for index, row in dat.iterrows():
    if not (row[0] in obs):
        obs[row[0]] = {
            'temp':numpy.float64(row[1]),
            'photo':24.0,
            'Date':[],
            'days':[],
            'E':[],
            'L':[],
            'P':[],
            'type':'CN0w',
            'compare':['P']
        }
    obs[row[0]]['Date'].append(str2date2(row[2]))
    obs[row[0]]['E'].append(row[3])
    obs[row[0]]['L'].append(row[4])
    obs[row[0]]['P'].append(row[5])
for r in obs:
    obs[r]['Date'] = numpy.array(obs[r]['Date'])
    obs[r]['E'] = numpy.array(obs[r]['E'])
    obs[r]['L'] = numpy.cumsum(obs[r]['L'])
    obs[r]['P'] = numpy.cumsum(obs[r]['P'])
    obs[r]['days'] = numpy.array([(d-obs[r]['Date'][0]).days for d in obs[r]['Date']])
    obs[r]['temp'] = numpy.repeat(obs[r]['temp'],obs[r]['days'][-1]*TSCALE+1)
    obs[r]['photo'] = numpy.repeat(obs[r]['photo'],obs[r]['days'][-1]*TSCALE+1)
