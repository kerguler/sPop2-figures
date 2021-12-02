TSCALE = 4
import numpy

# -------------------------------------------------------------------------------

import pandas
# ../data/data_culex_all.csv
dat = pandas.read_csv("data/culex/data_culex_all.csv")
obs = {}
for index, row in dat.iterrows():
    key = "%s-%s" %(int(row[0]),int(row[1]))
    if not (key in obs):
        obs[key] = {
            'temp':numpy.float64(row[0]),
            'photo':numpy.float64(24.0),
            'days':[],
            'L':[],
            'P':[],
            'A':[],
            'type':'CN0w',
            'compare':['P','A']
        }
    obs[key]['days'].append(row[2])
    obs[key]['L'].append(row[3])
    obs[key]['P'].append(row[4])
    obs[key]['A'].append(row[5])
for r in obs:
    obs[r]['days'] = numpy.array(obs[r]['days'],dtype=numpy.int64)
    obs[r]['L'] = numpy.array(obs[r]['L'])
    obs[r]['P'] = numpy.array(obs[r]['P'])
    obs[r]['A'] = numpy.array(obs[r]['A'])
    obs[r]['temp'] = numpy.repeat(obs[r]['temp'],obs[r]['days'][-1]*TSCALE+1)
    obs[r]['photo'] = numpy.repeat(obs[r]['photo'],obs[r]['days'][-1]*TSCALE+1)

