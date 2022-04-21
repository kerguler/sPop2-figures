TSCALE = 4
import numpy
import model
import matplotlib.pyplot as plt

# (Kiarie-Makara et al.) Effects of Temperature on the Growth and Development of Culex pipiens complex mosquitoes (Diptera: Culicidae) - Journal of Pharmacy and Biological Sciences, vol 10, issue 6 (2015)
# 13:11 L:D cycle
temp = numpy.array([  20.0,   24.0,   28.0])
d1m  = numpy.array([  2.18,    1.5,   1.16]) # days
d1s  = numpy.array([  0.28,    0.0,   0.28]) # days
d2m  = numpy.array([  17.7,   13.6,    9.4]) # days
d2s  = numpy.array([   0.6,    0.6,    0.4]) # days
d3m  = numpy.array([   6.4,   4.34,   2.28]) # days
d3s  = numpy.array([   0.6,    0.6,    0.4]) # days
# (Spanoudis et al.) Effect of temperature on biological parameters of the West Nile Virus vector Culex pipiens form “molestus” (Diptera:Culicidae) in Greece: Constant vs fluctuating Temperatures. Journal of Medical Entomology 2018
# 14:10 -> 16:8 L:D cycle
temp2 = numpy.array([  15, 17.5,   20, 22.5,   25, 27.5,   30, 32.5])
d1m2  = numpy.array([ 4.1,  3.1,  2.9,    2,    2,  1.5,  1.3,  0.9])
d1s2  = numpy.array([0.06, 0.05, 0.05, 0.03, 0.03, 0.09, 0.08, 0.08])
d2m2  = numpy.array([38.7, 32.5, 27.4, 18.4, 12.9, 12.2, 11.4,  8.8])
d2s2  = numpy.array([ 0.3, 0.53,  0.3, 0.43, 0.26, 0.27, 0.17, 0.29])
d3m2  = numpy.array([ 6.4,  4.4,  3.7,  2.7,  2.1,  1.5,  1.6,  1.2])
d3s2  = numpy.array([0.14, 0.19,  0.1, 0.08, 0.05, 0.09, 0.08,  0.1])