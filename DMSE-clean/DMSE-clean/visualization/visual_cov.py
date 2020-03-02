import numpy as np 
import matplotlib.pyplot as plt
import plotly.plotly as py
import plotly.graph_objs as go
from scipy.stats import norm


cov=np.load("cov.npy")
print "maximal cov:", np.max(cov)
print "minimal cov:", np.min(cov)
plt.imshow(cov, cmap='Greys', interpolation='nearest')
plt.show()

