import pandas as pd
import numpy as np
import sys

sys.path.append('/home/jvallet/app')

SECRET_KEY = '#un\xb2\x9bE\xd6\xaf7#\x7fT\xc8\xd7\x1f |\x9f}\xf7\t\xb1l\xcd'

# Project 4
ORIGIN_DEST = pd.read_csv('/home/jvallet/app/models/ORIGIN_DEST.csv', index_col=0)
O_NBFLIGHTS = pd.read_csv('/home/jvallet/app/models/O_NBFLIGHTS.csv', index_col=0)
D_NBFLIGHTS = pd.read_csv('/home/jvallet/app/models/D_NBFLIGHTS.csv', index_col=0)
FLNUM = pd.read_csv('/home/jvallet/app/models/FLNUM.csv', index_col=0)
CIE = np.load('/home/jvallet/app/models/CIE.npy')

# Project 6
X_TRAIN = pd.read_csv('/home/jvallet/app/modelsP6/X_train.csv', index_col=0)

MOIS_LIB = ['','Jan.','Fév.','Mars','Avr.','Mai','Juin','Juil.','Août','Sep.','Oct.','Nov.','Déc.']