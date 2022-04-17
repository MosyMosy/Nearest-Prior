
import numpy as np
import pandas as pd


list = np.asarray(pd.read_csv('./dataset/imagenet_to_caltech.csv', skiprows=[0], header=None), dtype=int)

np.savetxt("./dataset/selectedclasses.csv",list , delimiter=",", fmt='%s')