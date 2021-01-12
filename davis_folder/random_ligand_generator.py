import re
import random
# import pandas as pd
import hdfdict
import numpy as np


compound_id_file = open('ligands.txt','r')
compound_lines = compound_id_file.readlines()

dicti = {}
for i in compound_lines:
	vals = np.random.rand(24).astype("float32")
	dicti[re.split(r'\t+', i)[0]] = vals


fname = "testing_lignads.h5"
hdfdict.dump(dicti, fname)


	
