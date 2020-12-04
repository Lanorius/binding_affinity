import pandas as pd
import hdfdict
import numpy as np

cids = pd.read_csv("comp_index_to_id.csv",header=None)
vectors = pd.read_csv("inner_vectors.csv",header=None)

cids = cids[0].tolist()
vectors = vectors.values.tolist()
dicti = {}

for i in range(len(cids)):
	dicti[str(cids[i])] = [np.float32(val) for val in vectors[i]]


fname = "smile_vectors_with_cids.h5"
hdfdict.dump(dicti, fname)


# res = hdfdict.load(fname)

# print(res)


