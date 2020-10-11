'''
this file takes four arguments:
first:	a file with compounds to be compressed
second: a file with interactions that need to be cleaned of all the valeus of
	compounds that couldn't be encoded and of proteins that were to similar
third:	the raw fasta file of proteins
fourth:	the redudancy reduced fasta file of the proteins
'''

import sys
import pandas as pd
import csv

from os import environ
environ['KERAS_BACKEND'] = 'tensorflow'

# vae stuff
from chemvae.vae_utils import VAEUtils
from chemvae import mol_utils as mu

# import scientific py
import numpy as np
import pandas as pd
# rdkit stuff
from rdkit.Chem import AllChem as Chem
from rdkit.Chem import PandasTools

vae = VAEUtils(directory='../models/zinc_properties')

#######################################################################

file = open(sys.argv[1],"r")
lines= file.readlines()

interactions = open(sys.argv[2],"r").read()
interactions = [item.split() for item in interactions.split('\n')[:-1]]

fasta_all = open(sys.argv[3],"r")
fasta_all = fasta_all.readlines()

fasta_reduced = open(sys.argv[4],"r")
fasta_reduced = fasta_reduced.readlines()

#######################################################################

cids = [] #saving the cids so that they can be indexed over later
only_smiles = []
for line in lines:
	line_index = 0
	new_cid = ""
	for letter in line:
		if (letter == "\t"):
			only_smiles += [line[(line_index+1):-1]]
			break
		new_cid += letter
		line_index += 1
	cids += [new_cid]


fasta_all_ids = []
for line in fasta_all:
	if line[0] == ">":
		fasta_all_ids.append(line[:-1])

fasta_reduced_ids = []
for line in fasta_reduced:
	if line[0] == ">":
		fasta_reduced_ids.append(line[:-1])

all_vectors = []
all_encodeable_indices = []

num = 0
already_removed = 0

for smile in only_smiles:
	try:
		smiles_1 = mu.canon_smiles(smile.replace("'",""))

		X_1 = vae.smiles_to_hot(smiles_1,canonize_smiles=False)
		z_1 = vae.encode(X_1)
		all_vectors+=z_1.tolist()
		all_encodeable_indices.append(cids[num])
	except:
		print("Couldn't encode the smile in line ", num)
		del interactions[num-already_removed]
		already_removed+=1
	num+=1



all_vectors = pd.DataFrame(all_vectors)
interactions = pd.DataFrame(interactions)

index_for_fasta_reduced_ids = len(fasta_reduced_ids)-1
for ind in range(len(fasta_all_ids)-1,-1,-1):
	if fasta_all_ids[ind] == fasta_reduced_ids[index_for_fasta_reduced_ids]:
		index_for_fasta_reduced_ids -= 1
	else:
		interactions = interactions.drop(interactions.columns[ind],axis=1)

# add col and row names ass a column and row

interactions.insert(0,column="cids",value=all_encodeable_indices)
fasta_reduced_ids.insert(0, "cids")

interactions.loc[-1] = fasta_reduced_ids
interactions.index = interactions.index + 1
interactions = interactions.sort_index()


all_vectors.to_csv("inner_vectors.csv",index=False,header=False)
interactions.to_csv("cleaned_interactions.csv",index=False,header=False)
all_encodeable_indices = pd.DataFrame(all_encodeable_indices)
all_encodeable_indices.to_csv("comp_index_to_id.csv", index=False, header=False) # this is simply a file that has all cids
