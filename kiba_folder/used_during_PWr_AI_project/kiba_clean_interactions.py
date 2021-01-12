import csv
import numpy as np
import pandas as pd

#prepare the protein names

protein_id_file = open('sequences_file.fasta','r')
protein_lines = protein_id_file.readlines()
protein_ids = ['cids']
for i in protein_lines:
	if i[0] == ">":
		protein_ids+=[(i[:-1])]


#prepare the compound names

compound_id_file = open('ligands.txt','r')
compound_lines = compound_id_file.readlines()
compound_ids = []
for i in compound_lines:
	new_cid = ""
	for j in i:
		if j != "\t":
			new_cid+=j
		else:
			break
	compound_ids+=[new_cid]


#prepare the interactions


interactions = []

interactions_file = open('interactions.txt','r')
interaction_lines = interactions_file.readlines()
inde_x = 0
for i in interaction_lines:
	line = i.split()
	new_line = [compound_ids[inde_x]]
	for j in line:
		new_line += [float(j)]
	inde_x+=1
	interactions+= [new_line]


with open('cleaned_interactions.csv', 'w') as f: 
      
    # using csv.writer method from CSV package 
    write = csv.writer(f) 
    write.writerow(protein_ids) 
    write.writerows(interactions) 

