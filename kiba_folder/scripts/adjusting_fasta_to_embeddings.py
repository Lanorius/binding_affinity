# this file is needed to create a fasta file that only holds the proteins that were turned into embeddings

import hdfdict
import csv

fname="reduced_embeddings_file.h5"
res = hdfdict.load(fname)

file = open("mapping_file.csv", "r")
lines = file.readlines() 

fastas = open("sequences_file.fasta", "r")
fasta_lines = fastas.readlines()

clean_fastas = open("clean_sequences_file.fasta", "w")

print(len(res))
print("here")

dicti = {}
count = 0

for line in lines: 
	if count != 0:
		line = line.split(",")
		dicti[">"+line[1]] = line[0]
	else:
		count+=1

sequence_dict = {}

alpha = ""
beta = ""
for line in fasta_lines:
	if line[0] == ">":
		sequence_dict[alpha] = beta
		alpha = line[:-1]
		beta = ""
	else:
		beta+=line[:-1]
sequence_dict[alpha] = beta
alpha = line[:-1]

counter = 0
for line in fasta_lines:
	if line[0] == ">":
		if dicti[line[:-1]] in res:
			clean_fastas.write(line[:-1]+"\n")
			clean_fastas.write(sequence_dict[line[:-1]]+"\n")
			counter+=1

print(counter)

