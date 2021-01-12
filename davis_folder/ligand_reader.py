import hdfdict

file1 = open('ligands.txt', 'r') 
Lines = file1.readlines()

dicti = {}

for i in Lines:
	next_smile = i.split("\t")
	next_smile = next_smile[1][:-1]


	dicti[next_smile[0]] = next_smile[1][:-1]

print(dicti)

fname = "smile_vectors_with_cids.h5"
hdfdict.dump(dicti, fname)