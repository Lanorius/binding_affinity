from ast import literal_eval

file = open("raw_compound_file.txt", "r")

contents = file.read()
dict = literal_eval(contents)

file.close()

file_a = open("cids_smiles.txt", "w")
file_b = open("comp_index_to_id.csv","w")

for i in dict:
	file_a.write(i+"\t"+dict[i]+"\n")
	file_b.write(i+"\n")

