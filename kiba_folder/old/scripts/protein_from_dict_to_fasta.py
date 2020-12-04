from ast import literal_eval

file = open("temp_protein_file.txt", "r")

contents = file.read()
dict = literal_eval(contents)

file.close()

file = open("proteins_with_10_interactions.txt", "w")

for i in dict:
	file.write(">"+i+"\n"+dict[i]+"\n")
