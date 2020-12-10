s = open('davis_proteins.txt', 'r').read()
whip = eval(s)

#print(len(list(whip)))

w = open('davis_proteins.fasta','w')

for i in range(0,len(list(whip))):
	w.write(">"+list(whip.keys())[i]+"\n")
	w.write(list(whip.values())[i]+"\n")
