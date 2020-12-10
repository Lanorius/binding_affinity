s = open('kiba_ligands_raw.txt', 'r').read()
whip = eval(s)

w = open('kiba_ligands.txt','w')

for i in range(0,len(list(whip))):
	w.write(list(whip.keys())[i]+"\t"+list(whip.values())[i]+"\n")
