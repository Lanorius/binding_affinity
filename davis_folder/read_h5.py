from silx.io.dictdump import h5todict
prots_embedded = h5todict("reduced_embeddings_file.h5")
print(len(prots_embedded))