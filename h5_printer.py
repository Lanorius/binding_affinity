from silx.io.dictdump import h5todict

file = "reduced_embeddings.h5"
in_file = h5todict(file)
print(len(in_file))
