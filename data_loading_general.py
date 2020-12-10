import numpy as np
import pandas as pd
from silx.io.dictdump import h5todict
import torch
import random  # only for experimenting

class Dataset:
    def __init__(self, embeddings, compound_vectors, label_file, mapping_file, data_type, cluster_map=None):

        file = open(mapping_file, "r")
        lines = file.readlines()

        dicti = {}  # dicti is a dictionary that maps the protein ids to the encoded ids given by SeqVec
        # this is required since the h5 files always sorts its entries, instead of keeping them in the same
        # order they come in other files
        count = 0

        data_ranges = {"pkd": [5, 11], "kiba": [0, 18]}  # this dictionary is important for each new type of values
        # that are being evaluated
        for line in lines:
            if count != 0:
                line = line.split(",")
                dicti[">" + line[1]] = line[0]
            else:
                count += 1

        self.prots_embedded = h5todict(embeddings)
        self.compounds_vae = h5todict(compound_vectors)
        if cluster_map is not None:
            self.cluster_map = h5todict(cluster_map)

        self.labels = []

        self.interactions = pd.read_csv(label_file)
        self.proteins = self.interactions.columns.tolist()
        del self.proteins[0]
        self.compounds = [str(i) for i in self.interactions["cids"].tolist()]
        self.interactions = self.interactions.drop(self.interactions.columns[0], axis=1)
        self.labels = np.hstack(self.interactions.values)

        self.data_to_load = []
        label_index = -1

        self.existing_labels = []

        for comp in self.compounds:
            for prot in self.proteins:
                label_index += 1
                if (self.labels[label_index] > 0) & np.isfinite(self.labels[label_index]):
                    if self.labels[label_index] < data_ranges[data_type][1]:
                        if cluster_map is None:
                            self.data_to_load.append([dicti[prot], comp])
                        else:
                            self.data_to_load.append([dicti[self.cluster_map[prot].item()], comp])
                        self.existing_labels.append(max(self.labels[label_index], data_ranges[data_type][0]))
                # get keys from both and write to list

        # random.shuffle(self.existing_labels) # for random results # doesn't work so well yet I guess

    def __getitem__(self, index):

        key_prot, key_comp = self.data_to_load[index]
        prot_comp = [torch.from_numpy(self.prots_embedded[key_prot]), self.compounds_vae[key_comp]]

        label = self.existing_labels[index]  # for real results

        return prot_comp, label

    def __len__(self):
        return len(self.data_to_load)
