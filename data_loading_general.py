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