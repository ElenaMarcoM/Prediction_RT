from src_elena.classification_module import *
import os
import pandas as pd
import numpy as np

category = 'sub_class'
chosen = "Carbonyl compounds"
subgroup_fingerprints = os.path.join("resources","fingerprints_"+chosen+".pkl")
subgroup = os.path.join("resources","subgroups"+category+".tsv")
rtdata = os.path.join("resources","0186_rtdata.tsv")
all_fingerprints = os.path.join("resources","smrt_fingerprints.csv")

obtain_fpSubgroup(subgroup, rtdata, chosen, category, all_fingerprints, subgroup_fingerprints)
