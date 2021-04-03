# %%
from ast import literal_eval
import pandas as pd
import numpy as np

df = pd.read_csv("documents.csv")
df["codes"] = df["codes"].apply(lambda x: literal_eval(x))

# %% 

code_dict = {
    "C11": 0,
    "C12": 1,
    "C13": 2,
    "C14": 3,
    "C15": 4,
    "C151": 5,
    "C1511": 6,
    "C152": 7,
    "C16": 8,
    "C17": 9,
    "C171": 10,
    "C172": 11,
    "C173": 12,
    "C174": 13,
    "C18": 14,
    "C181": 15,
    "C182": 16,
    "C183": 17,
    "C21": 18,
    "C22": 19,
    "C23": 20,
    "C24": 21,
    "C31": 22,
    "C311": 23,
    "C312": 24,
    "C313": 25,
    "C32": 26,
    "C33": 27,
    "C331": 28,
    "C34": 29,
    "C41": 30,
    "C411": 31,
    "C42": 32,
    "CCAT": 33,
    "E11": 34,
    "E12": 35,
    "E121": 36,
    "E13": 37,
    "E131": 38,
    "E132": 39,
    "E14": 40,
    "E141": 41,
    "E142": 42,
    "E143": 43,
    "E21": 44,
    "E211": 45,
    "E212": 46,
    "E31": 47,
    "E311": 48,
    "E312": 49,
    "E313": 50,
    "E41": 51,
    "E411": 52,
    "E51": 53,
    "E511": 54,
    "E512": 55,
    "E513": 56,
    "E61": 57,
    "E71": 58,
    "ECAT": 59,
    "G15": 60,
    "G151": 61,
    "G152": 62,
    "G153": 63,
    "G154": 64,
    "G155": 65,
    "G156": 66,
    "G157": 67,
    "G158": 68,
    "G159": 69,
    "GCAT": 70,
    "GCRIM": 71,
    "GDEF": 72,
    "GDIP": 73,
    "GDIS": 74,
    "GENT": 75,
    "GENV": 76,
    "GFAS": 77,
    "GHEA": 78,
    "GJOB": 79,
    "GMIL": 80,
    "GOBIT": 81,
    "GODD": 82,
    "GPOL": 83,
    "GPRO": 84,
    "GREL": 85,
    "GSCI": 86,
    "GSPO": 87,
    "GTOUR": 88,
    "GVIO": 89,
    "GVOTE": 90,
    "GWEA": 91,
    "GWELF": 92,
    "M11": 93,
    "M12": 94,
    "M13": 95,
    "M131": 96,
    "M132": 97,
    "M14": 98,
    "M141": 99,
    "M142": 100,
    "M143": 101,
    "MCAT": 102	
}
code_df = pd.DataFrame(list(code_dict))

num_labels = len(code_dict)
vector_list = []

def vectorize_classes(x, code_dict={}):
    label_vector = np.zeros((num_labels), dtype=int)
    non_zero_idx = [code_dict[label] for label in x]
    label_vector[non_zero_idx] = 1
    label_vector = label_vector.tolist()
    vector_list.append(label_vector)
    return label_vector

df["labels"] = df["codes"].apply(vectorize_classes, code_dict=code_dict)
label_matrix = np.array(vector_list)
label_df = pd.DataFrame(label_matrix, columns=list(code_dict.keys()))

#%%
# Write label_df into a .csv file 

label_df.to_csv("labels.csv")

# %%
# Verification of the above method

# Verify label vectors are correct
from more_itertools import locate

# Get indexes that correspond to labels existing for row z
z = 0
has1 = list(locate(df["labels"][z], lambda x: x == 1))

# Display codes that correspong to the indexes, derived from the label vectors
code_df.iloc[has1]

# %%

# Display original codes
df["codes"][z]


# %%
df["labels"].head()

# %%
