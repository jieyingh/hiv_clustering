from Bio import SeqIO
from sklearn.cluster import KMeans

import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==============================================================================
# FUNCTIONS
# ==============================================================================

# reads fasta file
def read_fasta(fasta_file):
    sequences = SeqIO.parse(fasta_file, "fasta")
    seq = {}
    for record in sequences:
        seq[record.id] = record.seq
    return (seq)

# generates all possible kmers of a given length
def generate_kmers(k):
    kmers = []
    for kmer in itertools.product('ACGT', repeat = k):
        kmers.append(''.join(kmer))
    return kmers

# creates empty dataframe
def create_dataframe(kmers, seq):
    df = pd.DataFrame(data = 0, columns = kmers, index = seq.keys())
    return df
    
# counts the number of kmers in each sequence
def count_kmers(kmers, seq, df):
    for row in seq.keys():
        for col in kmers:
            df.at[row, col] = (seq[row]).count(col)
    return df

# ==============================================================================
# USER INPUTS
# ==============================================================================
fasta_file = input("Enter the path to the fasta file: ")
k = int(input("Enter the length of the kmer: "))

(seq) = read_fasta(fasta_file)
print("finished reading fasta file")
kmers = generate_kmers(k)
print("finished generating kmers")
df = create_dataframe(kmers, seq)
print("finished creating dataframe")
df = count_kmers(kmers, seq, df)
print("finished counting kmers")

# ==============================================================================
# KMEANS CLUSTERING
# ==============================================================================
print("starting clustering")
kmeans = KMeans(n_clusters = 3, random_state = 0, n_init="auto").fit(df)
print("finished clustering")

y_pred = kmeans.predict(df)

plt.figure(figsize=(15,5))

plt.subplot(131)
plt.scatter(df.iloc[:, 0], df.iloc[:, 1], c=y_pred)
plt.xlabel("Clusters reels")

plt.subplot(132)
plt.scatter(df.iloc[:, 0], df.iloc[:, 1])
plt.xlabel("sans partitionnement")

plt.subplot(133)
plt.scatter(df.iloc[:, 0], df.iloc[:, 1], c=y_pred)
plt.xlabel("Clusters predits")

plt.show()