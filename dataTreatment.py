from Bio import SeqIO
import csv
import itertools
import numpy as np

def y(class_csv):
    with open(class_csv, newline='') as csvfile:
        reader = csv.reader(csvfile)
        y = {rows[0]:rows[1] for rows in reader}
    y = np.array(list(y.values()))
    
    unique_values = np.unique(y)

    return (y, unique_values)

def x(fasta_file):
    sequences = SeqIO.parse(fasta_file, "fasta")
    x = {}
    for record in sequences:
        x[record.id] = record.seq
    return (x)

def generate_kmers(k):
    kmers = []
    for kmer in itertools.product('ACGT', repeat = k):
        kmers.append(''.join(kmer))
    return kmers

def matrice(kmers, seq):
    df = np.zeros((len(seq), len(kmers)), dtype=int)
    index = list(seq.keys())
    return (df, index)

def count_kmers(kmers, seq, df, index):
    for i, row_key in enumerate(index):
        row_seq = seq[row_key]
        for j, col_key in enumerate(kmers):
            df[i, j] = row_seq.count(col_key)
    x_matrix = np.array(df)
    return x_matrix