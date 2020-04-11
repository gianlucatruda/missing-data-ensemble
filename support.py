import numpy as np

def ham_distance(a, b):
    # http://code.activestate.com/recipes/499304-hamming-distance/
    diffs = 0
    for ch1, ch2 in zip(a, b):
        if ch1 != ch2:
            diffs += 1
    return diffs

def hamming_distances(vectors: list):
    dim = len(vectors)
    dists = np.zeros((dim, dim))
    for i, v1 in enumerate(vectors):
        for j, v2 in enumerate(vectors):
            dists[i, j] = ham_distance(v1, v2)

    return dists