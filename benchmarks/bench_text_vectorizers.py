# [MODULE DOC]
# Top-level benchmark script measuring run time (mean/std over 3 repeats) and peak
# memory (memory_profiler) of CountVectorizer, TfidfVectorizer and HashingVectorizer
# fit_transform on 1000 documents of 20 newsgroups, for word unigrams, word 1-2
# grams, char 4-grams and char_wb 4-grams. Prints two tables (time and memory)
# indexed by analyzer and ngram_range with one column per vectorizer.
# Side effects at import: downloads the dataset, runs the benchmark, prints.
"""

To run this benchmark, you will need,

 * scikit-learn
 * pandas
 * memory_profiler
 * psutil (optional, but recommended)

"""

import itertools
import timeit

import numpy as np
import pandas as pd
from memory_profiler import memory_usage

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import (
    CountVectorizer,
    HashingVectorizer,
    TfidfVectorizer,
)

n_repeat = 3


def run_vectorizer(Vectorizer, X, **params):
    # [FUNCTION DOC]
    # HELPER: Returns a zero-argument closure that builds Vectorizer(**params) and
    # fit-transforms X, for use with timeit and memory_usage.
    # Inputs: Vectorizer class; X iterable of documents; params constructor kwargs.
    def f():
        # [FUNCTION DOC]
        # HELPER: Closure body: constructs the vectorizer and discards the result of
        # fit_transform on X.
        vect = Vectorizer(**params)
        vect.fit_transform(X)

    return f


text = fetch_20newsgroups(subset="train").data[:1000]

print("=" * 80 + "\n#" + "    Text vectorizers benchmark" + "\n" + "=" * 80 + "\n")
print("Using a subset of the 20 newsgroups dataset ({} documents).".format(len(text)))
print("This benchmarks runs in ~1 min ...")

res = []

for Vectorizer, (analyzer, ngram_range) in itertools.product(
    [CountVectorizer, TfidfVectorizer, HashingVectorizer],
    [("word", (1, 1)), ("word", (1, 2)), ("char", (4, 4)), ("char_wb", (4, 4))],
):
    bench = {"vectorizer": Vectorizer.__name__}
    params = {"analyzer": analyzer, "ngram_range": ngram_range}
    bench.update(params)
    dt = timeit.repeat(
        run_vectorizer(Vectorizer, text, **params), number=1, repeat=n_repeat
    )
    bench["time"] = "{:.3f} (+-{:.3f})".format(np.mean(dt), np.std(dt))

    mem_usage = memory_usage(run_vectorizer(Vectorizer, text, **params))

    bench["memory"] = "{:.1f}".format(np.max(mem_usage))

    res.append(bench)


df = pd.DataFrame(res).set_index(["analyzer", "ngram_range", "vectorizer"])

print("\n========== Run time performance (sec) ===========\n")
print(
    "Computing the mean and the standard deviation "
    "of the run time over {} runs...\n".format(n_repeat)
)
print(df["time"].unstack(level=-1))

print("\n=============== Memory usage (MB) ===============\n")
print(df["memory"].unstack(level=-1))
