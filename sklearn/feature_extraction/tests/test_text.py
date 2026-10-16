# [MODULE DOC]
# Tests for sklearn.feature_extraction.text: accent stripping, word/char/char_wb
# analyzers, CountVectorizer, HashingVectorizer, TfidfTransformer and
# TfidfVectorizer (vocabulary handling, stop words, df thresholds, idf, norms,
# dtypes, pickling, sparse containers, grid search integration, warnings and error
# messages). Defines small food-themed corpora (JUNK_FOOD_DOCS, NOTJUNK_FOOD_DOCS,
# ALL_FOOD_DOCS) shared across tests.
import pickle
import re
import uuid
import warnings
from collections import defaultdict
from collections.abc import Mapping
from functools import partial
from io import StringIO
from itertools import product

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal
from scipy import sparse

from sklearn.base import clone
from sklearn.feature_extraction.text import (
    ENGLISH_STOP_WORDS,
    CountVectorizer,
    HashingVectorizer,
    TfidfTransformer,
    TfidfVectorizer,
    strip_accents_ascii,
    strip_accents_unicode,
    strip_tags,
)
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.utils import _align_api_if_sparse
from sklearn.utils._testing import (
    assert_allclose_dense_sparse,
    assert_almost_equal,
    skip_if_32bit,
)
from sklearn.utils.fixes import (
    _IS_WASM,
    CSC_CONTAINERS,
    CSR_CONTAINERS,
    _sparse_random_array,
)

JUNK_FOOD_DOCS = (
    "the pizza pizza beer copyright",
    "the pizza burger beer copyright",
    "the the pizza beer beer copyright",
    "the burger beer beer copyright",
    "the coke burger coke copyright",
    "the coke burger burger",
)

NOTJUNK_FOOD_DOCS = (
    "the salad celeri copyright",
    "the salad salad sparkling water copyright",
    "the the celeri celeri copyright",
    "the tomato tomato salad water",
    "the tomato salad water copyright",
)

ALL_FOOD_DOCS = JUNK_FOOD_DOCS + NOTJUNK_FOOD_DOCS


def uppercase(s):
    # [FUNCTION DOC]
    # HELPER: Custom preprocessor: strips accents (unicode normalization) then
    # uppercases. Input: s - str. Output: str.
    return strip_accents_unicode(s).upper()


def strip_eacute(s):
    # [FUNCTION DOC]
    # HELPER: Custom preprocessor replacing "é" by "e". Input: s - str. Output: str.
    return s.replace("é", "e")


def split_tokenize(s):
    # [FUNCTION DOC]
    # HELPER: Custom tokenizer splitting on whitespace. Output: list of str.
    return s.split()


def lazy_analyze(s):
    # [FUNCTION DOC]
    # HELPER: Custom analyzer ignoring its input and returning the single token
    # "the_ultimate_feature". Output: one-element list.
    return ["the_ultimate_feature"]


def test_strip_accents():
    # [FUNCTION DOC]
    # Checks strip_accents_unicode on Latin accented letters, an Arabic alef with
    # hamza, mixed text, pre-decomposed characters, lone combining marks (removed
    # entirely) and multiple combining marks on one base letter.
    # check some classical latin accentuated symbols
    a = "àáâãäåçèéêë"
    expected = "aaaaaaceeee"
    assert strip_accents_unicode(a) == expected

    a = "ìíîïñòóôõöùúûüý"
    expected = "iiiinooooouuuuy"
    assert strip_accents_unicode(a) == expected

    # check some arabic
    a = "\u0625"  # alef with a hamza below: إ
    expected = "\u0627"  # simple alef: ا
    assert strip_accents_unicode(a) == expected

    # mix letters accentuated and not
    a = "this is à test"
    expected = "this is a test"
    assert strip_accents_unicode(a) == expected

    # strings that are already decomposed
    a = "o\u0308"  # o with diaeresis
    expected = "o"
    assert strip_accents_unicode(a) == expected

    # combining marks by themselves
    a = "\u0300\u0301\u0302\u0303"
    expected = ""
    assert strip_accents_unicode(a) == expected

    # Multiple combining marks on one character
    a = "o\u0308\u0304"
    expected = "o"
    assert strip_accents_unicode(a) == expected


def test_to_ascii():
    # [FUNCTION DOC]
    # Checks strip_accents_ascii on Latin accented letters and mixed text, and that a
    # character without ASCII equivalent (Arabic alef with hamza) is dropped.
    # check some classical latin accentuated symbols
    a = "àáâãäåçèéêë"
    expected = "aaaaaaceeee"
    assert strip_accents_ascii(a) == expected

    a = "ìíîïñòóôõöùúûüý"
    expected = "iiiinooooouuuuy"
    assert strip_accents_ascii(a) == expected

    # check some arabic
    a = "\u0625"  # halef with a hamza below
    expected = ""  # halef has no direct ascii match
    assert strip_accents_ascii(a) == expected

    # mix letters accentuated and not
    a = "this is à test"
    expected = "this is a test"
    assert strip_accents_ascii(a) == expected


@pytest.mark.parametrize("Vectorizer", (CountVectorizer, HashingVectorizer))
def test_word_analyzer_unigrams(Vectorizer):
    # [FUNCTION DOC]
    # For CountVectorizer and HashingVectorizer, checks the default word analyzer
    # output: ascii accent stripping with default token pattern (2+ word chars,
    # lowercased), input="file" reading a StringIO, a custom preprocessor (which
    # replaces lowercasing), and a custom whitespace tokenizer keeping punctuation.
    wa = Vectorizer(strip_accents="ascii").build_analyzer()
    text = "J'ai mangé du kangourou  ce midi, c'était pas très bon."
    expected = [
        "ai",
        "mange",
        "du",
        "kangourou",
        "ce",
        "midi",
        "etait",
        "pas",
        "tres",
        "bon",
    ]
    assert wa(text) == expected

    text = "This is a test, really.\n\n I met Harry yesterday."
    expected = ["this", "is", "test", "really", "met", "harry", "yesterday"]
    assert wa(text) == expected

    wa = Vectorizer(input="file").build_analyzer()
    text = StringIO("This is a test with a file-like object!")
    expected = ["this", "is", "test", "with", "file", "like", "object"]
    assert wa(text) == expected

    # with custom preprocessor
    wa = Vectorizer(preprocessor=uppercase).build_analyzer()
    text = "J'ai mangé du kangourou  ce midi,  c'était pas très bon."
    expected = [
        "AI",
        "MANGE",
        "DU",
        "KANGOUROU",
        "CE",
        "MIDI",
        "ETAIT",
        "PAS",
        "TRES",
        "BON",
    ]
    assert wa(text) == expected

    # with custom tokenizer
    wa = Vectorizer(tokenizer=split_tokenize, strip_accents="ascii").build_analyzer()
    text = "J'ai mangé du kangourou  ce midi, c'était pas très bon."
    expected = [
        "j'ai",
        "mange",
        "du",
        "kangourou",
        "ce",
        "midi,",
        "c'etait",
        "pas",
        "tres",
        "bon.",
    ]
    assert wa(text) == expected


def test_word_analyzer_unigrams_and_bigrams():
    # [FUNCTION DOC]
    # Checks a word analyzer with ngram_range=(1, 2) yields all unigrams followed by
    # all bigrams in document order.
    wa = CountVectorizer(
        analyzer="word", strip_accents="unicode", ngram_range=(1, 2)
    ).build_analyzer()

    text = "J'ai mangé du kangourou  ce midi, c'était pas très bon."
    expected = [
        "ai",
        "mange",
        "du",
        "kangourou",
        "ce",
        "midi",
        "etait",
        "pas",
        "tres",
        "bon",
        "ai mange",
        "mange du",
        "du kangourou",
        "kangourou ce",
        "ce midi",
        "midi etait",
        "etait pas",
        "pas tres",
        "tres bon",
    ]
    assert wa(text) == expected


def test_unicode_decode_error():
    # [FUNCTION DOC]
    # Checks the word and char analyzers raise UnicodeDecodeError when given UTF-8
    # bytes with encoding="ascii" and the default strict decode_error.
    # decode_error default to strict, so this should fail
    # First, encode (as bytes) a unicode string.
    text = "J'ai mangé du kangourou  ce midi, c'était pas très bon."
    text_bytes = text.encode("utf-8")

    # Then let the Analyzer try to decode it as ascii. It should fail,
    # because we have given it an incorrect encoding.
    wa = CountVectorizer(ngram_range=(1, 2), encoding="ascii").build_analyzer()
    with pytest.raises(UnicodeDecodeError):
        wa(text_bytes)

    ca = CountVectorizer(
        analyzer="char", ngram_range=(3, 6), encoding="ascii"
    ).build_analyzer()
    with pytest.raises(UnicodeDecodeError):
        ca(text_bytes)


def test_char_ngram_analyzer():
    # [FUNCTION DOC]
    # Checks char analyzer (3-6 grams) output: whitespace normalized to single spaces,
    # first and last five n-grams match expectations; also works with input="file".
    cnga = CountVectorizer(
        analyzer="char", strip_accents="unicode", ngram_range=(3, 6)
    ).build_analyzer()

    text = "J'ai mangé du kangourou  ce midi, c'était pas très bon"
    expected = ["j'a", "'ai", "ai ", "i m", " ma"]
    assert cnga(text)[:5] == expected
    expected = ["s tres", " tres ", "tres b", "res bo", "es bon"]
    assert cnga(text)[-5:] == expected

    text = "This \n\tis a test, really.\n\n I met Harry yesterday"
    expected = ["thi", "his", "is ", "s i", " is"]
    assert cnga(text)[:5] == expected

    expected = [" yeste", "yester", "esterd", "sterda", "terday"]
    assert cnga(text)[-5:] == expected

    cnga = CountVectorizer(
        input="file", analyzer="char", ngram_range=(3, 6)
    ).build_analyzer()
    text = StringIO("This is a test with a file-like object!")
    expected = ["thi", "his", "is ", "s i", " is"]
    assert cnga(text)[:5] == expected


def test_char_wb_ngram_analyzer():
    # [FUNCTION DOC]
    # Checks char_wb analyzer pads each word with spaces and only builds n-grams inside
    # those padded words (first/last expected n-grams), also for file input.
    cnga = CountVectorizer(
        analyzer="char_wb", strip_accents="unicode", ngram_range=(3, 6)
    ).build_analyzer()

    text = "This \n\tis a test, really.\n\n I met Harry yesterday"
    expected = [" th", "thi", "his", "is ", " thi"]
    assert cnga(text)[:5] == expected

    expected = ["yester", "esterd", "sterda", "terday", "erday "]
    assert cnga(text)[-5:] == expected

    cnga = CountVectorizer(
        input="file", analyzer="char_wb", ngram_range=(3, 6)
    ).build_analyzer()
    text = StringIO("A test with a file-like object!")
    expected = [" a ", " te", "tes", "est", "st ", " tes"]
    assert cnga(text)[:6] == expected


def test_word_ngram_analyzer():
    # [FUNCTION DOC]
    # Checks word 3-6 gram analyzer first and last n-grams (ordered by n then
    # position), and that a file input gives the same result as the raw string.
    cnga = CountVectorizer(
        analyzer="word", strip_accents="unicode", ngram_range=(3, 6)
    ).build_analyzer()

    text = "This \n\tis a test, really.\n\n I met Harry yesterday"
    expected = ["this is test", "is test really", "test really met"]
    assert cnga(text)[:3] == expected

    expected = [
        "test really met harry yesterday",
        "this is test really met harry",
        "is test really met harry yesterday",
    ]
    assert cnga(text)[-3:] == expected

    cnga_file = CountVectorizer(
        input="file", analyzer="word", ngram_range=(3, 6)
    ).build_analyzer()
    file = StringIO(text)
    assert cnga_file(file) == cnga(text)


def test_countvectorizer_custom_vocabulary():
    # [FUNCTION DOC]
    # For dict, list, iterator and defaultdict vocabularies: fit keeps the mapping
    # (Mappings) or the term set (sequences), transform has one column per term, and
    # inverse_transform works on an unfitted vectorizer given a fixed vocabulary.
    vocab = {"pizza": 0, "beer": 1}
    terms = set(vocab.keys())

    # Try a few of the supported types.
    for typ in [dict, list, iter, partial(defaultdict, int)]:
        v = typ(vocab)
        vect = CountVectorizer(vocabulary=v)
        vect.fit(JUNK_FOOD_DOCS)
        if isinstance(v, Mapping):
            assert vect.vocabulary_ == vocab
        else:
            assert set(vect.vocabulary_) == terms
        X = vect.transform(JUNK_FOOD_DOCS)
        assert X.shape[1] == len(terms)
        v = typ(vocab)
        vect = CountVectorizer(vocabulary=v)
        inv = vect.inverse_transform(X)
        assert len(inv) == X.shape[0]


def test_countvectorizer_custom_vocabulary_pipeline():
    # [FUNCTION DOC]
    # Checks a fixed list vocabulary in a CountVectorizer+TfidfTransformer pipeline
    # is preserved and determines the output width.
    what_we_like = ["pizza", "beer"]
    pipe = Pipeline(
        [
            ("count", CountVectorizer(vocabulary=what_we_like)),
            ("tfidf", TfidfTransformer()),
        ]
    )
    X = pipe.fit_transform(ALL_FOOD_DOCS)
    assert set(pipe.named_steps["count"].vocabulary_) == set(what_we_like)
    assert X.shape[1] == len(what_we_like)


def test_countvectorizer_custom_vocabulary_repeated_indices():
    # [FUNCTION DOC]
    # Checks a vocabulary mapping two terms to the same index raises ValueError
    # "Vocabulary contains repeated indices" on fit.
    vocab = {"pizza": 0, "beer": 0}
    msg = "Vocabulary contains repeated indices"
    with pytest.raises(ValueError, match=msg):
        vect = CountVectorizer(vocabulary=vocab)
        vect.fit(["pasta_siziliana"])


def test_countvectorizer_custom_vocabulary_gap_index():
    # [FUNCTION DOC]
    # Checks a vocabulary whose indices do not start at 0 raises ValueError
    # "doesn't contain index" on fit.
    vocab = {"pizza": 1, "beer": 2}
    with pytest.raises(ValueError, match="doesn't contain index"):
        vect = CountVectorizer(vocabulary=vocab)
        vect.fit(["pasta_verdura"])


def test_countvectorizer_stop_words():
    # [FUNCTION DOC]
    # Checks get_stop_words: "english" returns ENGLISH_STOP_WORDS, unknown strings
    # raise ValueError, and a list is returned as a set.
    cv = CountVectorizer()
    cv.set_params(stop_words="english")
    assert cv.get_stop_words() == ENGLISH_STOP_WORDS
    cv.set_params(stop_words="_bad_str_stop_")
    with pytest.raises(ValueError):
        cv.get_stop_words()
    cv.set_params(stop_words="_bad_unicode_stop_")
    with pytest.raises(ValueError):
        cv.get_stop_words()
    stoplist = ["some", "other", "words"]
    cv.set_params(stop_words=stoplist)
    assert cv.get_stop_words() == set(stoplist)


def test_countvectorizer_empty_vocabulary():
    # [FUNCTION DOC]
    # Checks ValueError "empty vocabulary" for an empty fixed vocabulary and for a
    # corpus consisting only of English stop words.
    with pytest.raises(ValueError, match="empty vocabulary"):
        vect = CountVectorizer(vocabulary=[])
        vect.fit(["foo"])

    with pytest.raises(ValueError, match="empty vocabulary"):
        v = CountVectorizer(max_df=1.0, stop_words="english")
        # fit on stopwords only
        v.fit(["to be or not to be", "and me too", "and so do you"])


def test_fit_countvectorizer_twice():
    # [FUNCTION DOC]
    # Checks refitting on a different corpus replaces the vocabulary (different
    # number of columns).
    cv = CountVectorizer()
    X1 = cv.fit_transform(ALL_FOOD_DOCS[:5])
    X2 = cv.fit_transform(ALL_FOOD_DOCS[5:])
    assert X1.shape[1] != X2.shape[1]


def test_countvectorizer_custom_token_pattern():
    # [FUNCTION DOC]
    # Non-regression (#12971): a token_pattern with one capturing group uses the
    # captured group as the token, giving the expected feature names.
    """Check `get_feature_names_out()` when a custom token pattern is passed.
    Non-regression test for:
    https://github.com/scikit-learn/scikit-learn/issues/12971
    """
    corpus = [
        "This is the 1st document in my corpus.",
        "This document is the 2nd sample.",
        "And this is the 3rd one.",
        "Is this the 4th document?",
    ]
    token_pattern = r"[0-9]{1,3}(?:st|nd|rd|th)\s\b(\w{2,})\b"
    vectorizer = CountVectorizer(token_pattern=token_pattern)
    vectorizer.fit_transform(corpus)
    expected = ["document", "one", "sample"]
    feature_names_out = vectorizer.get_feature_names_out()
    assert_array_equal(feature_names_out, expected)


def test_countvectorizer_custom_token_pattern_with_several_group():
    # [FUNCTION DOC]
    # Non-regression (#12971): a token_pattern with two capturing groups makes fit
    # raise ValueError "More than 1 capturing group".
    """Check that we raise an error if token pattern capture several groups.
    Non-regression test for:
    https://github.com/scikit-learn/scikit-learn/issues/12971
    """
    corpus = [
        "This is the 1st document in my corpus.",
        "This document is the 2nd sample.",
        "And this is the 3rd one.",
        "Is this the 4th document?",
    ]

    token_pattern = r"([0-9]{1,3}(?:st|nd|rd|th))\s\b(\w{2,})\b"
    err_msg = "More than 1 capturing group in token pattern"
    vectorizer = CountVectorizer(token_pattern=token_pattern)
    with pytest.raises(ValueError, match=err_msg):
        vectorizer.fit(corpus)


def test_countvectorizer_uppercase_in_vocab():
    # [FUNCTION DOC]
    # Non-regression (#21251): with lowercase=True and a vocabulary containing
    # uppercase, fit warns (UserWarning) but transform does not (warnings turned into
    # errors to verify).
    # Check that the check for uppercase in the provided vocabulary is only done at fit
    # time and not at transform time (#21251)
    vocabulary = ["Sample", "Upper", "Case", "Vocabulary"]
    message = (
        "Upper case characters found in"
        " vocabulary while 'lowercase'"
        " is True. These entries will not"
        " be matched with any documents"
    )

    vectorizer = CountVectorizer(lowercase=True, vocabulary=vocabulary)

    with pytest.warns(UserWarning, match=message):
        vectorizer.fit(vocabulary)

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        vectorizer.transform(vocabulary)


def test_tf_transformer_feature_names_out():
    # [FUNCTION DOC]
    # Checks TfidfTransformer.get_feature_names_out returns the input feature names
    # unchanged.
    """Check get_feature_names_out for TfidfTransformer"""
    X = [[1, 1, 1], [1, 1, 0], [1, 0, 0]]
    tr = TfidfTransformer(smooth_idf=True, norm="l2").fit(X)

    feature_names_in = ["a", "c", "b"]
    feature_names_out = tr.get_feature_names_out(feature_names_in)
    assert_array_equal(feature_names_in, feature_names_out)


def test_tf_idf_smoothing():
    # [FUNCTION DOC]
    # With smooth_idf=True and l2 norm, checks tf-idf values are non-negative, rows
    # have unit l2 norm, and an all-zero column does not cause problems.
    X = [[1, 1, 1], [1, 1, 0], [1, 0, 0]]
    tr = TfidfTransformer(smooth_idf=True, norm="l2")
    tfidf = tr.fit_transform(X).toarray()
    assert (tfidf >= 0).all()

    # check normalization
    assert_array_almost_equal((tfidf**2).sum(axis=1), [1.0, 1.0, 1.0])

    # this is robust to features with only zeros
    X = [[1, 1, 0], [1, 1, 0], [1, 0, 0]]
    tr = TfidfTransformer(smooth_idf=True, norm="l2")
    tfidf = tr.fit_transform(X).toarray()
    assert (tfidf >= 0).all()


@pytest.mark.xfail(
    _IS_WASM,
    reason=(
        "no floating point exceptions, see"
        " https://github.com/numpy/numpy/pull/21895#issuecomment-1311525881"
    ),
)
def test_tfidf_no_smoothing():
    # [FUNCTION DOC]
    # With smooth_idf=False, checks non-negativity and unit norms, and that an
    # all-zero column emits a "divide by zero" RuntimeWarning (xfail on WASM, which
    # lacks floating-point exceptions).
    X = [[1, 1, 1], [1, 1, 0], [1, 0, 0]]
    tr = TfidfTransformer(smooth_idf=False, norm="l2")
    tfidf = tr.fit_transform(X).toarray()
    assert (tfidf >= 0).all()

    # check normalization
    assert_array_almost_equal((tfidf**2).sum(axis=1), [1.0, 1.0, 1.0])

    # the lack of smoothing make IDF fragile in the presence of feature with
    # only zeros
    X = [[1, 1, 0], [1, 1, 0], [1, 0, 0]]
    tr = TfidfTransformer(smooth_idf=False, norm="l2")

    in_warning_message = "divide by zero"
    with pytest.warns(RuntimeWarning, match=in_warning_message):
        tr.fit_transform(X).toarray()


def test_sublinear_tf():
    # [FUNCTION DOC]
    # Checks sublinear_tf (1 + log tf) without idf or norm: value 1 for tf=1,
    # increasing but less than the raw count for tf=2 and 3.
    X = [[1], [2], [3]]
    tr = TfidfTransformer(sublinear_tf=True, use_idf=False, norm=None)
    tfidf = tr.fit_transform(X).toarray()
    assert tfidf[0] == 1
    assert tfidf[1] > tfidf[0]
    assert tfidf[2] > tfidf[1]
    assert tfidf[1] < 2
    assert tfidf[2] < 3


def test_vectorizer():
    # [FUNCTION DOC]
    # End-to-end check of CountVectorizer, TfidfTransformer and TfidfVectorizer on the
    # food corpus: max_df=0.5 removes frequent words ("the", "copyright"), counts in
    # train/test documents are as expected, a vectorizer with a fixed copied vocabulary
    # behaves the same, tf-idf rows are l2-normalized with non-negative values, the
    # idf_ is consistent between TfidfTransformer and TfidfVectorizer (and settable),
    # use_idf=False gives normalized raw tf, invalid norm raises ValueError, and
    # inverse/feature name outputs have the expected length; empty documents raise.
    # KNOWN ISSUE: inline comments are misleading: the "build a vectorizer v1" comment
    # builds v2, and "the" is excluded by max_df thresholding, not by a fixed stop
    # word list (no stop_words are set).
    # KNOWN ISSUE: near the end the test assigns a string to v3.set_params instead of
    # calling it, so the analyzer is never set to an invalid type; build_analyzer
    # still raises ValueError only because strip_accents is left at the invalid value.
    # raw documents as an iterator
    train_data = iter(ALL_FOOD_DOCS[:-1])
    test_data = [ALL_FOOD_DOCS[-1]]
    n_train = len(ALL_FOOD_DOCS) - 1

    # test without vocabulary
    v1 = CountVectorizer(max_df=0.5)
    counts_train = v1.fit_transform(train_data)
    if hasattr(counts_train, "tocsr"):
        counts_train = counts_train.tocsr()
    assert counts_train[0, v1.vocabulary_["pizza"]] == 2

    # build a vectorizer v1 with the same vocabulary as the one fitted by v1
    v2 = CountVectorizer(vocabulary=v1.vocabulary_)

    # compare that the two vectorizer give the same output on the test sample
    for v in (v1, v2):
        counts_test = v.transform(test_data)
        if hasattr(counts_test, "tocsr"):
            counts_test = counts_test.tocsr()

        vocabulary = v.vocabulary_
        assert counts_test[0, vocabulary["salad"]] == 1
        assert counts_test[0, vocabulary["tomato"]] == 1
        assert counts_test[0, vocabulary["water"]] == 1

        # stop word from the fixed list
        assert "the" not in vocabulary

        # stop word found automatically by the vectorizer DF thresholding
        # words that are high frequent across the complete corpus are likely
        # to be not informative (either real stop words of extraction
        # artifacts)
        assert "copyright" not in vocabulary

        # not present in the sample
        assert counts_test[0, vocabulary["coke"]] == 0
        assert counts_test[0, vocabulary["burger"]] == 0
        assert counts_test[0, vocabulary["beer"]] == 0
        assert counts_test[0, vocabulary["pizza"]] == 0

    # test tf-idf
    t1 = TfidfTransformer(norm="l1")
    tfidf = t1.fit(counts_train).transform(counts_train).toarray()
    assert len(t1.idf_) == len(v1.vocabulary_)
    assert tfidf.shape == (n_train, len(v1.vocabulary_))

    # test tf-idf with new data
    tfidf_test = t1.transform(counts_test).toarray()
    assert tfidf_test.shape == (len(test_data), len(v1.vocabulary_))

    # test tf alone
    t2 = TfidfTransformer(norm="l1", use_idf=False)
    tf = t2.fit(counts_train).transform(counts_train).toarray()
    assert not hasattr(t2, "idf_")

    # test idf transform with unlearned idf vector
    t3 = TfidfTransformer(use_idf=True)
    with pytest.raises(ValueError):
        t3.transform(counts_train)

    # L1-normalized term frequencies sum to one
    assert_array_almost_equal(np.sum(tf, axis=1), [1.0] * n_train)

    # test the direct tfidf vectorizer
    # (equivalent to term count vectorizer + tfidf transformer)
    train_data = iter(ALL_FOOD_DOCS[:-1])
    tv = TfidfVectorizer(norm="l1")

    tv.max_df = v1.max_df
    tfidf2 = tv.fit_transform(train_data).toarray()
    assert not tv.fixed_vocabulary_
    assert_array_almost_equal(tfidf, tfidf2)

    # test the direct tfidf vectorizer with new data
    tfidf_test2 = tv.transform(test_data).toarray()
    assert_array_almost_equal(tfidf_test, tfidf_test2)

    # test transform on unfitted vectorizer with empty vocabulary
    v3 = CountVectorizer(vocabulary=None)
    with pytest.raises(ValueError):
        v3.transform(train_data)

    # ascii preprocessor?
    v3.set_params(strip_accents="ascii", lowercase=False)
    processor = v3.build_preprocessor()
    text = "J'ai mangé du kangourou  ce midi, c'était pas très bon."
    expected = strip_accents_ascii(text)
    result = processor(text)
    assert expected == result

    # error on bad strip_accents param
    v3.set_params(strip_accents="_gabbledegook_", preprocessor=None)
    with pytest.raises(ValueError):
        v3.build_preprocessor()

    # error with bad analyzer type
    v3.set_params = "_invalid_analyzer_type_"
    with pytest.raises(ValueError):
        v3.build_analyzer()


def test_tfidf_vectorizer_setters():
    # [FUNCTION DOC]
    # Checks TfidfVectorizer forwards norm, use_idf, smooth_idf and sublinear_tf to
    # its internal _tfidf transformer at fit time only: changing them on the
    # vectorizer has no effect on _tfidf until the next fit.
    norm, use_idf, smooth_idf, sublinear_tf = "l2", False, False, False
    tv = TfidfVectorizer(
        norm=norm, use_idf=use_idf, smooth_idf=smooth_idf, sublinear_tf=sublinear_tf
    )
    tv.fit(JUNK_FOOD_DOCS)
    assert tv._tfidf.norm == norm
    assert tv._tfidf.use_idf == use_idf
    assert tv._tfidf.smooth_idf == smooth_idf
    assert tv._tfidf.sublinear_tf == sublinear_tf

    # assigning value to `TfidfTransformer` should not have any effect until
    # fitting
    tv.norm = "l1"
    tv.use_idf = True
    tv.smooth_idf = True
    tv.sublinear_tf = True
    assert tv._tfidf.norm == norm
    assert tv._tfidf.use_idf == use_idf
    assert tv._tfidf.smooth_idf == smooth_idf
    assert tv._tfidf.sublinear_tf == sublinear_tf

    tv.fit(JUNK_FOOD_DOCS)
    assert tv._tfidf.norm == tv.norm
    assert tv._tfidf.use_idf == tv.use_idf
    assert tv._tfidf.smooth_idf == tv.smooth_idf
    assert tv._tfidf.sublinear_tf == tv.sublinear_tf


def test_hashing_vectorizer():
    # [FUNCTION DOC]
    # Checks HashingVectorizer default output shape and dtype, random-sign values
    # strictly inside (-1, 1) with both signs present, unit l2 row norms; then with
    # bigrams and l1 norm: more nonzeros than unigrams but fewer than twice as many,
    # bounded values and unit l1 row norms.
    v = HashingVectorizer()
    X = v.transform(ALL_FOOD_DOCS)
    token_nnz = X.nnz
    assert X.shape == (len(ALL_FOOD_DOCS), v.n_features)
    assert X.dtype == v.dtype

    # By default the hashed values receive a random sign and l2 normalization
    # makes the feature values bounded
    assert np.min(X.data) > -1
    assert np.min(X.data) < 0
    assert np.max(X.data) > 0
    assert np.max(X.data) < 1

    # Check that the rows are normalized (l2 norm)
    for row in X:
        assert_almost_equal(np.linalg.norm(row.data, 2), 1.0)

    # Check vectorization with some non-default parameters
    v = HashingVectorizer(ngram_range=(1, 2), norm="l1")
    X = v.transform(ALL_FOOD_DOCS)
    assert X.shape == (len(ALL_FOOD_DOCS), v.n_features)
    assert X.dtype == v.dtype

    # ngrams generate more non zeros
    ngrams_nnz = X.nnz
    assert ngrams_nnz > token_nnz
    assert ngrams_nnz < 2 * token_nnz

    # makes the feature values bounded
    assert np.min(X.data) > -1
    assert np.max(X.data) < 1

    # Check that the rows are normalized (l1 norm)
    for row in X:
        assert_almost_equal(np.linalg.norm(row.data, 1), 1.0)


def test_feature_names():
    # [FUNCTION DOC]
    # Checks get_feature_names_out raises ValueError before fit, that the learned
    # names (max_df=0.5) are the expected sorted list as an object ndarray consistent
    # with vocabulary_, and that a fixed list vocabulary yields the same names without
    # fitting and sets fixed_vocabulary_.
    cv = CountVectorizer(max_df=0.5)

    # test for Value error on unfitted/empty vocabulary
    with pytest.raises(ValueError):
        cv.get_feature_names_out()
    assert not cv.fixed_vocabulary_

    # test for vocabulary learned from data
    X = cv.fit_transform(ALL_FOOD_DOCS)
    n_samples, n_features = X.shape
    assert len(cv.vocabulary_) == n_features

    feature_names = cv.get_feature_names_out()
    assert isinstance(feature_names, np.ndarray)
    assert feature_names.dtype == object

    assert len(feature_names) == n_features
    assert_array_equal(
        [
            "beer",
            "burger",
            "celeri",
            "coke",
            "pizza",
            "salad",
            "sparkling",
            "tomato",
            "water",
        ],
        feature_names,
    )

    for idx, name in enumerate(feature_names):
        assert idx == cv.vocabulary_.get(name)

    # test for custom vocabulary
    vocab = [
        "beer",
        "burger",
        "celeri",
        "coke",
        "pizza",
        "salad",
        "sparkling",
        "tomato",
        "water",
    ]

    cv = CountVectorizer(vocabulary=vocab)
    feature_names = cv.get_feature_names_out()
    assert_array_equal(
        [
            "beer",
            "burger",
            "celeri",
            "coke",
            "pizza",
            "salad",
            "sparkling",
            "tomato",
            "water",
        ],
        feature_names,
    )
    assert cv.fixed_vocabulary_

    for idx, name in enumerate(feature_names):
        assert idx == cv.vocabulary_.get(name)


@pytest.mark.parametrize("Vectorizer", (CountVectorizer, TfidfVectorizer))
def test_vectorizer_max_features(Vectorizer):
    # [FUNCTION DOC]
    # For CountVectorizer and TfidfVectorizer, checks max_features=4 with max_df=0.6
    # keeps the 4 most frequent remaining terms.
    expected_vocabulary = {"burger", "beer", "salad", "pizza"}

    # test bounded number of extracted features
    vectorizer = Vectorizer(max_df=0.6, max_features=4)
    vectorizer.fit(ALL_FOOD_DOCS)
    assert set(vectorizer.vocabulary_) == expected_vocabulary


def test_count_vectorizer_max_features():
    # [FUNCTION DOC]
    # Regression (0.14): with max_features of 1, 3 and None on the junk corpus, the
    # most frequent term "the" (count 7) is always kept and identified via argmax.
    # Regression test: max_features didn't work correctly in 0.14.

    cv_1 = CountVectorizer(max_features=1)
    cv_3 = CountVectorizer(max_features=3)
    cv_None = CountVectorizer(max_features=None)

    counts_1 = cv_1.fit_transform(JUNK_FOOD_DOCS).sum(axis=0)
    counts_3 = cv_3.fit_transform(JUNK_FOOD_DOCS).sum(axis=0)
    counts_None = cv_None.fit_transform(JUNK_FOOD_DOCS).sum(axis=0)

    features_1 = cv_1.get_feature_names_out()
    features_3 = cv_3.get_feature_names_out()
    features_None = cv_None.get_feature_names_out()

    # The most common feature is "the", with frequency 7.
    assert 7 == counts_1.max()
    assert 7 == counts_3.max()
    assert 7 == counts_None.max()

    # The most common feature should be the same
    assert "the" == features_1[np.argmax(counts_1)]
    assert "the" == features_3[np.argmax(counts_3)]
    assert "the" == features_None[np.argmax(counts_None)]


def test_vectorizer_max_df():
    # [FUNCTION DOC]
    # Checks max_df on char analyzer: 1.0 keeps all 6 chars, while 0.5 (float
    # proportion, threshold 1.5 docs) and 1 (absolute count) both drop "a" and "e",
    # leaving 4 terms.
    test_data = ["abc", "dea", "eat"]
    vect = CountVectorizer(analyzer="char", max_df=1.0)
    vect.fit(test_data)
    assert "a" in vect.vocabulary_.keys()
    assert len(vect.vocabulary_.keys()) == 6

    vect.max_df = 0.5  # 0.5 * 3 documents -> max_doc_count == 1.5
    vect.fit(test_data)
    assert "a" not in vect.vocabulary_.keys()  # {ae} ignored
    assert len(vect.vocabulary_.keys()) == 4  # {bcdt} remain

    vect.max_df = 1
    vect.fit(test_data)
    assert "a" not in vect.vocabulary_.keys()  # {ae} ignored
    assert len(vect.vocabulary_.keys()) == 4  # {bcdt} remain


def test_vectorizer_min_df():
    # [FUNCTION DOC]
    # Checks min_df on char analyzer: 1 keeps all 6, integer 2 keeps only "a" and
    # "e", proportion 0.8 (threshold 2.4 docs) keeps only "a".
    test_data = ["abc", "dea", "eat"]
    vect = CountVectorizer(analyzer="char", min_df=1)
    vect.fit(test_data)
    assert "a" in vect.vocabulary_.keys()
    assert len(vect.vocabulary_.keys()) == 6

    vect.min_df = 2
    vect.fit(test_data)
    assert "c" not in vect.vocabulary_.keys()  # {bcdt} ignored
    assert len(vect.vocabulary_.keys()) == 2  # {ae} remain

    vect.min_df = 0.8  # 0.8 * 3 documents -> min_doc_count == 2.4
    vect.fit(test_data)
    assert "c" not in vect.vocabulary_.keys()  # {bcdet} ignored
    assert len(vect.vocabulary_.keys()) == 1  # {a} remains


def test_count_binary_occurrences():
    # [FUNCTION DOC]
    # Checks CountVectorizer char counts, binary=True clipping counts to 1, and that
    # dtype=np.float32 sets the output dtype.
    # by default multiple occurrences are counted as longs
    test_data = ["aaabc", "abbde"]
    vect = CountVectorizer(analyzer="char", max_df=1.0)
    X = vect.fit_transform(test_data).toarray()
    assert_array_equal(["a", "b", "c", "d", "e"], vect.get_feature_names_out())
    assert_array_equal([[3, 1, 1, 0, 0], [1, 2, 0, 1, 1]], X)

    # using boolean features, we can fetch the binary occurrence info
    # instead.
    vect = CountVectorizer(analyzer="char", max_df=1.0, binary=True)
    X = vect.fit_transform(test_data).toarray()
    assert_array_equal([[1, 1, 1, 0, 0], [1, 1, 0, 1, 1]], X)

    # check the ability to change the dtype
    vect = CountVectorizer(analyzer="char", max_df=1.0, binary=True, dtype=np.float32)
    X_sparse = vect.fit_transform(test_data)
    assert X_sparse.dtype == np.float32


def test_hashed_binary_occurrences():
    # [FUNCTION DOC]
    # Checks HashingVectorizer (no sign alternation, no norm) gives per-row maximum
    # counts 3 and 2, binary=True gives max 1, and the dtype is float64 by default and
    # when set explicitly.
    # KNOWN ISSUE: comment "counted as longs" is inaccurate; values are float64.
    # by default multiple occurrences are counted as longs
    test_data = ["aaabc", "abbde"]
    vect = HashingVectorizer(alternate_sign=False, analyzer="char", norm=None)
    X = vect.transform(test_data)
    assert np.max(X[0:1].data) == 3
    assert np.max(X[1:2].data) == 2
    assert X.dtype == np.float64

    # using boolean features, we can fetch the binary occurrence info
    # instead.
    vect = HashingVectorizer(
        analyzer="char", alternate_sign=False, binary=True, norm=None
    )
    X = vect.transform(test_data)
    assert np.max(X.data) == 1
    assert X.dtype == np.float64

    # check the ability to change the dtype
    vect = HashingVectorizer(
        analyzer="char", alternate_sign=False, binary=True, norm=None, dtype=np.float64
    )
    X = vect.transform(test_data)
    assert X.dtype == np.float64


@pytest.mark.parametrize("Vectorizer", (CountVectorizer, TfidfVectorizer))
def test_vectorizer_inverse_transform(Vectorizer):
    # [FUNCTION DOC]
    # For CountVectorizer and TfidfVectorizer, checks inverse_transform returns a
    # list whose per-document term sets equal the analyzer's unique tokens, that
    # fit_transform outputs CSR, and that dense and CSC inputs give the same result.
    # raw documents
    data = ALL_FOOD_DOCS
    vectorizer = Vectorizer()
    transformed_data = vectorizer.fit_transform(data)
    inversed_data = vectorizer.inverse_transform(transformed_data)
    assert isinstance(inversed_data, list)

    analyze = vectorizer.build_analyzer()
    for doc, inversed_terms in zip(data, inversed_data):
        terms = np.sort(np.unique(analyze(doc)))
        inversed_terms = np.sort(np.unique(inversed_terms))
        assert_array_equal(terms, inversed_terms)

    assert sparse.issparse(transformed_data)
    assert transformed_data.format == "csr"

    # Test that inverse_transform also works with numpy arrays and
    # scipy
    transformed_data2 = transformed_data.toarray()
    inversed_data2 = vectorizer.inverse_transform(transformed_data2)
    for terms, terms2 in zip(inversed_data, inversed_data2):
        assert_array_equal(np.sort(terms), np.sort(terms2))

    # Check that inverse_transform also works on non CSR sparse data:
    transformed_data3 = transformed_data.tocsc()
    inversed_data3 = vectorizer.inverse_transform(transformed_data3)
    for terms, terms3 in zip(inversed_data, inversed_data3):
        assert_array_equal(np.sort(terms), np.sort(terms3))


def test_count_vectorizer_pipeline_grid_selection():
    # [FUNCTION DOC]
    # Grid-searches ngram_range and LinearSVC loss in a CountVectorizer pipeline on a
    # train split (random_state=0, 20% test, cv=3) and checks perfect test
    # predictions, best CV score 1.0 and that ngram_range (1, 1) is selected.
    # KNOWN ISSUE: the inline comment says the bigram model is chosen, while the
    # assertion expects (1, 1) (first of tied candidates).
    # raw documents
    data = JUNK_FOOD_DOCS + NOTJUNK_FOOD_DOCS

    # label junk food as -1, the others as +1
    target = [-1] * len(JUNK_FOOD_DOCS) + [1] * len(NOTJUNK_FOOD_DOCS)

    # split the dataset for model development and final evaluation
    train_data, test_data, target_train, target_test = train_test_split(
        data, target, test_size=0.2, random_state=0
    )

    pipeline = Pipeline([("vect", CountVectorizer()), ("svc", LinearSVC())])

    parameters = {
        "vect__ngram_range": [(1, 1), (1, 2)],
        "svc__loss": ("hinge", "squared_hinge"),
    }

    # find the best parameters for both the feature extraction and the
    # classifier
    grid_search = GridSearchCV(pipeline, parameters, n_jobs=1, cv=3)

    # Check that the best model found by grid search is 100% correct on the
    # held out evaluation set.
    pred = grid_search.fit(train_data, target_train).predict(test_data)
    assert_array_equal(pred, target_test)

    # on this toy dataset bigram representation which is used in the last of
    # the grid_search is considered the best estimator since they all converge
    # to 100% accuracy models
    assert grid_search.best_score_ == 1.0
    best_vectorizer = grid_search.best_estimator_.named_steps["vect"]
    assert best_vectorizer.ngram_range == (1, 1)


def test_vectorizer_pipeline_grid_selection():
    # [FUNCTION DOC]
    # Same as the count-vectorizer grid test with TfidfVectorizer and an extra norm
    # grid (10% test split, default cv): perfect predictions, best score 1.0, chosen
    # ngram_range (1, 1), norm "l2" and no fixed vocabulary.
    # KNOWN ISSUE: same misleading bigram comment as the CountVectorizer variant.
    # raw documents
    data = JUNK_FOOD_DOCS + NOTJUNK_FOOD_DOCS

    # label junk food as -1, the others as +1
    target = [-1] * len(JUNK_FOOD_DOCS) + [1] * len(NOTJUNK_FOOD_DOCS)

    # split the dataset for model development and final evaluation
    train_data, test_data, target_train, target_test = train_test_split(
        data, target, test_size=0.1, random_state=0
    )

    pipeline = Pipeline([("vect", TfidfVectorizer()), ("svc", LinearSVC())])

    parameters = {
        "vect__ngram_range": [(1, 1), (1, 2)],
        "vect__norm": ("l1", "l2"),
        "svc__loss": ("hinge", "squared_hinge"),
    }

    # find the best parameters for both the feature extraction and the
    # classifier
    grid_search = GridSearchCV(pipeline, parameters, n_jobs=1)

    # Check that the best model found by grid search is 100% correct on the
    # held out evaluation set.
    pred = grid_search.fit(train_data, target_train).predict(test_data)
    assert_array_equal(pred, target_test)

    # on this toy dataset bigram representation which is used in the last of
    # the grid_search is considered the best estimator since they all converge
    # to 100% accuracy models
    assert grid_search.best_score_ == 1.0
    best_vectorizer = grid_search.best_estimator_.named_steps["vect"]
    assert best_vectorizer.ngram_range == (1, 1)
    assert best_vectorizer.norm == "l2"
    assert not best_vectorizer.fixed_vocabulary_


def test_vectorizer_pipeline_cross_validation():
    # [FUNCTION DOC]
    # Checks 3-fold cross_val_score of a TfidfVectorizer + LinearSVC pipeline is 1.0
    # on every fold.
    # raw documents
    data = JUNK_FOOD_DOCS + NOTJUNK_FOOD_DOCS

    # label junk food as -1, the others as +1
    target = [-1] * len(JUNK_FOOD_DOCS) + [1] * len(NOTJUNK_FOOD_DOCS)

    pipeline = Pipeline([("vect", TfidfVectorizer()), ("svc", LinearSVC())])

    cv_scores = cross_val_score(pipeline, data, target, cv=3)
    assert_array_equal(cv_scores, [1.0, 1.0, 1.0])


def test_vectorizer_unicode():
    # [FUNCTION DOC]
    # Checks CountVectorizer tokenizes a Cyrillic sentence into 12 features and that
    # HashingVectorizer (no norm, no sign) gives the same nnz and same sorted counts
    # (no collisions).
    # tests that the count vectorizer works with cyrillic.
    document = (
        "Машинное обучение — обширный подраздел искусственного "
        "интеллекта, изучающий методы построения алгоритмов, "
        "способных обучаться."
    )

    vect = CountVectorizer()
    X_counted = vect.fit_transform([document])
    assert X_counted.shape == (1, 12)

    vect = HashingVectorizer(norm=None, alternate_sign=False)
    X_hashed = vect.transform([document])
    assert X_hashed.shape == (1, 2**20)

    # No collisions on such a small dataset
    assert X_counted.nnz == X_hashed.nnz

    # When norm is None and not alternate_sign, the tokens are counted up to
    # collisions
    assert_array_equal(np.sort(X_counted.data), np.sort(X_hashed.data))


def test_tfidf_vectorizer_with_fixed_vocabulary():
    # [FUNCTION DOC]
    # Smoke test: TfidfVectorizer with a fixed list vocabulary gives the same result
    # for fit_transform and transform and reports fixed_vocabulary_.
    # non regression smoke test for inheritance issues
    vocabulary = ["pizza", "celeri"]
    vect = TfidfVectorizer(vocabulary=vocabulary)
    X_1 = vect.fit_transform(ALL_FOOD_DOCS)
    X_2 = vect.transform(ALL_FOOD_DOCS)
    assert_array_almost_equal(X_1.toarray(), X_2.toarray())
    assert vect.fixed_vocabulary_


def test_pickling_vectorizer():
    # [FUNCTION DOC]
    # Pickles and unpickles a range of fitted and unfitted vectorizers (including
    # custom preprocessors/analyzers/strip_accents callables) and checks type,
    # get_params and fit_transform output are preserved.
    instances = [
        HashingVectorizer(),
        HashingVectorizer(norm="l1"),
        HashingVectorizer(binary=True),
        HashingVectorizer(ngram_range=(1, 2)),
        CountVectorizer(),
        CountVectorizer(preprocessor=strip_tags),
        CountVectorizer(analyzer=lazy_analyze),
        CountVectorizer(preprocessor=strip_tags).fit(JUNK_FOOD_DOCS),
        CountVectorizer(strip_accents=strip_eacute).fit(JUNK_FOOD_DOCS),
        TfidfVectorizer(),
        TfidfVectorizer(analyzer=lazy_analyze),
        TfidfVectorizer().fit(JUNK_FOOD_DOCS),
    ]

    for orig in instances:
        s = pickle.dumps(orig)
        copy = pickle.loads(s)
        assert type(copy) == orig.__class__
        assert copy.get_params() == orig.get_params()
        assert_allclose_dense_sparse(
            copy.fit_transform(JUNK_FOOD_DOCS),
            orig.fit_transform(JUNK_FOOD_DOCS),
        )


@pytest.mark.parametrize(
    "factory",
    [
        CountVectorizer.build_analyzer,
        CountVectorizer.build_preprocessor,
        CountVectorizer.build_tokenizer,
    ],
)
def test_pickling_built_processors(factory):
    # [FUNCTION DOC]
    # Non-regression (#12833): analyzers, preprocessors and tokenizers built by
    # CountVectorizer can be pickled and give identical output after round trip.
    """Tokenizers cannot be pickled
    https://github.com/scikit-learn/scikit-learn/issues/12833
    """
    vec = CountVectorizer()
    function = factory(vec)
    text = "J'ai mangé du kangourou  ce midi, c'était pas très bon."
    roundtripped_function = pickle.loads(pickle.dumps(function))
    expected = function(text)
    result = roundtripped_function(text)
    assert result == expected


def test_countvectorizer_vocab_sets_when_pickling():
    # [FUNCTION DOC]
    # For 100 random 5-word set vocabularies, checks a pickled-then-unpickled
    # CountVectorizer yields the same feature names as the original after fit
    # (set ordering preserved through pickling).
    # ensure that vocabulary of type set is coerced to a list to
    # preserve iteration ordering after deserialization
    rng = np.random.RandomState(0)
    vocab_words = np.array(
        [
            "beer",
            "burger",
            "celeri",
            "coke",
            "pizza",
            "salad",
            "sparkling",
            "tomato",
            "water",
        ]
    )
    for x in range(0, 100):
        vocab_set = set(rng.choice(vocab_words, size=5, replace=False))
        cv = CountVectorizer(vocabulary=vocab_set)
        unpickled_cv = pickle.loads(pickle.dumps(cv))
        cv.fit(ALL_FOOD_DOCS)
        unpickled_cv.fit(ALL_FOOD_DOCS)
        assert_array_equal(
            cv.get_feature_names_out(), unpickled_cv.get_feature_names_out()
        )


def test_countvectorizer_vocab_dicts_when_pickling():
    # [FUNCTION DOC]
    # For 100 random 5-word dict vocabularies with indices 0..4, checks a pickled
    # round-trip CountVectorizer yields the same feature names after fit.
    rng = np.random.RandomState(0)
    vocab_words = np.array(
        [
            "beer",
            "burger",
            "celeri",
            "coke",
            "pizza",
            "salad",
            "sparkling",
            "tomato",
            "water",
        ]
    )
    for x in range(0, 100):
        vocab_dict = dict()
        words = rng.choice(vocab_words, size=5, replace=False)
        for y in range(0, 5):
            vocab_dict[words[y]] = y
        cv = CountVectorizer(vocabulary=vocab_dict)
        unpickled_cv = pickle.loads(pickle.dumps(cv))
        cv.fit(ALL_FOOD_DOCS)
        unpickled_cv.fit(ALL_FOOD_DOCS)
        assert_array_equal(
            cv.get_feature_names_out(), unpickled_cv.get_feature_names_out()
        )


def test_pickling_transformer():
    # [FUNCTION DOC]
    # Checks a fitted TfidfTransformer survives pickling: same type and identical
    # fit_transform output on the count matrix.
    X = CountVectorizer().fit_transform(JUNK_FOOD_DOCS)
    orig = TfidfTransformer().fit(X)
    s = pickle.dumps(orig)
    copy = pickle.loads(s)
    assert type(copy) == orig.__class__
    assert_array_equal(copy.fit_transform(X).toarray(), orig.fit_transform(X).toarray())


def test_transformer_idf_setter():
    # [FUNCTION DOC]
    # Checks assigning idf_ from a fitted TfidfTransformer to a fresh one makes its
    # transform match the original.
    X = CountVectorizer().fit_transform(JUNK_FOOD_DOCS)
    orig = TfidfTransformer().fit(X)
    copy = TfidfTransformer()
    copy.idf_ = orig.idf_
    assert_array_equal(copy.transform(X).toarray(), orig.transform(X).toarray())


def test_tfidf_vectorizer_setter():
    # [FUNCTION DOC]
    # Checks setting idf_ on a TfidfVectorizer with a fixed vocabulary reproduces the
    # original transform, and that setting idf_ with use_idf=False raises ValueError
    # with the documented message.
    orig = TfidfVectorizer(use_idf=True)
    orig.fit(JUNK_FOOD_DOCS)
    copy = TfidfVectorizer(vocabulary=orig.vocabulary_, use_idf=True)
    copy.idf_ = orig.idf_
    assert_array_equal(
        copy.transform(JUNK_FOOD_DOCS).toarray(),
        orig.transform(JUNK_FOOD_DOCS).toarray(),
    )
    # `idf_` cannot be set with `use_idf=False`
    copy = TfidfVectorizer(vocabulary=orig.vocabulary_, use_idf=False)
    err_msg = "`idf_` cannot be set when `user_idf=False`."
    with pytest.raises(ValueError, match=err_msg):
        copy.idf_ = orig.idf_


def test_tfidfvectorizer_invalid_idf_attr():
    # [FUNCTION DOC]
    # Checks setting an idf_ whose length differs from the vocabulary size raises
    # ValueError.
    vect = TfidfVectorizer(use_idf=True)
    vect.fit(JUNK_FOOD_DOCS)
    copy = TfidfVectorizer(vocabulary=vect.vocabulary_, use_idf=True)
    expected_idf_len = len(vect.idf_)
    invalid_idf = [1.0] * (expected_idf_len + 1)
    with pytest.raises(ValueError):
        setattr(copy, "idf_", invalid_idf)


def test_non_unique_vocab():
    # [FUNCTION DOC]
    # Checks a list vocabulary with duplicate terms makes fit raise ValueError.
    vocab = ["a", "b", "c", "a", "a"]
    vect = CountVectorizer(vocabulary=vocab)
    with pytest.raises(ValueError):
        vect.fit([])


def test_hashingvectorizer_nan_in_docs():
    # [FUNCTION DOC]
    # Checks HashingVectorizer.fit_transform raises ValueError with an explicit
    # message when a document is np.nan.
    # np.nan can appear when using pandas to load text fields from a csv file
    # with missing values.
    message = "np.nan is an invalid document, expected byte or unicode string."
    exception = ValueError

    def func():
        # [FUNCTION DOC]
        # HELPER: Builds a default HashingVectorizer and fits it on a corpus containing
        # np.nan, which is expected to raise.
        hv = HashingVectorizer()
        hv.fit_transform(["hello world", np.nan, "hello hello"])

    with pytest.raises(exception, match=message):
        func()


def test_tfidfvectorizer_binary():
    # [FUNCTION DOC]
    # Non-regression: TfidfVectorizer honours binary=True (without idf and norm)
    # in both fit_transform and transform, giving 0/1 counts.
    # Non-regression test: TfidfVectorizer used to ignore its "binary" param.
    v = TfidfVectorizer(binary=True, use_idf=False, norm=None)
    assert v.binary

    X = v.fit_transform(["hello world", "hello hello"]).toarray()
    assert_array_equal(X.ravel(), [1, 1, 1, 0])
    X2 = v.transform(["hello world", "hello hello"]).toarray()
    assert_array_equal(X2.ravel(), [1, 1, 1, 0])


def test_tfidfvectorizer_export_idf():
    # [FUNCTION DOC]
    # Checks TfidfVectorizer.idf_ exposes the internal _tfidf.idf_ values.
    vect = TfidfVectorizer(use_idf=True)
    vect.fit(JUNK_FOOD_DOCS)
    assert_array_almost_equal(vect.idf_, vect._tfidf.idf_)


def test_vectorizer_vocab_clone():
    # [FUNCTION DOC]
    # Checks a clone of a TfidfVectorizer with a fixed vocabulary learns the same
    # vocabulary_ as the original after fitting.
    vect_vocab = TfidfVectorizer(vocabulary=["the"])
    vect_vocab_clone = clone(vect_vocab)
    vect_vocab.fit(ALL_FOOD_DOCS)
    vect_vocab_clone.fit(ALL_FOOD_DOCS)
    assert vect_vocab_clone.vocabulary_ == vect_vocab.vocabulary_


@pytest.mark.parametrize(
    "Vectorizer", (CountVectorizer, TfidfVectorizer, HashingVectorizer)
)
def test_vectorizer_string_object_as_input(Vectorizer):
    # [FUNCTION DOC]
    # For Count, Tfidf and Hashing vectorizers, checks fit_transform, fit and
    # transform raise ValueError when given a single string instead of an iterable of
    # documents.
    message = "Iterable over raw text documents expected, string object received."
    vec = Vectorizer()

    with pytest.raises(ValueError, match=message):
        vec.fit_transform("hello world!")

    with pytest.raises(ValueError, match=message):
        vec.fit("hello world!")
    vec.fit(["some text", "some other text"])

    with pytest.raises(ValueError, match=message):
        vec.transform("hello world!")


@pytest.mark.parametrize("X_dtype", [np.float32, np.float64])
def test_tfidf_transformer_type(X_dtype):
    # [FUNCTION DOC]
    # Checks TfidfTransformer preserves float32/float64 dtype of a random sparse
    # input.
    X = _sparse_random_array((10, 20000), dtype=X_dtype, random_state=42)
    X_trans = TfidfTransformer().fit_transform(X)
    assert X_trans.dtype == X.dtype


@pytest.mark.parametrize(
    "csc_container, csr_container", product(CSC_CONTAINERS, CSR_CONTAINERS)
)
def test_tfidf_transformer_sparse(csc_container, csr_container):
    # [FUNCTION DOC]
    # For every CSC/CSR container pair, checks TfidfTransformer gives numerically
    # equal results and the same output format for CSC and CSR inputs.
    X = _sparse_random_array((10, 20000), dtype=np.float64, random_state=42)
    X_csc = csc_container(X)
    X_csr = csr_container(X)

    X_trans_csc = TfidfTransformer().fit_transform(X_csc)
    X_trans_csr = TfidfTransformer().fit_transform(X_csr)
    assert_allclose_dense_sparse(X_trans_csc, X_trans_csr)
    assert X_trans_csc.format == X_trans_csr.format


@pytest.mark.parametrize(
    "vectorizer_dtype, output_dtype, warning_expected",
    [
        (np.int32, np.float64, True),
        (np.int64, np.float64, True),
        (np.float32, np.float32, False),
        (np.float64, np.float64, False),
    ],
)
def test_tfidf_vectorizer_type(vectorizer_dtype, output_dtype, warning_expected):
    # [FUNCTION DOC]
    # Checks TfidfVectorizer warns (UserWarning about 'dtype') and outputs float64
    # for integer dtypes, and outputs float32/float64 without warning for float dtypes.
    X = np.array(["numpy", "scipy", "sklearn"])
    vectorizer = TfidfVectorizer(dtype=vectorizer_dtype)

    warning_msg_match = "'dtype' should be used."
    if warning_expected:
        with pytest.warns(UserWarning, match=warning_msg_match):
            X_idf = vectorizer.fit_transform(X)
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            X_idf = vectorizer.fit_transform(X)
    assert X_idf.dtype == output_dtype


@pytest.mark.parametrize(
    "vec",
    [
        HashingVectorizer(ngram_range=(2, 1)),
        CountVectorizer(ngram_range=(2, 1)),
        TfidfVectorizer(ngram_range=(2, 1)),
    ],
)
def test_vectorizers_invalid_ngram_range(vec):
    # [FUNCTION DOC]
    # For each vectorizer constructed with ngram_range=(2, 1), checks fit and
    # fit_transform (and transform for HashingVectorizer) raise ValueError with the
    # "lower boundary larger than the upper boundary" message.
    # vectorizers could be initialized with invalid ngram range
    # test for raising error message
    invalid_range = vec.ngram_range
    message = re.escape(
        f"Invalid value for ngram_range={invalid_range} "
        "lower boundary larger than the upper boundary."
    )

    with pytest.raises(ValueError, match=message):
        vec.fit(["good news everyone"])

    with pytest.raises(ValueError, match=message):
        vec.fit_transform(["good news everyone"])

    if isinstance(vec, HashingVectorizer):
        with pytest.raises(ValueError, match=message):
            vec.transform(["good news everyone"])


def _check_stop_words_consistency(estimator):
    # [FUNCTION DOC]
    # HELPER: Calls the estimator's private _check_stop_words_consistency with its
    # stop words, preprocessor and tokenizer. Outputs: True (consistent), False
    # (inconsistent, warns), "error" (check failed), or None (already checked for this
    # stop list).
    stop_words = estimator.get_stop_words()
    tokenize = estimator.build_tokenizer()
    preprocess = estimator.build_preprocessor()
    return estimator._check_stop_words_consistency(stop_words, preprocess, tokenize)


def test_vectorizer_stop_words_inconsistent():
    # [FUNCTION DOC]
    # For each vectorizer: stop words whose tokenization yields tokens not in the list
    # trigger a UserWarning listing them on fit; after deleting the cached
    # _stop_words_id the helper warns and returns False; refitting does not warn
    # again and the helper returns None (cached); a new stop list warns again.
    lstr = r"\['and', 'll', 've'\]"
    message = (
        "Your stop_words may be inconsistent with your "
        "preprocessing. Tokenizing the stop words generated "
        "tokens %s not in stop_words." % lstr
    )
    for vec in [CountVectorizer(), TfidfVectorizer(), HashingVectorizer()]:
        vec.set_params(stop_words=["you've", "you", "you'll", "AND"])
        with pytest.warns(UserWarning, match=message):
            vec.fit_transform(["hello world"])
        # reset stop word validation
        del vec._stop_words_id
        with pytest.warns(UserWarning, match=message):
            assert _check_stop_words_consistency(vec) is False

        # Only one warning per stop list
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            vec.fit_transform(["hello world"])
        assert _check_stop_words_consistency(vec) is None

        # Test caching of inconsistency assessment
        vec.set_params(stop_words=["you've", "you", "you'll", "blah", "AND"])
        with pytest.warns(UserWarning, match=message):
            vec.fit_transform(["hello world"])


@skip_if_32bit
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_countvectorizer_sort_features_64bit_sparse_indices(csr_container):
    # [FUNCTION DOC]
    # Checks CountVectorizer._sort_features keeps int64 indices dtype of a CSR
    # matrix whose indices/indptr were forced to int64 (skipped on 32-bit).
    """
    Check that CountVectorizer._sort_features preserves the dtype of its sparse
    feature matrix.

    This test is skipped on 32bit platforms, see:
        https://github.com/scikit-learn/scikit-learn/pull/11295
    for more details.
    """

    X = csr_container((5, 5), dtype=np.int64)

    # force indices and indptr to int64.
    INDICES_DTYPE = np.int64
    X.indices = X.indices.astype(INDICES_DTYPE)
    X.indptr = X.indptr.astype(INDICES_DTYPE)

    vocabulary = {"scikit-learn": 0, "is": 1, "great!": 2}

    Xs = CountVectorizer()._sort_features(X, vocabulary)

    assert INDICES_DTYPE == Xs.indices.dtype


@pytest.mark.parametrize(
    "Estimator", [CountVectorizer, TfidfVectorizer, HashingVectorizer]
)
def test_stop_word_validation_custom_preprocessor(Estimator):
    # [FUNCTION DOC]
    # Checks the stop-word consistency helper returns True for defaults, "error" then
    # None (cached) when a preprocessor expects dicts, "error" for a subclass
    # overriding build_preprocessor, and True with a custom regex tokenizer; also that
    # fit_transform still works with the dict-based preprocessor.
    data = [{"text": "some text"}]

    vec = Estimator()
    assert _check_stop_words_consistency(vec) is True

    vec = Estimator(preprocessor=lambda x: x["text"], stop_words=["and"])
    assert _check_stop_words_consistency(vec) == "error"
    # checks are cached
    assert _check_stop_words_consistency(vec) is None
    vec.fit_transform(data)

    class CustomEstimator(Estimator):
        # [CLASS DOC]
        # HELPER: Subclass of the parametrized estimator whose preprocessor extracts
        # the "text" field from dict documents.
        def build_preprocessor(self):
            # [FUNCTION DOC]
            # HELPER: Returns a preprocessor that indexes a document by "text".
            return lambda x: x["text"]

    vec = CustomEstimator(stop_words=["and"])
    assert _check_stop_words_consistency(vec) == "error"

    vec = Estimator(
        tokenizer=lambda doc: re.compile(r"\w{1,}").findall(doc), stop_words=["and"]
    )
    assert _check_stop_words_consistency(vec) is True


@pytest.mark.parametrize(
    "Estimator", [CountVectorizer, TfidfVectorizer, HashingVectorizer]
)
@pytest.mark.parametrize(
    "input_type, err_type, err_msg",
    [
        ("filename", FileNotFoundError, ""),
        ("file", AttributeError, "'str' object has no attribute 'read'"),
    ],
)
def test_callable_analyzer_error(Estimator, input_type, err_type, err_msg):
    # [FUNCTION DOC]
    # Checks a callable analyzer with input="filename" raises FileNotFoundError and
    # with input="file" raises AttributeError (str has no read), since documents are
    # decoded/opened before calling the analyzer.
    data = ["this is text, not file or filename"]
    with pytest.raises(err_type, match=err_msg):
        Estimator(analyzer=lambda x: x.split(), input=input_type).fit_transform(data)


@pytest.mark.parametrize(
    "Estimator",
    [
        CountVectorizer,
        TfidfVectorizer,
        pytest.param(HashingVectorizer),
    ],
)
@pytest.mark.parametrize(
    "analyzer", [lambda doc: open(doc, "r"), lambda doc: doc.read()]
)
@pytest.mark.parametrize("input_type", ["file", "filename"])
def test_callable_analyzer_change_behavior(Estimator, analyzer, input_type):
    # [FUNCTION DOC]
    # Checks callable analyzers that themselves open or read the document raise
    # FileNotFoundError or AttributeError with input "file"/"filename", for every
    # vectorizer.
    data = ["this is text, not file or filename"]
    with pytest.raises((FileNotFoundError, AttributeError)):
        Estimator(analyzer=analyzer, input=input_type).fit_transform(data)


@pytest.mark.parametrize(
    "Estimator", [CountVectorizer, TfidfVectorizer, HashingVectorizer]
)
def test_callable_analyzer_reraise_error(tmpdir, Estimator):
    # [FUNCTION DOC]
    # Checks an exception raised inside a custom analyzer propagates unchanged when
    # the input is a real file object (created in tmpdir).
    # check if a custom exception from the analyzer is shown to the user
    def analyzer(doc):
        # [FUNCTION DOC]
        # HELPER: Analyzer that always raises Exception("testing").
        raise Exception("testing")

    f = tmpdir.join("file.txt")
    f.write("sample content\n")

    with pytest.raises(Exception, match="testing"):
        Estimator(analyzer=analyzer, input="file").fit_transform([f])


@pytest.mark.parametrize(
    "Vectorizer", [CountVectorizer, HashingVectorizer, TfidfVectorizer]
)
@pytest.mark.parametrize(
    (
        "stop_words, tokenizer, preprocessor, ngram_range, token_pattern,"
        "analyzer, unused_name, ovrd_name, ovrd_msg"
    ),
    [
        (
            ["you've", "you'll"],
            None,
            None,
            (1, 1),
            None,
            "char",
            "'stop_words'",
            "'analyzer'",
            "!= 'word'",
        ),
        (
            None,
            lambda s: s.split(),
            None,
            (1, 1),
            None,
            "char",
            "'tokenizer'",
            "'analyzer'",
            "!= 'word'",
        ),
        (
            None,
            lambda s: s.split(),
            None,
            (1, 1),
            r"\w+",
            "word",
            "'token_pattern'",
            "'tokenizer'",
            "is not None",
        ),
        (
            None,
            None,
            lambda s: s.upper(),
            (1, 1),
            r"\w+",
            lambda s: s.upper(),
            "'preprocessor'",
            "'analyzer'",
            "is callable",
        ),
        (
            None,
            None,
            None,
            (1, 2),
            None,
            lambda s: s.upper(),
            "'ngram_range'",
            "'analyzer'",
            "is callable",
        ),
        (
            None,
            None,
            None,
            (1, 1),
            r"\w+",
            "char",
            "'token_pattern'",
            "'analyzer'",
            "!= 'word'",
        ),
    ],
)
def test_unused_parameters_warn(
    Vectorizer,
    stop_words,
    tokenizer,
    preprocessor,
    ngram_range,
    token_pattern,
    analyzer,
    unused_name,
    ovrd_name,
    ovrd_msg,
):
    # [FUNCTION DOC]
    # For each vectorizer and each combination where a parameter is overridden by
    # another (e.g. stop_words or tokenizer with a char analyzer, token_pattern with a
    # tokenizer, preprocessor or ngram_range with a callable analyzer), checks fit
    # emits the "The parameter ... will not be used since ..." UserWarning.
    train_data = JUNK_FOOD_DOCS
    # setting parameter and checking for corresponding warning messages
    vect = Vectorizer()
    vect.set_params(
        stop_words=stop_words,
        tokenizer=tokenizer,
        preprocessor=preprocessor,
        ngram_range=ngram_range,
        token_pattern=token_pattern,
        analyzer=analyzer,
    )
    msg = "The parameter %s will not be used since %s %s" % (
        unused_name,
        ovrd_name,
        ovrd_msg,
    )
    with pytest.warns(UserWarning, match=msg):
        vect.fit(train_data)


@pytest.mark.parametrize(
    "Vectorizer, X",
    (
        (HashingVectorizer, [{"foo": 1, "bar": 2}, {"foo": 3, "baz": 1}]),
        (CountVectorizer, JUNK_FOOD_DOCS),
    ),
)
def test_n_features_in(Vectorizer, X):
    # [FUNCTION DOC]
    # Checks CountVectorizer and HashingVectorizer have no n_features_in_ before or
    # after fit.
    # KNOWN ISSUE: HashingVectorizer is fitted on a list of dicts rather than text;
    # this only passes because its fit does not analyze the documents.
    # For vectorizers, n_features_in_ does not make sense
    vectorizer = Vectorizer()
    assert not hasattr(vectorizer, "n_features_in_")
    vectorizer.fit(X)
    assert not hasattr(vectorizer, "n_features_in_")


def test_tie_breaking_sample_order_invariance():
    # [FUNCTION DOC]
    # Non-regression (#17939): with max_features=1 and tied counts, the kept term
    # does not depend on document order.
    # Checks the sample order invariance when setting max_features
    # non-regression test for #17939
    vec = CountVectorizer(max_features=1)
    vocab1 = vec.fit(["hello", "world"]).vocabulary_
    vocab2 = vec.fit(["world", "hello"]).vocabulary_
    assert vocab1 == vocab2


def test_nonnegative_hashing_vectorizer_result_indices():
    # [FUNCTION DOC]
    # Non-regression (PR 19035): hashed indices are non-negative for an input whose
    # hash would be the minimum int32 value.
    # add test for pr 19035
    hashing = HashingVectorizer(n_features=1000000, ngram_range=(2, 3))
    indices = hashing.transform(["22pcs efuture"]).indices
    assert indices[0] >= 0


@pytest.mark.parametrize(
    "Estimator", [CountVectorizer, TfidfVectorizer, TfidfTransformer, HashingVectorizer]
)
def test_vectorizers_do_not_have_set_output(Estimator):
    # [FUNCTION DOC]
    # Checks the text vectorizers and TfidfTransformer do not expose set_output.
    """Check that vectorizers do not define set_output."""
    est = Estimator()
    assert not hasattr(est, "set_output")


@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_tfidf_transformer_copy(csr_container):
    # [FUNCTION DOC]
    # Checks TfidfTransformer.transform with copy=True leaves the input unchanged and
    # returns a new object, while copy=False modifies the input in place, returning
    # the same object (or, if the sparse API is realigned, a new object sharing
    # indptr/indices/data buffers).
    """Check the behaviour of TfidfTransformer.transform with the copy parameter."""
    X = _sparse_random_array((10, 20000), dtype=np.float64, random_state=42)
    X_csr = csr_container(X)

    # keep a copy of the original matrix for later comparison
    X_csr_original = X_csr.copy()

    transformer = TfidfTransformer().fit(X_csr)

    X_transform = transformer.transform(X_csr, copy=True)
    assert_allclose_dense_sparse(X_csr, X_csr_original)
    assert X_transform is not X_csr

    X_transform = transformer.transform(X_csr, copy=False)
    # allow for config["sparse_interface"] to change output type
    # there should be no data copied, but the `id` will change.
    if _align_api_if_sparse(X_csr) is X_csr:
        assert X_transform is X_csr
    else:
        assert X_transform is not X_csr
        assert X_transform.indptr is X_csr.indptr
        assert X_transform.indices.base is X_csr.indices.base
        assert X_transform.data.base is X_csr.data.base

    with pytest.raises(AssertionError):
        assert_allclose_dense_sparse(X_csr, X_csr_original)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_tfidf_vectorizer_preserve_dtype_idf(dtype):
    # [FUNCTION DOC]
    # Non-regression (#30016): idf_ dtype follows the vectorizer dtype (float32 or
    # float64) on a large corpus of random UUID strings.
    """Check that `idf_` has the same dtype as the input data.

    Non-regression test for:
    https://github.com/scikit-learn/scikit-learn/issues/30016
    """
    X = [str(uuid.uuid4()) for i in range(100_000)]
    vectorizer = TfidfVectorizer(dtype=dtype).fit(X)
    assert vectorizer.idf_.dtype == dtype


def test_hashing_vectorizer_requires_fit_tag():
    # [FUNCTION DOC]
    # Checks HashingVectorizer tags report requires_fit as False.
    """Test that HashingVectorizer has requires_fit=False tag."""
    vectorizer = HashingVectorizer()
    tags = vectorizer.__sklearn_tags__()
    assert not tags.requires_fit


def test_hashing_vectorizer_transform_without_fit():
    # [FUNCTION DOC]
    # Checks HashingVectorizer.transform works without fit and returns shape (2, 10).
    """Test that HashingVectorizer can transform without fitting."""
    vectorizer = HashingVectorizer(n_features=10)
    corpus = ["This is test", "Another test"]
    result = vectorizer.transform(corpus)
    assert result.shape == (2, 10)
