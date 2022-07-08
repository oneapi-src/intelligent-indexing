# !/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=C0415,E0401,R0914

"""
Helper functions for preprocessing
"""

import re
from typing import List

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.util import ngrams

# Ensure that the required NLTK libraries are downloaded
nltk.download('punkt')
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()


def clean_headline(headline: str) -> str:
    """cleans the headline field

    Args:
        headline (str): headline associated with an article

    Returns:
        str: cleaned headline
    """
    headline = headline.lower()
    headline = re.sub(r"[^\w]", " ", headline)
    return headline


def clean_link(link: str) -> str:
    """cleans the hyperlink field

    Args:
        link (str): hyperlink associated with an article

    Returns:
        str: cleaned hyperlink
    """
    link = link.lower()
    link = link.replace("https://www.huffingtonpost.com/entry/", "")
    link = re.sub(
        r'https://www.huffingtonpost.com'
        r'https?://|www\d{0,3}[.][a-z0-9.\-]+[.]com/',
        "",
        link
    )
    link = re.sub(r"(\W|_)+", " ", link)

    link = re.sub(r"[^\w]", " ", link)
    link = link.replace("html", " ")
    link = re.sub(r"\b[a-zA-Z]\b", "", link).strip()
    return " ".join(link.strip().split()[:-1])


def clean_short_description(short_description: str) -> str:
    """cleans the short description field

    Args:
        short_description (str): short description associated with an article

    Returns:
        str: cleaned short description
    """
    short_description = short_description.lower()
    short_description = re.sub(r"[^\w]", " ", short_description)
    return short_description


def tokenize(text: str) -> List[str]:
    """turns a body of text into a collection of tokens

    Args:
        text (str): the body of text to tokenize

    Returns:
        List[str] : collection of tokens
    """
    tokens = word_tokenize(text)
    tokens = [stemmer.stem(tk) for tk in tokens]
    tokens = [tk for tk in tokens if tk not in stop_words]
    n_grams = ngrams(tokens, 2)
    tokens = tokens + [' '.join(grams) for grams in n_grams]
    return tokens
