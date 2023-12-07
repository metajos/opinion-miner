import pandas as pd
import numpy as np
import os
import re
from typing import List
from icecream import ic

basepath = os.getcwd()
raw_txt_dir = "data"
full_path = os.path.join(basepath, raw_txt_dir)


def parsefile(file):
    try:
        with open(file, mode="r") as f:
            contents = f.read()
            return contents
    except Exception as e:
        print(e)


def split_stream(stream):
    try:
        split_reviews = stream.split("\n")
        review_tuples = []
        for sent in split_reviews:
            sent = sent.strip()
            if sent == "":
                continue
            split_sent = sent.split("##")
            assert len(split_sent) == 2, f"This sentence review does not split into two: {sent}"
            split_sent[0] = split_sent[0].strip()
            if split_sent[0] == "":
                split_sent[0] = "no_sentiment[0]"
            review_tuples.append((split_sent[0], split_sent[1]))
        return review_tuples
    except AssertionError as e:
        print(e)

    # def remove_annotation(stream:str) -> str:



def remove_titles(stream: str) -> str:
    pattern = r"(\[t\].*)"
    return recurse_remove(stream=stream, pattern=pattern)


def recurse_remove(stream, pattern):
    match = re.search(pattern, stream)
    if match is None:
        return stream
    else:
        stream = stream[:match.start()] + stream[match.end():]
        return recurse_remove(stream, pattern)


# def preprocess_pipeline(filepath: str) -> list[str]:
#     try:
#         raw_text = parsefile(filepath)
#         clean_of_annotations_text = remove_annotation(stream=raw_text)
#         clean_of_review_titles_text = remove_titles(stream=clean_of_annotations_text)
#         tuples = split_stream(clean_of_review_titles_text)
#         return tuples
#     except AssertionError as e:
#         print(e)


def read_files():
    for folder in os.listdir(full_path):
        try:
            for file in os.listdir(os.path.join(full_path, folder)):
                if str(file) != "Readme.txt" and str(file) != ".DS_Store":
                    try:
                        preprocessed_tuples = preprocess_pipeline(os.path.join(full_path, folder, file))
                        return preprocessed_tuples
                    except Exception as e:
                        print(e)
        except NotADirectoryError:
            pass


class Product:
    def __init__(self, product, string):
        self.product: str = product
        self.reviews: list = self.parse_reviews(string)
        self.opinions: list = []
        self.raw_string = string

    def __str__(self) -> str:
        return f"{self.product}, revcount:{len(self.reviews)}"

    def __repr__(self) -> str:
        return f"{self.product}, #Reviews:{len(self.reviews)}, #Sentences:{len([sentence. for sentence in self.reviews])}"

    def parse_reviews(self, stream: str) -> list:
        review_list: list[Review] = []
        modified_stream: str = stream
        # if the review is delimited by [t] symbol, break it into subsets of reviews
        if re.search(pattern=r"(\[t\])", string=stream):
            matches = True
            while matches:
                match = self.match_review_tabs(modified_stream)
                if not match:
                    break
                modified_stream = modified_stream[:match.start()] + "\n" + modified_stream[match.end():]
                # remove the title from the review
                review_string = recurse_remove(match.group(), r"(\[t\].*)")
                review_list.append(Review(self.product, review_string))
        else:
            for line in modified_stream.split("\n"):
                review_list.append(Review(self.product, line))
        return review_list

    def match_review_tabs(self, stream: str) -> re.match or None:
        pattern = r"(\[t\](.*?)(?=\[t\]))"
        match = re.search(pattern, stream, re.DOTALL)
        if match:
            return match
        return None

    def remove_annotation(self, stream: str) -> str:
        pattern = r"\*(.*)\*"
        match = re.search(pattern, stream, re.DOTALL)
        if match:
            stream = stream[:match.start()] + stream[match.end():]
            return stream
        return stream

    def recurse_remove(self, stream: str, pattern: re.Pattern) -> str:
        match = re.search(pattern, stream)
        if match is None:
            return stream
        else:
            stream = stream[:match.start()] + stream[match.end():]
            return recurse_remove(stream, pattern)


class Review:
    def __init__(self, product: Product, string: str):
        self.id: str = None
        self.product: Product = product
        self.raw_review: str = string
        self.sentences: list[Sentence] = []
        self.is_single_ln = False

    def split_lines(self, string):
        lns = string.split("\n")
        if len(lns) <= 1:
            return [Sentence(review=self, product=self.product, string=string)]
        else:
            sentences = []
            for ln in lns:
                sentences.append(Sentence(review=self, product=self.product, string=string))

    def __str__(self) -> str:
        return f"review: {self.raw_review}"

    def __repr__(self) -> str:
        return f"review: {self.raw_review}"


class Sentence:
    def __init__(self, review, product, string):
        self.id: str = None
        self.product: Product = product
        self.review: Review = review
        self.raw_sentence: str = string
        self.sentence_tuples: list[tuple] = self.split_stream(string)
        self.user_category_scores: list = []
        self.extracted_categories: list = []

    def split_stream(self, stream):
        try:
            split_reviews = stream.split("\n")
            review_tuples = []
            for sent in split_reviews:
                sent = sent.strip()
                if sent == "":
                    continue
                split_sent = sent.split("##")
                assert len(split_sent) == 2, f"This sentence review does not split into two: {sent}"
                split_sent[0] = split_sent[0].strip()
                if split_sent[0] == "":
                    split_sent[0] = "no_sentiment[0]"
                review_tuples.append((split_sent[0], split_sent[1]))
            return review_tuples
        except AssertionError as e:
            print(e)


def TextCrawler():
    paths = ["data/CustomerReviews-3_domains/Speaker.txt"]
    products = []
    for folder in os.listdir(full_path):
        try:
            for file in os.listdir(os.path.join(full_path, folder)):
                if str(file) != "Readme.txt" and str(file) != ".DS_Store":
                    try:
                        contents = parsefile(os.path.join(full_path, folder, file))
                        products.append(Product(file, contents))
                    except Exception as e:
                        print(e)
        except NotADirectoryError:
            pass

    ic(products)


    # for path in paths:
    #     with open(path, "r", encoding="utf-8", newline="\n") as f:
    #         stream = f.read()
    #     product_name = path.split("/")[-1]
    #     product = Product(product_name, stream)
    #     print(len(product.reviews))






