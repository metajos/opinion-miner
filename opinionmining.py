import pandas as pd
import numpy as np
import os
import re
from typing import List, Tuple
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


class Product:
    def __init__(self, product, string):
        self.product: str = product
        self.reviews: list = self.parse_reviews(string)
        self.opinions: list = []
        self.raw_string = string

    def __str__(self) -> str:
        return f"{self.product}, revcount:{len(self.reviews)}"

    def __repr__(self) -> str:
        return f"{self.product}, #Reviews:{len(self.reviews)}, #Sentences:{len(self.get_sentences())}"

    def get_sentences(self):
        sentences = []
        for review in self.reviews:
            sentences_ = review.sentences
            for sent in sentences_:
                sentences.append(sent)
        return sentences

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
        self.id: str = str(hash(string)).split(" ")[-1]
        self.product_id: str = str(hash(product)).split(" ")[-1]
        # Process each review into its subsequent sentences
        self.sentence_tuples:list[tuple] = self.split_stream(string)
        self.sentences: list[Sentence] = [Sentence(self.id, self.product_id, score, sent) for (score, sent) in self.sentence_tuples]

    def split_stream(self, stream:str):
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

    def __str__(self) -> str:
        return f"review: {self.raw_review}"

    def __repr__(self) -> str:
        return f"review: {self.raw_review}"


class Sentence:
    def __init__(self, review_id, product_id, ground_truth_score, string):
        self.id: str = None
        self.product_id: Product = product_id
        self.review_id: Review = review_id
        self.sentence: str = string
        self.ground_category_score_tuples = self.extract_ground_scores_and_categories(ground_truth_score)
        self.user_category_scores: list = []
        self.extracted_categories: list = []

    def extract_ground_scores_and_categories(self, string)-> List[Tuple[str, int]]:
        pattern = r"(?:[\w\s-]*?\[[+|-]\d\])"
        matches = re.findall(pattern, string)
        category_score_tuples = []
        if not matches:
            category_score_tuples.append(("no sentiment", 0))
        else:
            for match in matches:
                score = re.search("(?:[+-]\d)", match).group()
                text = re.search(r"(?:[\w\s-]*(?=\[))", match).group()
                category_score_tuples.append((text.lower(), int(score)))
        ic(category_score_tuples)
        return category_score_tuples



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



    # for path in paths:
    #     with open(path, "r", encoding="utf-8", newline="\n") as f:
    #         stream = f.read()
    #     product_name = path.split("/")[-1]
    #     product = Product(product_name, stream)
    #     print(len(product.reviews))






