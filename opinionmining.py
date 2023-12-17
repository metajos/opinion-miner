import pandas as pd
import numpy as np
import spacy
import os
import re
from typing import List, Tuple
from spacy.language import Doc
from icecream import ic
from IPython.display import display
from nltk.stem import PorterStemmer, LancasterStemmer, SnowballStemmer
from collections import Counter


nlp = spacy.load("en_core_web_md")
basepath = os.getcwd()
raw_txt_dir = "data"
full_path = os.path.join(basepath, raw_txt_dir)
pd.set_option("display.max_columns", 50)
pd.set_option("display.max_rows", 50)
pd.set_option("display.width", 2000)
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
    _id = 1
    @classmethod
    def next_id(cls):
        next = cls._id
        cls._id += 1
        return next

    def __init__(self, product, string):
        self.id:int = Product.next_id()
        self.name: str = product.split(".")[0].split("/")[-1]
        self.reviews: List[Review] = self.parse_reviews(string)
        self.raw_string:str = string
        self.sentences: List[Sentence] = self.get_all_sentences()


    def __repr__(self) -> str:
        return f"{self.name.upper()}:R:{len(self.reviews)}:S:{len(self.get_all_sentences())}"

    def get_all_sentences(self):
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
                review_list.append(Review(self, review_string))
        else:
            for line in modified_stream.split("\n"):
                review_list.append(Review(self, line))
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
    _id = 1
    @classmethod
    def next_id(cls):
        next = cls._id
        cls._id += 1
        return next

    def __init__(self, product: Product, string: str):
        self.id:int = Review.next_id()
        self.product_id:int = product.id
        # Process each review into its subsequent sentences
        self.sentence_tuples:list[tuple] = self.split_stream(string)
        self.sentences: list[Sentence] = [Sentence(self.id, self.product_id, tup[0], tup[1]) for tup in self.sentence_tuples if tup is not None]

    def split_stream(self, stream:str):
        try:
            split_reviews = stream.split("\n")
            review_tuples = []
            for sent in split_reviews:
                sent = sent.strip()
                is_forbidden = re.search(r"(?:[*]+[*]$|^[*]+.*)", sent)
                if sent == "" or sent is None or is_forbidden is not None:
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
    _id = 1
    @classmethod
    def next_id(cls):
        next = cls._id
        cls._id += 1
        return next


    def __init__(self, review_id, product_id, ground_truth_score, string):
        self.sentence_id:int = Sentence.next_id()
        self.product_id: int = product_id
        self.review_id: int = review_id
        self.sentence: str = string
        self.gt_categories, self.gt_scores = self.extract_ground_scores_and_categories(ground_truth_score)
        self.series = pd.Series({"Product_ID":self.product_id,
                                    "Review_ID":self.review_id,
                                 "Sentence_ID":self.sentence_id,
                                    "Sentence":self.sentence,
                                    "gt_categories":self.gt_categories,
                                    "gt_score" :self.gt_scores})

    def extract_ground_scores_and_categories(self, string)-> List[Tuple[str, int]]:
        pattern = r"(?:[\w\s-]*?\[[+|-]\d\])"
        matches = re.findall(pattern, string)
        category_score_tuples = []
        if not matches:
            category_score_tuples.append(("no sentiment", 0))
        else:
            for match in matches:
                score = re.search(r"(?:[+-]\d)", match).group()
                text = re.search(r"(?:[\w\s-]*(?=\[))", match).group()
                category_score_tuples.append((text.lower(), int(score)))
        categories, scores = zip(*category_score_tuples)
        return categories, scores

    def __repr__(self):
        return self.sentence

    def __str__(self):
        return self.sentence





class ReviewDatabase:
    def __init__(self, paths):
        self.dataframe: pd.DataFrame = self.get_dataframe(self.process_product_paths(paths))

    def get_dataframe(self, products):
        dataframe = pd.DataFrame()
        for product in products:
            for sentence in product.sentences:
                temp_df = sentence.series.to_frame().T
                dataframe = pd.concat([dataframe, temp_df], ignore_index=True)
        return dataframe

    def process_product_paths(self, paths):
        products = []
        for file in paths:
            try:
                contents = parsefile(file)
                products.append(Product(file, contents))
            except Exception as e:
                print(e)
        return products

    def __repr__(self):
        return f"products: {[product.name for product in self.products]}"


class FeatureExtraction:

    @classmethod
    def categories(cls, string):
        """"""
        doc = nlp(string.lower())
        prohibited_ents = {"MONEY", "DATE", "TIME", "QUANTITY", "CARDINAL", "PERCENT"}
        named_entities = [ents for ents in doc.ents if str(ents.label_) not in prohibited_ents]
        named_entities_set = set(
            [token for word in named_entities for token in word if not token.is_punct and not token.is_digit])
        np = [" ".join([str(ent) for ent in doc.ents if str(ent.label_) not in prohibited_ents])]
        for chunk in doc.noun_chunks:
            chunk_ = []
            for i, token in enumerate(chunk):
                if token in named_entities_set or token.is_punct or token.is_digit or token.is_stop:
                    continue
                if i >= 1:
                    if str(chunk[i - 1]) == "no":
                        chunk_.append(f"NOT_{str(token)}")
                        continue
                chunk_.append(str(token))
            if len(chunk_) != 0:
                np.append("_".join(chunk_))
        result = [t for t in np if t != ""]
        return result


    @classmethod
    def stemming(cls, sentence:str, stemmer = None) -> str:
        doc = nlp(sentence)
        if stemmer is None:
            stemmer = SnowballStemmer("english")


        return " ".join([stemmer.stem(token.text) for token in doc])


    @classmethod
    def subtree(cls, sentence:str) -> pd.DataFrame:
        doc = nlp(sentence)
        df = pd.DataFrame()
        for t in doc:
            pos_series = pd.Series({"token": t,
                                    "pos": t.pos_,
                                    "lemma": t.lemma_,
                                    "dep": t.dep_,
                                    "children": [child for child in t.children]
                                    })
            df = pd.concat([df, pos_series.to_frame().T])
        return df








