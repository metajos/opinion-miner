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
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from nltk.metrics import distance
from matplotlib import pyplot as plt
import seaborn as sns


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
        self.name: str = product.split(".")[0].split("/")[-1].lower()
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
        self.sentences: list[Sentence] = [Sentence(self.id, self.product_id, product.name,tup[0], tup[1]) for tup in self.sentence_tuples if tup is not None]

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


    def __init__(self, review_id, product_id,product_name, ground_truth_score, string):
        self.sentence_id:int = Sentence.next_id()
        self.product_name = product_name
        self.product_id: int = product_id
        self.review_id: int = review_id
        self.sentence: str = string
        self.gt_categories, self.gt_scores = self.extract_ground_scores_and_categories(ground_truth_score)
        self.series = pd.Series({"Product_name":self.product_name,
                                    "Product_ID":self.product_id,
                                    "Review_ID":self.review_id,
                                 "Sentence_ID":self.sentence_id,
                                    "Sentence":self.sentence,
                                    "gt_categories":self.gt_categories,
                                    "gt_score" :sum(self.gt_scores)})

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
        result = FeatureExtraction.extract_nouns(string)
        if result is None:
            raise ValueError("Category extraction can't return None")
        return result

    @classmethod
    def extract_nouns(cls, string):
        doc = nlp(string.lower())
        nouns = []
        for token in doc:
            if token.pos_ in ["NOUN", "PROPN"] and not FeatureExtraction.is_illegal(token):
                nouns.append(token.text)
        if list(set(nouns)) is None:
            raise ValueError(f"Noun list cannot return None: {doc}")
       
        return list(set(nouns))


    @classmethod
    def is_illegal(cls, token):
        return any([token.is_punct,
                    token.is_digit,
                    ])

    @classmethod
    def add_negations(cls, sentence):
        doc = [token for token in nlp(sentence)]
        negation_tokens = ['not', 'no', 'never', "n't"]
        new_sentence = []

        for i, token in enumerate(doc):
            word = token.text
            if i > 0 and doc[i-1].text.lower() in negation_tokens:
                word = f"NOT_{word}"
            new_sentence.append(word)

        return " ".join(new_sentence)
        
    @classmethod
    def stemming(cls, string_list:List[str], stemmer = None) -> List[str]:
        if stemmer is None:
            stemmer = SnowballStemmer("english")
        stemmed_strings = []
        for string in string_list:
            string.replace("-", "")
            doc = nlp(string)
            stemmed_strings.append(" ".join([stemmer.stem(token.text) for token in doc if not FeatureExtraction.is_illegal(token)]))
        return [string for string in stemmed_strings if string != " "]


    @classmethod
    def remove_stop(cls, string_list:List[str]) -> List[str]:
        stemmed_strings = []
        for string in string_list:
            doc = nlp(string)
            stemmed_strings.append(" ".join([token.text for token in doc if not token.is_stop]))
        return stemmed_strings

    @classmethod
    def subtree(cls, sentence:str) -> pd.DataFrame:
        doc = nlp(sentence)
        df = pd.DataFrame()
        for t in doc:
            pos_series = pd.Series({"token": t,
                                    "pos": t.pos_,
                                    "head": t.head,
                                    "dep": t.dep_,
                                    "children": [child for child in t.children]
                                    })
            df = pd.concat([df, pos_series.to_frame().T])
        return df

    # search through the original words to retrieve the keywords associated to the stem in a new
    @classmethod
    def fuzzy_match_categories(cls, test_categories:List, target_categories:List):
        matched_categories = []
        for word in test_categories:
            match = FeatureExtraction.fuzzymatch(target_categories, word)
            if match:
                matched_categories.append(match)
            else:
                matched_categories.append(word)
        return matched_categories

    @classmethod
    def fuzzymatch(cls, target_words, test_word):
        for target_word in target_words:
            if distance.edit_distance(test_word, target_word) <= 2:
                return target_word
        return None
    
    @classmethod
    def clean_sentence(cls, sentence:str)-> List[str]:
        doc = nlp(sentence)
        strings = " ".join([token.text.lower() for token in doc if not FeatureExtraction.is_illegal(token) and not token.is_stop])
        return strings
    

    
    @classmethod
    def deconstruct_sentence(cls, stemmed_sentence):
        sentence_str = stemmed_sentence.split(" ")
        sentence_set = set(sentence_str)
        return sentence_set, sentence_str
        

def conf_matrix(validations, predictions, cfm):
    # This function plots a confusion matrix
    percentages = ["{0:.2%}".format(value) for value in cfm.flatten() / np.sum(cfm)]
    labels = np.array([f"True Neg:{cfm[0][0]} \n {percentages[0]}", f"False Pos:{cfm[0][1]} \n  {percentages[1]}",
                       f"False Neg: {cfm[1][0]} \n {percentages[2]}",
                       f"True Pos: {cfm[1][1]} \n {percentages[3]}"]).reshape(2, 2)
    confusion_matrix = pd.crosstab(validations, predictions, rownames=['Actual'], colnames=['Predicted'])
    return sns.heatmap(confusion_matrix, annot=labels, fmt="", cmap=sns.cubehelix_palette(start=1.5, rot=0.4, dark=.5, light=.75,reverse=False, as_cmap=True))



def accuracy_table(validations, predictions, model_name):
    # This function returns a dataframe with the accuracy, precision, recall and f1 score of a model
    accuracy = accuracy_score(validations, predictions)
    precision = precision_score(validations, predictions)
    return pd.DataFrame({"Model": model_name, "Accuracy": accuracy, "Precision": precision},
                        index=[0])

