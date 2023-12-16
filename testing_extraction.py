import unittest
from opinionmining import *

class MyTestCase(unittest.TestCase):
    #
    # def test_n_product_reviews(self):
    #     paths = ["data/Customer_review_data/Nokia 6610.txt"]
    #     products = []
    #     n = 3
    #     for folder in os.listdir(full_path):
    #         counter = 0
    #         try:
    #             for file in os.listdir(os.path.join(full_path, folder)):
    #                 if counter >= n:
    #                     continue
    #                 if str(file) != "Readme.txt" and str(file) != ".DS_Store":
    #                     try:
    #                         contents = parsefile(os.path.join(full_path, folder, file))
    #                         products.append(Product(file, contents))
    #                         counter += 1
    #                     except Exception as e:
    #                         print(e)
    #         except NotADirectoryError:
    #             pass
    #     print(products)

    def test_single_file(self):
        paths = ["data/Customer_review_data/Nokia 6610.txt"]

        db = ReviewDatabase(paths).dataframe.head()
        print(db)
        db["nouns"] =  list(map(lambda x : part_of_speech.noun_phrases(x), db["String"]))
        print(db["nouns"])



if __name__ == '__main__':
    unittest.main()
