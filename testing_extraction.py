import unittest
from opinionmining import *

class MyTestCase(unittest.TestCase):

    def test_category_extractions(self):
        pwd = os.getcwd()
        filename = "db.csv"
        db = pd.read_csv(filename)
        
        database = db.where(db["Product_ID"] == 2).copy().dropna()

        
    

        # #Create a copy of the database to perform feature extraction
        # category_table = database.loc[:,"Product_ID":"Sentence"].copy()
        # # Perform Stemming on the sentences
        # category_table["Stemmed_Sentence"] = database.Sentence.apply(lambda x: FeatureExtraction.stemming([x])[0])
        # # Remove the stop words
        # category_table["Clean_Sentence"] = category_table["Stemmed_Sentence"].apply(lambda x: FeatureExtraction.remove_stop([x])[0])
        #
        # flattened_nouns = [item for sublist in database.ExtractedCategories for item in sublist]
        # frequency_sorted_nouns = [item for item, count in Counter(flattened_nouns).most_common()]
        # midpoint = len(frequency_sorted_nouns)//2
        # firsthalf = frequency_sorted_nouns[:midpoint]
        # secondhalf = frequency_sorted_nouns[midpoint:]


if __name__ == '__main__':
    unittest.main()
