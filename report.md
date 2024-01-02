
# Opinon Miner  
  
## Step 1: Task and Data Analyses  
Typically, sentiment analysis involves determining whether any given document contains positive or negative sentiment. Determining the direction of sentiment can be useful in many applications where it might be valuable to understand the performance  of some offering, in the broadest of respects. Whether for a company, product, service, automatic reporting of sentiment allows a quick way to gauge the how the device performs and saves time for the reader. However, going further than reporting pure positive or negative sentiment, it might sometimes be more valuable to understand exactly which components of a given report, review or document might lead to an overall feeling of sentiment, and therefore a more in-depth analysis to the core strengths and weaknesses of a given offering is required. 

Undertaking such a task is called opinion mining and this is the basis of this project. The task of this project is to perform opinion mining on a set of product reviews given in the data folder which are taken from Amazon.com. The project will aim to replicate the opinion mining procedure done on the same set of reviews prepared by (Hu and Liu, 2004) whose sentiment classification was performed using lexicographical approach and compare the performance to a machine learning approach using a suitable algorithm. Finally, the aim is to mine frequent features from each product so that they can be reported with a summary of their corresponding sentiment that is either positive or negative. For each feature, we report the $n_{features}$  most commonly occurring feature  with  individual totals of positive and negative reviews, along with some $n_{reviews}$ , user-determined number of example reviews of that each positive and negative class. 

Through reviewing the product reviews, its clear that a comprehensive pipeline is required to transform the unstructured data in the review sets into structured opinion sets reporting mined features. For each product there is a set of reviews that have split into sentences with corresponding positive and negative review scores, if the user provided a score. With this in mind, an algorithm that uses regular expressions to match all the sentences and their scores is required and the process of simply extracting each sentence and its review score to create a database is an important step in initialising the data preprocessing. This will be discussed more in the next section. The quantity of the number of reviews varies between a few thousand and a few hundred. This is an important note as it steers which types of machine learning models are suitable to this task. Doing so will require a few steps including data preprocessing which first involves separating each set of reviews into sentences, where there are inconsistencies between the review sets, tokenisation, part-of-speech tagging, fuzzy-matching, and stemming. feature extraction for most commonly occurring features and the pruning irrelevant features, sentiment analysis using both the lexical approach outlined in (Hu and Liu, 2004) and a machine learning approach, performance evaluation and finally, summarisation and reporting. The outcome of the performance evaluation phrase will determine the final algorithm with which the mined sentiments are reported.



## Step 2: Data Preprocessing  

The relation between each step in the methodology is shown in *diagram 1*. There are two major steps to this process - the first is segmenting each sentence and its ground truth score from the inconsistent data sets. To do so, a number of regular expressions and error handling techniques are used to handle different cases. For example, removing title annotations, review tags delimited by `[t]`, extracting each sentence limited by `##` and then extracting the ground truth scores like `[-1]`. Regular expressions make use of deterministic finite automata to search a corpus of text to match strings in a given vocabulary (Jurafsky and Martin, 2023, p.5). It is necessary to extract these review scores as they provide a ground truth with which performance can be evaluated. Sentence segmentation is appropriate since the task is to provide feedback on frequently occurring features and typically commenting on features happens at the sentence level as opposed to the paragraph level.  For example "*I use Norton 2003 at work with absolutely no trouble.*" and "*I've heard Norton 2004 version is fine too*" are part of the same review paragraph but discuss two features independently of each other. The combinations of products, sentences and their respective scores is the foundational database which can facilitate further data preprocessing. 

If the aim is to eventually perform sentiment analysis using a machine learning model then one of the targets for data preprocessing is to first normalise the text. A text that is normalised is a text that is in a compact form by means of reducing the size and complexity of the text. Doing so is advantageous because it reduces the dimensionality of the vocabulary that describes "bag-of-words" vectors. This is also useful in frequent feature extraction because constraining the search space for semantically similar words will group them together and allow more efficient extraction of features. Working backwards, the aim is to process the data into a form that is as semantically compact as possible by incorporating stemming. Stemming is the process of reducing different forms of the same word (and sometimes other words) to a more simple versions of itself. By doing so, stemming can make vocabularies of bag-of-word type vectors more compact and reduce vector sparsity which is desirable in  machine learning models like Naive Bayes and Logistic Regression. (Wang and Manning, 2012). 
### Tokenisation
To perform stemming it is necessary to first break up sentences into their constituent words by applying tokenisation. The process of splitting tokens from sentences is closely related to named-entity-recognition algorithms. This is because words cannot simply be split on whitespaces, punctuation and clitics because these are common features in meaningful compound words and objects like website addresses and place names. (Jurafsky and Martin, 2008, p.5). In order to improve the accuracy of the meaning of negations, we transform words with preceding negations into "not_word". This allows us to reverse the polarity in the lexical classifier. Following tokenisation, part of speech tagging is performed before stemming to preserve the semantic meaning for the POS tagger.
### Part of Speech Tagging
To identify and extract the noun phrases part-of-speech tagging is used from the `spacy` library. There are two types of part of speech tagging: rule based part for speech tagging and Hidden Markov Model tagging that use statistical models. 

#### HMM tagging
Since we are using `spacy`'s POS tagger, it is more likely tagging is done using a Hidden Markov Model (HMM). HMM taggers use bayesian inference to perform sequence classification tasks that allow for determining the parts of speech tags given a sequence of words. The equation below taken from (Jurafsky and Martin, 2008, p.140) describes the essence of the mathematics behind part of speech tagging with Hidden Markov models.
$$
\hat t_{1:n} = \text{argmax}_{t_{1:n}}P(t_{1}...t_{n}| w_{1}...w_{n})  	
$$
The intuition behind this equation is to choose the  tag sequence from all the possible combinations of tags $t_{1:n}$, word sequence $w_{1:n}$ of a fixed length $n$ the set of tags which maximise the probability observing a set of tags, given some sequence of words. What it is saying is that the word sequence has been observed, the tag sequence that maximises the posterior probability of what has been observed is most likely to be the correct tag sequence, and should therefore be picked. This equation can be expressed using the HMM emission and transition matrices, approximately, by the following equation from (Kochmar,  2023):
$$
\approx \underset{t_{1} \ldots t_{n}}{\operatorname{argmax}} \prod_{i=1}^{n} \overbrace{P\left(w_{i} \mid t_{i}\right)}^{\text
{emission }} \overbrace{P\left(t_{i} \mid t_{i-1}\right)}^{\text {transition }}$$
Here, the emission matrix is the set of observable features (the words) and the hidden features (tags) are the transition matrix. The equation is asking to find the combination of hidden states (emission) that maximises the observable states (transition). In other words, find the set of tags that maximise the chances of observing this sequence of words. Since this equation has time complexity of $O(n^{t})$, the *Viterbi* algorithm is commonly used to improve this efficiency. Using dynamic programming, the *Viterbi* algorithm prunes states that cannot yield maximum probabilities thereby significantly reducing the number of combinations needed to check for a correct tag sequence. 

### Minimum Edit Distance
After part of speech tagging, perform fuzzy matching on the nouns. Split the list into two halves where the bottom half is matched with the top half. Using Zipf's law to guide this heuristic, it is unlikely that a word is misspelled frequently enough for it to appear in the top half of the word distribution. Use `nltk`'s minimum edit distance. The minimum edit distance measures the number of insertions, deletions, and substitutions required to change one string into another (Jurafsky and Martin, 2008, p.75). A minimum edit distance of 2 is used for spelling fixes. Following fuzzy matching, the nouns and sentences are stemmed. `nltk`'s  `SnowballStemmer` is used, an improved modification on the `PorterStemmer`. 
### Stemming
The Porter stemmer (Porter, 1980) is a stemming algorithm that transforms words into base forms of consonants and verbs by removes suffixes through a 5-step series of rewrite rules.  The first step performs rules on a string that rewrites plurality, past participles, the following four steps removes and adds features to the string following morphological structures. *Diagram 2* shows how the algorithm stems an arbitrarily word "predicated" to "pred". 

![[porter_stemmer.jpg]]
*Diagram 2: Porter stemmer performing stemming on "predicate". Words at bottom of each step move to top of following step. Stemmer rules denoted in blue.*

Finally, after stemming is complete, a dictionary is created to map each stemmed word into an expanded word. The expanded word will be used later in the process when it is required to re-present the nouns as labelled features in the summary document. The dictionary maps each stem to the most frequent word that is reduced to that stem. 

## Step 3: Product Feature Extraction  
The product feature extraction pipeline closely follows the algorithms described in (Hu and Liu, 2004) with a few modifications. The purpose is to improve the set of extracted nouns by pairing nouns, removing redundant features and finding adjectives. 
### Frequent Feature Selection
To perform frequent selection an association rule mining technique using the Apriori algorithm is used from the `mlxtend` library. The purpose is to identify the noun tuples that appear often enough to be useful. Let $N$ be a set of nouns $n_1, n_2 ...$ where $max(|N|)=3$. The Apriori algorithm happens in two steps calculating support and confidence. The following equations are sourced from (Han, Pei, and Tong, 2022, p.246)  For nouns $n_{1}, n_{2}$:
$$
\text{Support}(n_{1}) = \frac{\text{number of sets containing } n_{1}}{\text{total number of sets}}
$$
For support, a minimum threshold of 1% is used. The confidence of each set measures the strength of association for the elements in set $N = \set{n_{1}, n_{2}}$ , or mathematically:
$$
\text{Confidence}(n_{1}\rightarrow n_{2}) = \frac{\text{Support}(n_{1}\cup n_{{2}})}{\text{Support}(n_{1})}
$$
In other words, use support to extract and prune features where the frequency of combinations of nouns that appear above a certain threshold. Use the confidence to prune those combinations that are not strongly associated above threshold - 60% is used. This creates sets like `{{"internet","norton","security"}, {"internet", "security"}, {"norton"}}`. This is useful in extracting compound word features. Since the words are unordered sets, check every permutation of the set, find most frequent ordering that can be used in reporting stage. This operation has time complexity $O(N!)$ but we have already constrained $max(|N|)=3$ so this is acceptable. 
### Feature Pruning 
There are two steps to removing redundant features. Firstly,  where $|N|>1$, let $d_{i} =|i - j|$ where $i,j$ denotes word index positions. Then prune the  nouns at $i$ where $d_{i} > 2$. This filters out noun phrases that are not in close proximity to each other. The second stage of feature pruning is perform *psupport* pruning, described in (Hu and Liu, 2004, p.3). The objective of psupport pruning is to remove singular nouns that do no to count towards frequent compound features. For a sets where $|N| = 1$ and noun $n\in N$. 
For every sentence $s$, for every feature set $m$ count the number of times $(n \in s ) \land (n \notin m)$ if count is less than 3, cull $n$ from the feature set. By doing so, we remove those singular noun features that are commonly associated with compound features, but do not occur alone. 

### Opinion Word Extraction
The lexical sentiment classifier will use the polarity of the opinion words of a sentence to determine the sentiment.  For every sentence, extract the adjectives and adverbs. The adverbs are included because the performance of the product could be commented on as a verb, consider "the router performs terribly." By only extracting the adjectives, like (Hu and Liu, 2004)  the sample sentence bears no sentiment which is clearly false. Extract the noun adjective pairings where the adjective is a modifier of the feature as the effective opinion. This opinion word will a greater weighting in the final determination of sentiment as it directly modifies the feature. 

## Step 4: Sentiment Analysis  
Sentiment analysis is performed using two methodologies the first is a lexical approach that uses SentiwordNet's synsets to determine polarity and classify sentences. This approach is suitable because it is an unsupervised approach that does not require a pre-labelled dataset, and leveraging the inherent sentiment of the words in the sentence to determine sentiment. The second approach uses multinomial Naive Bayes (MNB) algorithm.  MNB is chosen both because of the relatively small size of the data set and that it is shown to have the best performance in where the vectors are snippets from sentences as opposed to full length reviews (Wang and Manning, 2012). 

### Lexical Classifier

1. For each sentence $S$, we have two sets of words: _opinion words_ $w_o$ and _effective opinion words_ $w_e$ and overall classification sentiment $y \in \{-1,1\}$.
2. create two counts where $s_o=$*opinion word score*, $s_e=$  *effective opinion score*.
3. For each word in $w_o$, $w_e$, increase $s_o$, $s_e$ based on the polarity of the synset from `SentiWordNet`, inverting the score of negations and antonyms.
5. Determine  $y$:
    - If $s_o + s_e=0$ , calculate $y$ using $S$ as argument in `textblob.sentiment.polarity`
	    - $y = 0 \rightarrow y = 1$ (this is where potential positive bias is introduced) 
	    - $y < 0 \rightarrow y = -1$
    - $((|s_e| > |s_o|) \land (s_e > 1)) \rightarrow y = 1$  
    - $((|s_e| > |s_o|) \land (s_e < 1)) \rightarrow y = -1$  
    - $(s_o + s_e > 1) \rightarrow y=1$ 
    - $(s_o + s_e < 1) \rightarrow y=1$ 
This algorithm closely follows (Hu and Liu, 2004)  but introduces potential positive bias since `SentiWordNet` and `textblob` polarity scores occasionally fail to find polarity. In such cases, try the antonym score. Otherwise if still neutral, return positive. In this case, we decide to return positive since the review is not negative. Similar failures in detecting neutral sentiment will occur in the MNB methodology since it is a binary classification problem. 

### Naive Bayes

The MNB classifier predicts which class a sentence belongs to by calculating the relative frequencies of each word belonging to a class and use bayesian inference to determine the most likely class for each sentence, based on observed probabilities in the training data. The posterior probabilities are computed using below formula (Chirney, 2023) where $n_{c, w}$ = count for each word in class $c$ of the training data, $k = |vocabulary|$  with added laplacian smoothing such that $\theta_{c,w} \neq 0$.  
$$
\begin{equation}
{\theta_{c, w}} = \frac{n_{c, w} + \alpha}{n_{c} + k \alpha}
\end{equation}$$
Which can be substituted into a simplified form of the MNB prediction function that uses logarithm to prevent underflow (Chirney, 2023):
$$
\hat c= \underset{c \in \{-1,1\}}{\operatorname{argmax}} \ \left[ \log( {p(C=c)}) + \sum_{i = 1}^k w_i \ \log \left({\theta_{c, w_i}} \right) \right]

$$

- Since we use a bag of words vector, $w_i \in \{0, 1\}$, \
- $p(C=c)=\frac{\text{count of class } c}{\text{count of all combined classes}}$, $c \in \{1,-1\}$

To prevent overfitting we perform Leave One Out Cross Validation (LOOXVE) to maximise the number of training  examples. For a dataset of $n$ training samples, LOOXVE uses of $n-1$ training split, with a test size of 1, finalising the model on the average error across all training samples.

## Step 5: Evaluation and discussion

## Bibliography




