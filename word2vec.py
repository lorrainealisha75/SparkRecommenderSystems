"""
Spark Job for the cluster.
This script leverages Spark capability of word embededdings
using word2vec model to build recommender systems.
"""
from pyspark import SparkContext
from pyspark import SparkConf
from pyspark.sql import SQLContext, Window
from pyspark.sql import functions as sf
from pyspark.sql.types import StructType, StructField
from pyspark.sql.types import StringType, IntegerType, ArrayType

from pyspark.ml.feature import RegexTokenizer
from pyspark.ml.feature import StopWordsRemover
from pyspark.ml.feature import Word2Vec

from sklearn.cross_validation import train_test_split

from nltk.stem import SnowballStemmer

import numpy as np
import math

#Getting the sparkContext and sqlContext objects for further use

conf = SparkConf().set('spark.kryoserializer.buffer.max.mb', '1g')
sc = SparkContext(conf=conf)
sql = SQLContext(sc)

#Construct a schema for the DF
schema = StructType([StructField('paper_id', IntegerType(), True),
                     StructField('type', StringType(), True),
                     StructField('journal', StringType(), True),
                     StructField('book_title', StringType(), True),
                     StructField('series', StringType(), True),
                     StructField('publisher', StringType(), True),
                     StructField('pages', IntegerType(), True),
                     StructField('volume', StringType(), True),
                     StructField('number', StringType(), True),
                     StructField('year', StringType(), True),
                     StructField('month', StringType(), True),
                     StructField('postedat', StringType(), True),
                     StructField('address', StringType(), True),
                     StructField('title', StringType(), True),
                     StructField('abstract', StringType(), True)])

#Load data
papers_df = sql.read.csv("papers.csv", schema = schema, header = None)

#Selecting paper_id, paper title and abstract columns respectively 
papers_df = papers_df.select(papers_df[0], papers_df[13], papers_df[14])
#papers_df.show(truncate=False)

#Concatenate the title and abstract column
title_abstract_concat = papers_df.withColumn\
('title_abstract', sf.concat_ws(' ', papers_df.title, papers_df.abstract))

title_abstract_concat = title_abstract_concat.select("paper_id", "title_abstract")

#title_abstract_concat.show(truncate=False)

#Tokenize the content in column 'title_abstract' on all non word characters, excluding '-' and '_'.
#Allow only tokens with length greater than 3
regexTokenizer = RegexTokenizer(minTokenLength=3, inputCol="title_abstract", outputCol="words",\
                                pattern="[^A-Za-z_-]" )

#Apply the regex tokenizer
papers_tokenized_df = regexTokenizer.transform(title_abstract_concat).select("paper_id", "words")
#papers_tokenized_df.show(truncate=False)

#Convert DF into RDD
papers_tokenized_rdd = papers_tokenized_df.rdd.map(tuple)

#Function to remove '_' and '-' from the words
def removeHypenUnderscore(x):
    result = []
    for w in x[1]:
        if '-' in w:
            w = w.replace('-', '')
        if '_' in w:
            w = w.replace('_', '')
        result.append(w)  
    return (x[0], result)

papers_tokenized_rdd = papers_tokenized_rdd.map(lambda x: removeHypenUnderscore(x))

#Create a schema which has columns ('paper_id', 'words')
schema = StructType([StructField('paper_id', StringType(), True),
                     StructField('words', ArrayType(StringType()), True)])

#Create a DF with columns (user_hash_id, list(paper_id))
papers_cp_df = sql.createDataFrame(papers_tokenized_rdd, schema)
#papers_cp_df.show(truncate=False)

#Get Word2Vec instance with vector size 100
word2Vec = Word2Vec(vectorSize=100, numPartitions=10, inputCol="words", outputCol="word_vector_cp")

#Fit model on Conservative pre-processing DF
cp_model = word2Vec.fit(papers_cp_df)

#Get vector for each word
word_vector_cp_df = cp_model.getVectors()
#word_vector_cp_df.show(truncate=False)

#Get top 10 words which are most similar to the word 'science'
cp_synonyms_df = cp_model.findSynonyms("science", 10)
#cp_synonyms_df.show(truncate=False)

#Load all the stopwords and convert it to a list
RDD = sc.textFile('stopwords_en.txt')
stop_rdd = RDD.map(lambda x: x)
stopWordList = stop_rdd.collect()

#Apply StopWordsRemover
remover = StopWordsRemover(inputCol="words", outputCol="minWords", stopWords=stopWordList)
papers_minwords_df = remover.transform(papers_cp_df).select('paper_id', 'minWords')
#papers_minwords_df.show(truncate=False)

#Convert DF into RDD
papers_minwords_rdd = papers_minwords_df.rdd.map(tuple)

#Function to stem the words using SnowballStemmer
def stemmingWords(x):
    stemmer = SnowballStemmer('english')
    # Set contains only unique entries
    stemmedSet = set()
    for w in x[1]:
        stemmedSet.add(stemmer.stem(w))
    return (x[0], list(stemmedSet))

stem_words_rdd = papers_minwords_rdd.map(lambda x: stemmingWords(x))

#Create a schema which has columns ('paper_id', 'words')
schema = StructType([StructField('paper_id', StringType(), True),
                     StructField('words', ArrayType(StringType()), True)])

#Create a DF with columns (user_hash_id, list(paper_id))
papers_ip_df = sql.createDataFrame(stem_words_rdd, schema)
#papers_ip_df.show(truncate=False)

#Get Word2Vec instance with vector size 100
word2Vec = Word2Vec(vectorSize=100, numPartitions=10, inputCol="words", outputCol="word_vector_ip")

#Fit model on Intensive pre-processing DF
ip_model = word2Vec.fit(papers_ip_df)

#Get vector for each word
word_vector_ip_df = ip_model.getVectors()
#word_vector_ip_df.show(truncate=False)

#Get top 10 words which are most similar to the word 'science'
#ip_synonyms_df = ip_model.findSynonyms("science", 10)
#ip_synonyms_df.show(truncate=False)

#Get top 10 words which are most similar to the word 'scienc'
ip_synonyms_df = ip_model.findSynonyms("scienc", 10)
#ip_synonyms_df.show(truncate=False)

def analogy(word1, word2, word3, word2vec_model):
    
    #Get the (word, vector) DF
    word_vector_df = word2vec_model.getVectors()
    
    word_list = []
    def getVector(df, word):
        #Spilt on " " if there are multiple words
        words = word.split(" ")
        
        #Add all words into a list for further use
        for w in words:
            word_list.append(w)
        
        #If it is a single word, return the corresponding vector
        if len(words) == 1:
            return df.filter(word_vector_df['word'] == word).select('vector').collect()[0][0]
        else:
            vector = 0
            for word in words:
                #If there are multiple words, return the sum of all vectors of individual words
                vector = vector + (df.filter(word_vector_df['word'] == word).select('vector').collect()[0][0])
            return vector
    
    #Get the vector of the corresponding word(s)
    word1_vec = getVector(word_vector_df, word1)
    word2_vec = getVector(word_vector_df, word2)
    word3_vec = getVector(word_vector_df, word3)
    
    #Get the resultant vector
    result_vec = (word1_vec - word2_vec) + word3_vec
    
    #Get top 5 synonyms for the resultant vector
    w = word2vec_model.findSynonyms(result_vec, 5).select('word').rdd.map(tuple).flatMap(lambda x:x).collect()
    for i in range(0, len(word_list)):
        if word_list[i] in w:
            #Remove word if it is contained in the intial arguments
            w.remove(word_list[i])
    #Return first word in the list        
    return w[0]

w_cp = analogy("machine learning", "predictions", "recommender systems", cp_model)
print("With conservative model: Machine Learning is to predictions as Recommender Systems is to " + w_cp)

w_ip = analogy("machin learn", "predict", "recommend systems", ip_model)
print("With intensive model: Machine Learning is to predictions as Recommender Systems is to " + w_ip)

paper_profile_df = cp_model.transform(papers_cp_df)
paper_profile_df = paper_profile_df.select("paper_id", "word_vector_cp")\
                    .selectExpr("paper_id as paper_id", "word_vector_cp as embedding_profile")
#paper_profile_df.show(truncate=False)

#Load user library file
RDD = sc.textFile('users_libraries.txt')

users_rdd = RDD.map(lambda x: (x.split(";")[0], list(x.split(";")[1].split(","))))
users_rdd = users_rdd.map(lambda x:[(x[0], i) for i in x[1]]).flatMap(lambda x:x)

#Create a schema which has columns ('user_hash_id', 'paper_id') of string type
schema = StructType([StructField('user_id', StringType(), True),
                     StructField('paper_id', StringType(), True)])

#Create a DF with columns (user_id, paper_id)
user_df = sql.createDataFrame(users_rdd, schema)

#Get a mapping between user_id and the collection of words belonging to each paper that he has liked
user_temp_df = user_df.join(papers_cp_df, user_df['paper_id'] == papers_cp_df['paper_id'], 'inner')

def merge_lists(row):
    e = []
    
    for i in row[1]:
       #Merge the list of words from each paper into a single list 
       e = e + i
    
    return (row[0], e)
        
user_words_rdd = user_temp_df.groupBy('user_id').agg(sf.collect_list('words')).rdd.map(tuple)\
                .map(lambda x: merge_lists(x))

#Create a schema which has columns ('user_hash_id', 'paper_id') of string type
schema = StructType([StructField('user_id', StringType(), True),
                     StructField('words', ArrayType(StringType()), True)])

user_words_df = sql.createDataFrame(user_words_rdd, schema)
#user_words_df.show(truncate=False)

word2Vec = Word2Vec(vectorSize=100, numPartitions=10, inputCol="words", outputCol="user_profile")
w2v_model = word2Vec.fit(user_words_df)

user_profile_df = w2v_model.transform(user_words_df)
#user_profile_df.show(truncate=False)

def dataSampler(user_df, user_profile_df, n):
    
    #Select n random users
    sample_df = user_df.select('user_hash_id', 'paper_ids').orderBy(sf.rand()).limit(n)

    def splitAsTrainAndTest(x):
        user = x[0]
        paper_ids = x[1]
        train, test = train_test_split(paper_ids, train_size=0.8, test_size=0.2)
        return (user, train, test)
    
    #Split each users library into a train and test set of 0.8 and 0.2 partition respectively
    train_test_df = sample_df.rdd.map(lambda x: splitAsTrainAndTest(x))\
                        .toDF(['user_id','train_set_paper_ids','test_set_paper_ids'])
    
    #Join the DF obtained from the previous step with the train_test_df to get a new DF of the form
    #(user_id, user_profile, training_set, test_set)
    user_profile = user_profile_df.join(train_test_df, user_profile_df['user_id'] == train_test_df['user_id']).\
                    drop(train_test_df['user_id']).drop(user_profile_df['words'])
    
    return user_profile

#Function to calculate recommendations based to conservative user profile
def cbrs(x):
    
    def computeSimilarity(userProfileVector, itemProfileVector):
        #Multiply the user and item profile and sum up individual elements
        prodProfiles = np.sum(np.multiply(userProfileVector, itemProfileVector))

        #Multiply the square roots of the summed elements of the user and item profiles raised to power of 2 
        prodSqrt = np.multiply(np.sqrt(np.sum(np.power(userProfileVector, 2))),\
                               np.sqrt(np.sum(np.power(itemProfileVector, 2))))
        result = float(prodProfiles/prodSqrt)
        if not np.isnan(result):
            return result

    if x[4] not in x[2]:
        return (x[0], x[3], x[4], computeSimilarity(x[1], x[5]))

users_rdd = RDD.map(lambda x: (x.split(";")[0], list(x.split(";")[1].split(","))))

#Create a schema which has columns ('user_hash_id', 'paper_id') of string type
schema = StructType([StructField('user_hash_id', StringType(), True),
                     StructField('paper_ids', ArrayType(StringType()), True)])

#Create a DF with columns (user_id, paper_ids)
user_df = sql.createDataFrame(users_rdd, schema)

n = 1
k = 10

#Get profile of user 1eac022a97d683eace8815545ce3153f
user_sampled_profile_df = dataSampler(user_df.filter(sf.col('user_hash_id') == '1eac022a97d683eace8815545ce3153f'),\
                            user_profile_df.filter(sf.col('user_id') == '1eac022a97d683eace8815545ce3153f'), n)

#Get a mapping between all user and item profiles
user_item_profiles_df = user_sampled_profile_df.crossJoin(paper_profile_df)

#Get similarity score for each paper recommended for the user
user_item_similarity_scores_rdd = user_item_profiles_df.rdd.map(lambda x: cbrs(x)).filter(lambda x: x is not None)
user_item_similarity_scores_df = user_item_similarity_scores_rdd\
                            .toDF(["user_id", "test_set", "paper_id", "similarity_score"])

#Partion by user_id and sort by descending similarity score
window = Window.partitionBy("user_id").orderBy(sf.desc("similarity_score"))

#Get the top k recommendations for a user by doing a dense rank
paper_recomm_df = user_item_similarity_scores_df\
                .withColumn("rank", sf.dense_rank().over(window).alias('rank')).filter(sf.col('rank') <= k) 

#Collect all paper_ids for a user in a list
paper_recomm_top_k = paper_recomm_df.select("user_id", "test_set", "paper_id")
paper_recomm_top_k = paper_recomm_top_k.groupBy("user_id", "test_set")\
                                .agg(sf.collect_list("paper_id").alias("recommendations"))

print("Top 10 recommendations for user 1eac022a97d683eace8815545ce3153f are: ")
#paper_recomm_top_k.show(truncate=False)

n = 50
k = 35

user_sampled_profile_df = dataSampler(user_df, user_profile_df, n)

#Get a mapping between all user and item profiles
user_item_profiles_df = user_sampled_profile_df.crossJoin(paper_profile_df)

#Get similarity score for each paper recommended for the user
user_item_similarity_scores_rdd = user_item_profiles_df.rdd.map(lambda x: cbrs(x)).filter(lambda x: x is not None)
user_item_similarity_scores_df = user_item_similarity_scores_rdd.toDF(["user_id", "test_set", "paper_id", "similarity_score"])

#Partion by user_id and sort by descending similarity score
window = Window.partitionBy("user_id").orderBy(sf.desc("similarity_score"))

#Get the top k recommendations for a user by doing a dense rank
paper_recomm_df = user_item_similarity_scores_df.withColumn("rank", sf.dense_rank().over(window).alias('rank')).filter(sf.col('rank') <= k) 

#Collect all paper_ids for a user in a list
paper_recomm_top_k = paper_recomm_df.select("user_id", "test_set", "paper_id")

#The recommendation DF is of the form (user_id, test_set, recommendation_list)
paper_recomm_top_k = paper_recomm_top_k.groupBy("user_id", "test_set").agg(sf.collect_list("paper_id").alias("recommendations"))

print("Top k recommendations for all users: ")
#paper_recomm_top_k.show(truncate=False)

def precision_u_at_k(user_input_df, k):
    user_input_rdd = user_input_df.rdd.map(tuple)
    
    def calculate(user_id, test_set, top_k_recomm, k):
        #Get only first k items in the list
        list1 = top_k_recomm[:k]
        hits = 0
        for i in range(0, k):
            #If element matches any element in the test_set, we have a hit
            if list1[i] in test_set:
                hits = hits + 1     
        return (user_id, test_set, top_k_recomm, float(hits/k))
    
    user_score_df = user_input_rdd.map(lambda x: calculate(x[0], x[1], x[2], k)).toDF(["user_hash_id", "test_set", "top_k_recomm", "precision_at_k"])
    return user_score_df

def recall_u_at_k(user_input_df, k):
    user_input_rdd = user_input_df.rdd.map(tuple)
    
    def calculate(user_id, test_set, top_k_recomm, k):
        #Get only first k items in the list
        list1 = top_k_recomm[:k]
        hits = 0
        for i in range(0, k):
            #If element matches any element in the test_set, we have a hit
            if list1[i] in test_set:
                hits = hits + 1
        return (user_id, test_set, top_k_recomm, float(hits/len(test_set)))
    
    user_score_df = user_input_rdd.map(lambda x: calculate(x[0], x[1], x[2], k)).toDF(["user_hash_id", "test_set", "top_k_recomm", "recall_at_k"])
    return user_score_df

def mrr_u_at_k(user_input_df, k):
    user_input_rdd = user_input_df.rdd.map(tuple)
    
    def calculate(user_id, test_set, top_k_recomm, k):
        list1 = top_k_recomm[:k]
        for i in range(0, k):
            if list1[i] in test_set:
                break
        if i == k:
            #If there were no hits, mrr is 0
            return (user_id, test_set, top_k_recomm, float(0))
        else:
            return (user_id, test_set, top_k_recomm, float(1/(i+1)))
    
    user_score_df = user_input_rdd.map(lambda x: calculate(x[0], x[1], x[2], k)).toDF(["user_hash_id", "test_set", "top_k_recomm", "mrr_at_k"])
    return user_score_df

def ndcg_u_at_k(user_input_df, k):
    user_input_rdd = user_input_df.rdd.map(tuple)
    
    def calculate(user_id, test_set, top_k_recomm, k):
        list1 = top_k_recomm[:k]
        dcg = 0
        idcg = 1.0
        hits = 0

        for i in range(0, k):
            if list1[i] in test_set:
                hits = hits + 1
                if k == 0:
                    #If the first position is a hit, set dcg to the relevance, which is 1
                    dcg = 1
                else :    
                    dcg = dcg + (1.0 / math.log((k + 1.0), 2))
        
        for i in range(0, hits):
            #idcg (ideal dcg) is when all the hits where at the beginning of the recommendation list
            idcg = idcg + (1.0 / math.log((hits + 1), 2))
         
        return (user_id, test_set, top_k_recomm, dcg/idcg)
    
    user_score_df = user_input_rdd.map(lambda x: calculate(x[0], x[1], x[2], k)).toDF(["user_hash_id", "test_set", "top_k_recomm", "ndcg_at_k"])
    return user_score_df

def evaluate_metrics(recommendation_df):

    k = [5, 10, 30]

    print("CBRS with Word Embeddings")
    for i in k:
        precision_df = precision_u_at_k(recommendation_df, i)
        #precision_df.show(truncate=False)

        recall_df = recall_u_at_k(recommendation_df, i)
        #recall_df.show(truncate=False)

        mrr_df = mrr_u_at_k(recommendation_df, i)
        #mrr_df.show(truncate=False)

        ndcg_df =  ndcg_u_at_k(recommendation_df, i)
        #ndcg_df.show(truncate=False)

        #Calculate average evaluation metrics for k = 5, 10, 30
        avg_precision_df = precision_df.agg(sf.avg(sf.col("precision_at_k")))  
        avg_recall_df = recall_df.agg(sf.avg(sf.col("recall_at_k")))
        avg_mrr_df = mrr_df.agg(sf.avg(sf.col("mrr_at_k")))
        avg_ndcg_df = ndcg_df.agg(sf.avg(sf.col("ndcg_at_k")))

        print("When k = " + str(i) + " Average precision is " + str(avg_precision_df.collect()[0][0]))
        print("When k = " + str(i) + " Average recall is " + str(avg_recall_df.collect()[0][0]))
        print("When k = " + str(i) + " Average mrr is " + str(avg_mrr_df.collect()[0][0]))
        print("When k = " + str(i) + " Average ndcg is " + str(avg_ndcg_df.collect()[0][0]))

#Evaluate metrics for all user recommendations
evaluate_metrics(paper_recomm_top_k)

n = 50
k = 35

#Get only those users who have moret than 20 papers in their library
user_df = user_df.filter(sf.size(sf.col('paper_ids')) > 20)

user_profile_above_20_df = user_df.join(user_profile_df, user_df['user_hash_id'] == user_profile_df['user_id'])\
                            .drop(user_df['user_hash_id']).drop(user_df['paper_ids'])

user_sampled_profile_above_20_df = dataSampler(user_df, user_profile_above_20_df, n)

#Get a mapping between all user and item profiles
user_item_profiles_df = user_sampled_profile_above_20_df.crossJoin(paper_profile_df)

#Get similarity score for each paper recommended for the user
user_item_similarity_scores_rdd = user_item_profiles_df.rdd.map(lambda x: cbrs(x)).filter(lambda x: x is not None)
user_item_similarity_scores_df = user_item_similarity_scores_rdd.toDF(["user_id", "test_set", "paper_id", "similarity_score"])

#Partion by user_id and sort by descending similarity score
window = Window.partitionBy("user_id").orderBy(sf.desc("similarity_score"))

#Get the top k recommendations for a user by doing a dense rank
paper_recomm_df = user_item_similarity_scores_df.withColumn("rank", sf.dense_rank().over(window).alias('rank')).filter(sf.col('rank') <= k) 

#Collect all paper_ids for a user in a list
paper_recomm_top_k = paper_recomm_df.select("user_id", "test_set", "paper_id")

#The recommendation DF is of the form (user_id, test_set, recommendation_list)
paper_recomm_above_20_top_k = paper_recomm_top_k.groupBy("user_id", "test_set").agg(sf.collect_list("paper_id").alias("recommendations"))

#Evaluate metrics for user recommendations with more than 20 papers in the library
evaluate_metrics(paper_recomm_above_20_top_k)