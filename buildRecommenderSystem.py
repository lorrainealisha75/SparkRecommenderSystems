"""
Spark Job for the cluster.
This script the performance of two kinds of recommender systems are compared:
LDA (Latent Direchlet Allocation) and TF-IDF (Term Frequency-Inverse Document Frequency)
"""
from pyspark import SparkContext
from pyspark.sql import SQLContext, Window
from pyspark.sql import functions as sf
from pyspark.sql.functions import udf, col, desc, dense_rank, collect_list
from pyspark.sql.types import StructType, StructField
from pyspark.sql.types import StringType, IntegerType, FloatType, ArrayType

from pyspark.ml.feature import RegexTokenizer, CountVectorizer
from pyspark.ml.feature import StopWordsRemover, IDF

from pyspark.ml.linalg import DenseVector, VectorUDT, _convert_to_vector
from pyspark.ml.clustering import LDA

from sklearn.cross_validation import train_test_split
from nltk.stem import SnowballStemmer

import scipy.sparse
import numpy as np
import csv

#Getting the sparkContext and sqlContext objects for further use

sc = SparkContext()
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
#papers_df.rdd.saveAsTextFile("part1.txt")

#Get the total number of unique papers
num_of_papers = papers_df.distinct().count()

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

#Load all the stopwords and convert it to a list
RDD = sc.textFile('stopwords_en.txt')
stop_rdd = RDD.map(lambda x: x)
stopWordList = stop_rdd.collect()

#Apply StopWordsRemover
remover = StopWordsRemover(inputCol="words", outputCol="minWords", stopWords=stopWordList)
papers_minwords_df = remover.transform(papers_tokenized_df).select('paper_id', 'minWords')
#papers_minwords_df.show(truncate=False)

#Convert DF into RDD
papers_minwords_rdd = papers_minwords_df.rdd.map(tuple)

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

papers_minwords_rdd = papers_minwords_rdd.map(lambda x: removeHypenUnderscore(x))

#Function to stem the words using SnowballStemmer
def stemmingWords(x):
    stemmer = SnowballStemmer('english')
    # Set contains only unique entries
    stemmedSet = set()
    for w in x[1]:
        stemmedSet.add(stemmer.stem(w))
    return (x[0], list(stemmedSet))

stem_words_rdd = papers_minwords_rdd.map(lambda x: stemmingWords(x))

#Create an RDD of the form (word, paper_id)
word_paper_map = stem_words_rdd.map(lambda x:[(i, x[0]) for i in x[1]]).flatMap(lambda x:x)

#Groupby word so that we get a list of papers that contains the word
word_paperlist_map = word_paper_map.groupByKey().mapValues(list)

#Filter out words that are contained in more than 10% of the papers and then less than 20 papers
word_filtered_rdd = word_paperlist_map.map(lambda x: (x[0], x[1], len(x[1]), len(x[1])/num_of_papers)).filter(lambda x: x[3]<=0.1).filter(lambda x: x[2] >= 20).map(lambda x: (x[0], x[2]))

#set T
word_filtered_sorted = word_filtered_rdd.sortBy(lambda x: x[1], False).top(1000, key=lambda x: x[1])
T = sc.parallelize(word_filtered_sorted).map(lambda x: x[0]).collect()

def stemmingWords(x):
    stemmer = SnowballStemmer('english')  
    stemmedList = []
    for w in x[1]:
        stemmedList.append(stemmer.stem(w))
    return (x[0], stemmedList)

#This rdd of the form (paper_is, list(words)) contains duplicate words too, which will be used to build
#the final DF
stem_dup_words_rdd = papers_minwords_rdd.map(lambda x: stemmingWords(x))
paper_word_map = stem_dup_words_rdd.map(lambda x: [(x[0], i) for i in x[1]]).flatMap(lambda x:x)

#Keep only those tuples that contain words that appear in set T
paper_word_map = paper_word_map.filter(lambda x: x[1] in T)

#This RDD will only have words that appear in T
paper_wordlist_map = paper_word_map.groupByKey().mapValues(list)

#Convert RDD to DF
schema = StructType([StructField('paper_id', StringType(), True),
                     StructField('word_list', ArrayType(StringType()), True)])

paper_words_df = sql.createDataFrame(paper_wordlist_map, schema)
#paper_words_df.show(truncate=False)

#Use count vectorizer to build a term frequency sparse vector
cv_instance = CountVectorizer(inputCol="word_list", outputCol="features")

cv_model = cv_instance.fit(paper_words_df)

tf_vector = cv_model.transform(paper_words_df)

tf_vector = tf_vector.select(tf_vector['paper_id'], tf_vector['features'])
tf_vector.write.save("tf_vector.txt")


#Readfile from HDFS as DF

#Apply PySpark IDF estimator to get tf-idf scores
idf_instance = IDF(inputCol='features', outputCol='tf_idf_vector')
idf_model = idf_instance.fit(tf_vector)
tf_idf_vector = idf_model.transform(tf_vector).select("paper_id", "tf_idf_vector") 
tf_idf_vector.write.save("tf_idf_vector.txt")
tf_idf_vector.rdd.map(tuple).saveAsTextFile("tf_idf_vector_text.txt")


#Readfile from HDFS as DF
#tf_vector = sql.read.parquet("tf_vector.txt")

#Apply LDA algorithm with no:of latent topics(k) =40 and max iterations=10
lda_instance = LDA(k=40, maxIter=10)
lda_model = lda_instance.fit(tf_vector)

paper_topics_df = lda_model.transform(tf_vector)
paper_topics_df = paper_topics_df.select(paper_topics_df['paper_id'], paper_topics_df['topicDistribution'])

paper_topics_df.write.save("lda_vector.txt")
paper_topics_df.rdd.map(tuple).saveAsTextFile("lda_vector_text.txt")


#User Profile as a summation of TF-IDF vectors
def sum_TF_IDF_Vectors(tf_idf_vector, user_df):

    #Join user_df and tf_idf_vector to get a list of tf_idf_vectors for all the papers in the user's library
    user_sparsevec_df = user_df.join(tf_idf_vector, user_df['paper_id'] == tf_idf_vector['paper_id']).\
                            select(user_df['user_id'], tf_idf_vector['tf_idf_vector'])
    
    #Sum up all the tf_idf_vectors for a user
    user_sparsevec_rdd = user_sparsevec_df.rdd.mapValues(lambda v: v.toArray()).reduceByKey(lambda x, y: x+y) .mapValues(lambda x: DenseVector(x))
    user_profile_df = user_sparsevec_rdd.toDF(["user_id", "tf_idf_vector"])

    #Convert from dense to sparse vector
    def dense_to_sparse(vector):
        return _convert_to_vector(scipy.sparse.csc_matrix(vector.toArray()).T)

    convert_to_sparse = udf(dense_to_sparse, VectorUDT())
    user_profile_df = user_profile_df.withColumn("tf_idf_vector_sparse", convert_to_sparse(user_profile_df['tf_idf_vector']))
    user_profile_df = user_profile_df.select(user_profile_df['user_id'], user_profile_df['tf_idf_vector_sparse'])
    
    return user_profile_df

#User Profile as a summation of paper-topic distribution vectors
def sumPaperTopicDistribution(lda_vector, user_df):

    paper_topic_df = lda_vector.select(lda_vector['paper_id'], lda_vector['topicDistribution'])

    #Join user_df and paper_topic_df to get a list of paper-topic distribution vectors for 
    #all the papers in the user libaray
    user_topic_df = user_df.join(paper_topic_df, user_df['paper_id'] == paper_topic_df['paper_id']).\
                            select(user_df['user_id'], paper_topic_df['topicDistribution'])

    #Sum up all the paper-topic distribution vectors for a user
    user_topic_sparsevec_rdd = user_topic_df.rdd.mapValues(lambda v: v.toArray())\
                                    .reduceByKey(lambda x, y: x + y).mapValues(lambda x: DenseVector(x))

    user_topic_sparsevec_df = user_topic_sparsevec_rdd.toDF(['user_id', 'topic_distribution'])
    
    return user_topic_sparsevec_df


def dataSampler(df, n):
    
    tf_vector = sql.read.parquet("tf_vector.txt")
    tf_idf_vector = sql.read.parquet("tf_idf_vector.txt")
    lda_vector = sql.read.parquet("lda_vector.txt")

    #Select n random users
    sample_df = df.select('user_hash_id', 'paper_ids').orderBy(sf.rand()).limit(n)
    
    def splitAsTrainAndTest(x):
        user = x[0]
        paper_ids = x[1]
        train, test = train_test_split(paper_ids, train_size=0.8, test_size=0.2)
        return (user, train, test)
    
    #Split each users library into a train and test set of 0.8 and 0.2 partition respectively
    train_test_df = sample_df.rdd.map(lambda x: splitAsTrainAndTest(x)).toDF(['user_id','train_set','test_set'])
    
    #Training set
    train_df = train_test_df.selectExpr("user_id as user_id", "train_set as paper_ids").\
                select('user_id', 'paper_ids')
    
    #Explode paper_ids list in train_set to get DF of form (user_id, paper_id)
    training_df = train_df.select(train_df['user_id'], sf.explode(train_df['paper_ids']).alias('paper_id'))
    
    #Test set
    test_df = train_test_df.selectExpr("user_id as user_id", "test_set as paper_ids").\
                select('user_id', 'paper_ids')
    
    user_tf_idf_df = sum_TF_IDF_Vectors(tf_idf_vector, training_df)
    paper_topic_dist_df = sumPaperTopicDistribution(lda_vector, training_df)
    
    return user_tf_idf_df, paper_topic_dist_df, train_df, test_df


def computeSimilarity(userProfileVector, itemProfileVector):
    print(itemProfileVector, userProfileVector)
    prodProfiles = np.sum(np.multiply(userProfileVector, itemProfileVector))
    prodSqrt = np.multiply(np.sqrt(np.sum(np.power(userProfileVector, 2))),\
                           np.sqrt(np.sum(np.power(itemProfileVector, 2))))
    return float(prodProfiles/prodSqrt)


#Content-based recommendations

def cbrs_tf_idf(ItemProfileInfo):

    def intermediate(paper_id, item_prof, user_prof):
        return computeSimilarity(user_prof, item_prof)
    
    listr = []
    for row in user_tf_idf_broadcast.value:
        if ItemProfileInfo[0] in row[2]:
            continue
        else:
            listr.append((row[0], row[3], ItemProfileInfo[0], intermediate(ItemProfileInfo[0], ItemProfileInfo[1], row[1])))
    return listr

def cbrs_lda(ItemProfileInfo):
    
    def intermediate(paper_id, item_prof, user_prof):
        return computeSimilarity(user_prof, item_prof)
    
    listr = []
    for row in user_lda_broadcast.value:
        if ItemProfileInfo[0] in row[2]:
            continue
        else:
            listr.append((row[0], row[3], ItemProfileInfo[0], intermediate(ItemProfileInfo[0], ItemProfileInfo[1], row[1])))
    return listr

#Load user library file
RDD = sc.textFile('users_libraries.txt')

users_rdd = RDD.map(lambda x: (x.split(";")[0], list(x.split(";")[1].split(","))))

#Create a schema which has columns ('user_hash_id', 'user_library') of string type
schema = StructType([StructField('user_hash_id', StringType(), True),
                     StructField('paper_ids', ArrayType(StringType()), True)])

#Create a DF with columns (user_hash_id, list(paper_id))
user_papers_df = sql.createDataFrame(users_rdd, schema)

tf_idf_vector = sql.read.parquet("tf_idf_vector.txt")
tf_idf_vector_rdd = tf_idf_vector.rdd.map(tuple)

lda_vector = sql.read.parquet("lda_vector.txt")
lda_vector_rdd = lda_vector.rdd.map(tuple)

#Set number of users to be sampled
n = 20
user_tf_idf_df, user_lda_df, train_df, test_df = dataSampler(user_papers_df, n) #.filter(user_papers_df["user_hash_id"] == "1eac022a97d683eace8815545ce3153f"), n)


k = 40
user_tf_idf_df1 = user_tf_idf_df.join(train_df, user_tf_idf_df["user_id"] == train_df["user_id"])\
                  .select(user_tf_idf_df["user_id"], user_tf_idf_df["tf_idf_vector_sparse"], train_df["paper_ids"])\
                  .selectExpr("user_id as user_id", "tf_idf_vector_sparse as tf_idf_profile", "paper_ids as train_set_paper_ids")

user_tf_idf = user_tf_idf_df1.join(test_df, user_tf_idf_df1["user_id"] == test_df["user_id"])\
                 .select(user_tf_idf_df1["user_id"], user_tf_idf_df1["tf_idf_profile"], user_tf_idf_df1["train_set_paper_ids"], test_df["paper_ids"])\
                 .selectExpr("user_id as user_id", "tf_idf_profile as tf_idf_profile", "train_set_paper_ids as train_set_paper_ids", "paper_ids as test_set_paper_ids")

user_tf_idf_rdd = user_tf_idf.rdd.map(tuple)
user_tf_idf_broadcast = sc.broadcast(user_tf_idf_rdd.collect())

tf_idf_recomm_rdd = tf_idf_vector_rdd.map(lambda x: cbrs_tf_idf(x)).flatMap(lambda x:x)
tf_idf_recomm_df = tf_idf_recomm_rdd.toDF(["user_id", "test_set", "paper_id", "similarity_score"])

window = Window.partitionBy("user_id").orderBy(desc("similarity_score"))
tf_idf_recomm_df = tf_idf_recomm_df.withColumn("rank", dense_rank().over(window).alias('rank')).filter(col('rank') <= k) 
tf_idf_recomm_top_k = tf_idf_recomm_df.select("user_id", "test_set", "paper_id")
tf_idf_recomm_top_k = tf_idf_recomm_top_k.groupBy("user_id", "test_set").agg(collect_list("paper_id").alias("recommendations"))

tf_idf_recomm_top_k.rdd.map(tuple).saveAsTextFile("tf_idf_recomm.txt")

user_lda_df1 = user_lda_df.join(train_df, user_lda_df["user_id"] == train_df["user_id"])\
                  .select(user_lda_df["user_id"], user_lda_df["topic_distribution"], train_df["paper_ids"])\
                  .selectExpr("user_id as user_id", "topic_distribution as lda_profile", "paper_ids as train_set_paper_ids")

user_lda = user_lda_df1.join(test_df, user_lda_df1["user_id"] == test_df["user_id"])\
                 .select(user_lda_df1["user_id"], user_lda_df1["lda_profile"], user_lda_df1["train_set_paper_ids"], test_df["paper_ids"])\
                 .selectExpr("user_id as user_id", "lda_profile as lda_profile", "train_set_paper_ids as train_set_paper_ids", "paper_ids as test_set_paper_ids")

user_lda_rdd = user_lda.rdd.map(tuple)
user_lda_broadcast = sc.broadcast(user_lda_rdd.collect())

lda_recomm_rdd = lda_vector_rdd.map(lambda x: cbrs_lda(x)).flatMap(lambda x:x)
lda_recomm_df = lda_recomm_rdd.toDF(["user_id", "test_set", "paper_id", "similarity_score"])

window = Window.partitionBy("user_id").orderBy(desc("similarity_score"))
lda_recomm_df = lda_recomm_df.withColumn("rank", dense_rank().over(window).alias('rank')).filter(col('rank') <= k) 
lda_recomm_top_k = lda_recomm_df.select("user_id", "test_set", "paper_id")
lda_recomm_top_k = lda_recomm_top_k.groupBy("user_id", "test_set").agg(collect_list("paper_id").alias("recommendations"))

#lda_recomm_top_k.write.save("lda_recomm.txt")
lda_recomm_top_k.rdd.map(tuple).saveAsTextFile("lda_recomm_text.txt")


def precision_u_at_k(user_input_df, k):
    user_input_rdd = user_input_df.rdd.map(tuple)
    
    def calculate(user_id, test_set, top_k_recomm, k):
        list1 = top_k_recomm[:k]
        hits = 0
        for i in range(0, k):
            if list1[i] in test_set:
                hits = hits + 1
        print(hits)        
        return (user_id, test_set, top_k_recomm, float(hits/k))
    
    user_score_df = user_input_rdd.map(lambda x: calculate(x[0], x[1], x[2], k)).toDF(["user_hash_id", "test_set", "top_k_recomm", "precision_at_k"])
    return user_score_df

def recall_u_at_k(user_input_df, k):
    user_input_rdd = user_input_df.rdd.map(tuple)
    
    def calculate(user_id, test_set, top_k_recomm, k):
        list1 = top_k_recomm[:k]
        hits = 0
        for i in range(0, k):
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
            return (user_id, test_set, top_k_recomm, float(0))
        else:
            return (user_id, test_set, top_k_recomm, float(1/(i+1)))
    
    user_score_df = user_input_rdd.map(lambda x: calculate(x[0], x[1], x[2], k)).toDF(["user_hash_id", "test_set", "top_k_recomm", "mrr_at_k"])
    return user_score_df


k = [5, 10, 30]

print("CBRS_TF_IDF")
for i in k:
    precision_df = precision_u_at_k(tf_idf_recomm_top_k, k)
    #precision_df.write.save("precision_tf_idf.txt")
    precision_df.rdd.map(tuple).saveAsTextFile("precision_tf_idf_text.txt")

    recall_df = recall_u_at_k(tf_idf_recomm_top_k, k)
    #recall_df.write.save("recall_tf_idf.txt")
    recall_df.rdd.map(tuple).saveAsTextFile("recall_tf_idf_text.txt")

    mrr_df = mrr_u_at_k(tf_idf_recomm_top_k, k)
    #mrr_df.write.save("mrr_tf_idf.txt")
    mrr_df.rdd.map(tuple).saveAsTextFile("mrr_tf_idf_text.txt")

    avg_precision_df = precision_df.agg(sf.avg(sf.col("precision_at_k")))  
    avg_recall_df = recall_df.agg(sf.avg(sf.col("recall_at_k")))
    avg_mrr_df = mrr_df.agg(sf.avg(sf.col("mrr_at_k")))
    
    
    print("When k = " + str(i) + " Average precision is " + str(avg_precision_df.collect()[0][0]))
    print("When k = " + str(i) + " Average recall is " + str(avg_recall_df.collect()[0][0]))
    print("When k = " + str(i) + " Average mrr is " + str(avg_mrr_df.collect()[0][0]))
    

    avg_precision_df.rdd.map(tuple).saveAsTextFile("avg_tf_idf_precision_at_k.txt")
    avg_recall_df.rdd.map(tuple).saveAsTextFile("avg_tf_idf_recall_at_k.txt")
    avg_mrr_df.rdd.map(tuple).saveAsTextFile("avg_tf_idf_mrr_at_k.txt")

print("CBRS_LDA")
for i in k:
    precision_df = precision_u_at_k(lda_recomm_top_k, i)
    #precision_df.write.save("precision_lda.txt")
    precision_df.rdd.map(tuple).saveAsTextFile("precision_lda_text.txt")

    recall_df = recall_u_at_k(lda_recomm_top_k, i)
    #recall_df.write.save("recall_lda.txt")
    recall_df.rdd.map(tuple).saveAsTextFile("recall_lda_text.txt")

    mrr_df = mrr_u_at_k(lda_recomm_top_k, i)
    #mrr_df.write.save("mrr_lda.txt")
    mrr_df.rdd.map(tuple).saveAsTextFile("mrr_lda_text.txt")

    avg_precision_df = precision_df.agg(sf.avg(sf.col("precision_at_k")))  
    avg_recall_df = recall_df.agg(sf.avg(sf.col("recall_at_k")))
    avg_mrr_df = mrr_df.agg(sf.avg(sf.col("mrr_at_k")))
    
    
    print("When k = " + str(i) + " Average precision is " + str(avg_precision_df.collect()[0][0]))
    print("When k = " + str(i) + " Average recall is " + str(avg_recall_df.collect()[0][0]))
    print("When k = " + str(i) + " Average mrr is " + str(avg_mrr_df.collect()[0][0]))
    

    avg_precision_df.rdd.map(tuple).saveAsTextFile("avg_lda_precision_at_k.txt")
    avg_recall_df.rdd.map(tuple).saveAsTextFile("avg_lda_recall_at_k.txt")
    avg_mrr_df.rdd.map(tuple).saveAsTextFile("avg_lda_mrr_at_k.txt")
