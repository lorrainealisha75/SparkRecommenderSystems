"""
This script leverages the ML capabilities of Spark
by posing the recommendation problem as a classification
problem and uses SVM and Random Forest to solve it.
"""
from pyspark import SparkContext
from pyspark import SparkConf
from pyspark.sql import SQLContext
from pyspark.sql import functions as sf
from pyspark.sql.types import StructType, StructField
from pyspark.sql.types import StringType, IntegerType, FloatType

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import Imputer
from pyspark.ml.feature import OneHotEncoderEstimator

from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.classification import LinearSVC
from pyspark.ml.evaluation import RegressionEvaluator

import matplotlib.pyplot as plt
import numpy as np
import operator
import random

#Getting the sparkContext and sqlContext objects for further use

conf = SparkConf().set('spark.kryoserializer.buffer.max.mb', '1g')
sc = SparkContext(conf=conf)
sql = SQLContext(sc)

#Load users data
usersRDD = sc.textFile('users_libraries.txt')

#Construct a schema for the DF
schema = StructType([StructField('paper_idx', IntegerType(), True),
                     StructField('type', StringType(), True),
                     StructField('journal', StringType(), True),
                     StructField('book_title', StringType(), True),
                     StructField('series', StringType(), True),
                     StructField('publisher', StringType(), True),
                     StructField('pages', FloatType(), True),
                     StructField('volume', StringType(), True),
                     StructField('number', StringType(), True),
                     StructField('year', FloatType(), True),
                     StructField('month', StringType(), True),
                     StructField('postedat', StringType(), True),
                     StructField('address', StringType(), True),
                     StructField('title', StringType(), True),
                     StructField('abstract', StringType(), True)])

#Load papers data
papers_df = sql.read.csv("papers.csv", schema = schema, header = None)

#Create a map between user_hash_id and user_id (a unique positive integer)
user_id_map = usersRDD.map(lambda x: (x.split(";")[0])).zipWithUniqueId().collectAsMap()

#Create a list of paper_ids
paper_id_list = papers_df.rdd.map(tuple).map(lambda x: x[0]).collect()

#Create RDD of the form (user_hash_id, paper_id)
usersRDD_1 = usersRDD.map(lambda x: (x.split(";")[0], list(x.split(";")[1].split(","))))
usersRDD_1 = usersRDD_1.map(lambda x:[(x[0], i) for i in x[1]],).flatMap(lambda x:x)

#Create RDD of the form (user_id, paper_id, rating) where value of rating is 1. These are the papers that the user
#has in her library
user_paper_rating = usersRDD_1.map(lambda x: (user_id_map[x[0]], x[1], 1))

#Create RDD of the form (user_id, [paper_id], len([paper_id]))
user_paper = usersRDD.map(lambda x: (user_id_map[x.split(";")[0]], list(x.split(";")[1].split(","))))\
.map(lambda x: (x[0], x[1], len(x[1])))

def buildUnratedPapersRDD(rdd):
    unratedPaperList = []
    # Loop until unrated paper list length doesn't equal length of list of rated papers for that user.
    while not (len(unratedPaperList) == rdd[2]):
        #Choose a random paper
        random_paper_id = random.choice(paper_id_list)
        #Add to unrated paper list if it is not contained in the rated paper list. 
        if(random_paper_id not in rdd[1]):
            rdd[1].append(random_paper_id)
            unratedPaperList.append((rdd[0], random_paper_id, 0))
    return unratedPaperList

# Do a union of user_paper_rating (which has positive ratings) with a new RDD that randomly picks unrated papers
# for that particular user and assigns a rating of 0 using a function called 'buildUnratedPapersRDD'.

user_paper_rating_mix = sc.union([user_paper_rating, user_paper.map(lambda x: buildUnratedPapersRDD(x))\
                              .flatMap(lambda x:x)])

#Create a schema which has columns ('user_id', 'paper_id', 'rating') of string type
schema = StructType([StructField('user_id', IntegerType(), True),
                     StructField('paper_id', StringType(), True),
                     StructField('label', IntegerType(), True)])

#Create DF out of the schema
user_paper_rating_df = sql.createDataFrame(user_paper_rating_mix, schema)

#Separately casting paper_id to integer and it doesn't work with schema.
user_paper_rating_df = user_paper_rating_df.withColumn("paper_id", user_paper_rating_df['paper_id'].cast("integer"))

#user_paper_rating_df = user_paper_rating_df.filter(user_paper_rating_df[2] == 0)

with_paper_features_df = user_paper_rating_df.\
                             join(papers_df, papers_df['paper_idx'] == user_paper_rating_df['paper_id'])

#Selecting columns 'pages', 'year' and 'type' to represent the paper features
with_paper_features_df = with_paper_features_df.select(['user_id', 'paper_id', 'label', 'pages', 'year', 'type'])

#Create RDD of the form (user_id, paper_id)
user_paper_rating = usersRDD_1.map(lambda x: (user_id_map[x[0]], x[1]))

users_df = user_paper_rating.toDF(['user_idx', 'paper_id'])

user_paper_df = users_df.join(papers_df, papers_df['paper_idx'] == users_df['paper_id'])

#Select column 'user_id' and 'pages'
user_paper_df = user_paper_df.select(['user_idx', 'pages'])

#Add a count to each (user, paper) pair so that we can sum up number of papers per user later.
user_paper_df = user_paper_df.withColumn("count", sf.lit(1))

#Calculate the average paper length and number of papers of each user 
user_paper_df = user_paper_df.groupBy('user_idx').agg({'pages': 'mean', 'count': 'sum'})

with_user_paper_features_df = with_paper_features_df.\
                                join(user_paper_df, user_paper_df['user_idx'] == with_paper_features_df['user_id'])
with_user_paper_features_df = with_user_paper_features_df.\
                        select(['user_id', 'paper_id', 'label', 'pages', 'year', 'type', 'avg(pages)', 'sum(count)'])

#Using a string indexer for paper feature 'type' as the vector assembler doesn't accept non-numerical values
indexer = StringIndexer(inputCol="type", outputCol="typeIndex", handleInvalid="keep")
with_features_df = indexer.fit(with_user_paper_features_df).transform(with_user_paper_features_df)

with_features_df = with_features_df.select(['user_id', 'paper_id', 'label', 'pages', 'year', 'typeIndex', 'avg(pages)', 'sum(count)'])

#Imputation
imputer = Imputer(inputCols=["pages", "year", "typeIndex"], outputCols=["pages", "year", "typeIndex"])
model = imputer.fit(with_features_df)

user_paper_features_imputed_df = model.transform(with_features_df)

#One-Hot enconding
one_hot_encoder = OneHotEncoderEstimator(inputCols=["typeIndex"],
                                 outputCols=["typeIndexEncoded"])
one_hot_encoder_model = one_hot_encoder.fit(user_paper_features_imputed_df)
user_paper_features_df = one_hot_encoder_model.transform(user_paper_features_imputed_df)

#Selecting colums pages, year, type, avg_pages, sum_count to be made into a feature using the Vector Assembler
assembler = VectorAssembler(
    inputCols=["pages", "year", "typeIndexEncoded", "avg(pages)", "sum(count)"],
    outputCol="features", handleInvalid="keep")

vector_feature_df = assembler.transform(user_paper_features_df)
vector_feature_df = vector_feature_df.select(['user_id', 'paper_id', 'label', 'features'])

#Split the ratings DF randomly into training set and test set with 70% and 30% of the ratings respectively.
label_features_df = vector_feature_df.select(['label', 'features'])
(training_set, test_set) = label_features_df.randomSplit([0.8, 0.2])

#Get the evaluator instance
evaluator = RegressionEvaluator(metricName="rmse", labelCol="label", predictionCol="prediction")

#Linear SVM
svm = LinearSVC(maxIter=5, regParam=0.01)
lsvm_model = svm.fit(training_set)

#Random Forest Classifier
rf = RandomForestClassifier(numTrees=3, maxDepth=2, labelCol="label", seed=42)
rf_model = rf.fit(training_set)

#Calculate RMSE on Linear SVM model
lsvm_predictions = lsvm_model.transform(test_set)
lsvm_rmse = evaluator.evaluate(lsvm_predictions)
lsvm_df = sql.createDataFrame([(lsvm_rmse)],["lvsm_rmse"])
lsvm_df.rdd.map(tuple).saveAsTextFile("lsvm_rmse.txt")

#Calculate RMSE on Random Forest model
rf_predictions = rf_model.transform(test_set)
rf_rmse = evaluator.evaluate(rf_predictions)
rf_df = sql.createDataFrame([(rf_rmse)],["rf_rmse"])
rf_df.rdd.map(tuple).saveAsTextFile("rf_rmse.txt")
