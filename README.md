Thought for Food: Providing Quality Recipe Recommendations to the Bourgeoning Cook

Abstract: Recommender systems have gained traction in recommending movies, songs, and merchandise. Here, we look at a recommender system for food recipes using ratings and recipe datasets collected from Food.com over 18 years. We implement two separate approaches of collaborative filtering— matrix factorization and user-user similarity to recommend users unseen recipes. We’ve incorporated metrics such as top k MSE, Ranked Biased Overlap, Normalized Discounted Cummulative Gain, and Spearman Rank Correlation. Future work will include ultilizing different datasets, looking at content-based regression approaches, other similarity metrics, and exploring different evaluation metrics.

Quickstart: Run RecipeRecommender.ipynb which contains most of this project's functionality, along with explanations for our other relevant notebooks. 

Requirements:
* dask
* pyspark
* RegressionEvaluator
* ALS
* pyspark.sql
* pandas
* numpy
* pickle
* matplotlib
* Ast
* functools
* scipy.stats
* rbo
* sparse
* dask_ml
* time

This notebook provides the EDA, data preparation, modeling, results, and evaluation for our recipe recommender system.

Part1: Data Loading and EDA
Installing and Running the Project.
##[Dataset](https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions)
Download RAW_interactions and RAW_recipes datasets.
1. Shunyang Li et al. provide mulitle datasets. The RAW_interactions data is the raw data for user-recipe ratings and the RAW_recipes data is the raw data for recipes with ingredients. We use the user-recipe rating data for building matrix factorization models as desribed. The othe data are processed versions of the rating and recipe data and the ingr_map provides a tokenized dataset for the recipes
2. EDA Includes distributions for number of ratings per user, number of ratings per recipe, mean ratings, 

Part 2: Data Preparation
We remove users who rated more than 100 recipes and fewer than 4; users who had an average rating of less than 2.5; recipes with fewer than 2 ratings; recipes with an average rating less than 2.5.

Part 3: Recipe Recommender Modeling
3a: ALS Collaborative Filtering, Pyspark.
Our first model for building the recommender system. It takes the user-recipe rating data. We build an ALS collaborative filtering model to predict the rating a user will give a recipe. Evaluation: We look at Mean Squared Error (MSE), top-k-MSE, ranked biased overlap, Kendall’s Tau, normalized discounted cummulative gain, and Spearman Rank Correlation.

3b: User-user similarity, Dask.
This uses an alternative approach to predicting user ratings through similarity scoring metrics and includes different types of rating averaging for baseline evaluation. Evaluation: We look at Mean Squared Error (MSE), top-k-MSE, ranked biased overlap, Kendall’s Tau, normalized discounted cummulative gain, and Spearman Rank Correlation.
