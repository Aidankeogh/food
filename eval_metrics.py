# Object for providing metrics to evaluate the recommender system

""" USAGE:

			from eval_metrics import RecEvalMetrics as rem

			rem.top_k_evaluator(predictions, k)

"""


from functools import reduce
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, countDistinct, collect_set
from pyspark.ml.evaluation import RegressionEvaluator
import numpy as np
import rbo

class RecEvalMetrics(object):

	# PRIMARY METHODS:

	# Takes user-recipe rating predictions pyspark dataframe, returns evaluation metric for top k
	# recipes of each user predicted ratings
	""" Parameters: 	
		predictions: Dataframe of true and predicted ratings
		metricName: Metric used to evaluted predictions (rmse, mse, mae, r2)
		k: Top k predicted ratings to evaluate
	"""
	@staticmethod
	def top_k_evaluator(predictions, metricName, k):
		users = list(predictions.select('userId').distinct().toPandas()['userId'])
		
		top_k_predictions = []
		for user in users:
			user_ratings = predictions.where(predictions.userId == user)
			user_ratings = user_ratings.orderBy('prediction', ascending = False)
			top_k_user_ratings = user_ratings.limit(k)
			top_k_predictions.append(top_k_user_ratings)
		top_k_predictions_df = reduce(DataFrame.unionAll, top_k_predictions)
		
		evaluator = RegressionEvaluator(metricName=metricName, labelCol='rating', predictionCol='prediction')
		evalMetric = evaluator.evaluate(top_k_predictions_df)
		
		return(evalMetric) 


	# Takes in user-recipe rating predictions pyspark dataframe, returns percent of recipes that ended
	# up in someone's top k
	""" Parameters:
		predictions: Dataframe of true and predicted ratings
		k: Top k recipes to count in percentage
	"""
	@staticmethod
	def percent_in_top_ratings(predictions, k):
		total_recipes = predictions.select('recipeId').distinct().count()
		users = list(predictions.select('userId').distinct().toPandas()['userId'])

		top_k_predictions = []
		for user in users:
			user_ratings = predictions.where(predictions.userId == user)
			user_ratings = user_ratings.orderBy('prediction', ascending = False)
			top_k_user_ratings = user_ratings.limit(k)
			top_k_predictions.append(top_k_user_ratings)
		top_k_predictions_df = reduce(DataFrame.unionAll, top_k_predictions)

		top_recipes_count = top_k_predictions_df.select('recipeId').distinct().count()

		return(top_recipes_count/total_recipes)


	# Take in user-recipe rating predictios pyspark dataframe, returns ranked biased overlap
	# between top k predicted ratings and top k actual ratings
	# Refer to: https://github.com/changyaochen/rbo
	""" Parametes:
		predictions: Dataframe of true and predicted ratings
		k: Number of k recipes in the ranked list to evaluate with RBO
	"""
	@staticmethod
	def rbo_evaluation(predictions, k):
		users = list(predictions.select('userId').distinct().toPandas()['userId'])

		rbo_values = []
		for user in users:
			user_ratings = predictions.where(predictions.userId == user)
			user_actual_ordered = user_ratings.orderBy('rating', ascending = False)
			user_pred_ordered = user_ratings.orderBy('prediction', ascending = False)
			top_k_user_actual = user_actual_ordered.limit(k)
			top_k_user_pred = user_pred_ordered.limit(k)
			recipes_actual = list(top_k_user_actual.select('recipeId').toPandas()['recipeId'])
			recipes_pred = list(top_k_user_pred.select('recipeId').toPandas()['recipeId'])
			user_rank_similarity = rbo.RankingSimilarity(recipes_actual, recipes_pred).rbo()
			rbo_values.append(user_rank_similarity)

		return(np.mean(rbo_values))


	# SECDONDARY METHODS:

	# WARNING: These methods have not been tested rigorously as it's unlikely they are used


	# Takes user-recipe rating predictions pyspark dataframe, returns evaluation metric for top K 
	# recipes of a single user's predicted ratings
	""" Parameters: 	
		predictions: Dataframe of true and predicted ratings for just a single user
		metricName: Metric used to evaluted predictions (rmse, mse, mae, r2)
		k: Top k predicted ratings to evaluate
	"""
	@staticmethod
	def one_user_top_k_evaluator(predictions, metricName, k):
		predictions = predictions.orderBy('prediction', ascending = False)
		predictions = predictions.limit(k)
		evaluator = RegressionEvaluator(metricName=metricName, labelCol='rating', predictionCol='prediction')
		evalMetric = evaluator.evaluate(predictions)
		
		return(evalMetric)



	# Takes pyspark dataframe N recipes by M Users, returns the percent of recipes that ended up
	# in someones top k
	""" Parameters:
		rankings: Recipe rankings for every user in every column
		k: For evaluating top k
	"""
	@staticmethod
	def percent_in_top_k(rankings, k):
		total_recipes = rankings.count()
		top_k_rankings = rankings.limit(k)
		all_top_k_recipes = top_k_rankings.agg(*(
			countDistinct(col(c)).alias(c) for c in top_k_rankings.columns))

		return(all_top_k_recipes/total_recipes)


	# Takes pyspark dataframe k rows by M Users, returns the percent of recipes that ended up
	# in someones top k
	""" Parameters:
		rankings: Top k recipes for each user in each column
		total_recipes: Total recipes in the dataset
	"""
	@staticmethod
	def percent_in_top_k(top_k_rankings, total_recipes):
		all_top_k_recipes = top_k_rankings.agg(*(
			countDistinct(col(c)).alias(c) for c in top_k_rankings.columns))

		return(all_top_k_recipes/total_recipes)









		

		
