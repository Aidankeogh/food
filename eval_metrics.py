# Object for providing metrics to evaluate the recipe recommender system

""" USAGE:

			from eval_metrics import RecEvalMetrics as rem

			rem.top_k_evaluator(predictions, k)

"""

# !pip install rbo
# !pip install recmetrics

from functools import reduce
from scipy.stats import kendalltau
from sklearn.metrics import mean_squared_error, ndcg_score
import pandas as pd
import numpy as np
import recmetrics
import rbo

class RecEvalMetrics(object):

	# PRIMARY METHODS:

	# Takes user-recipe rating predictions dataframe, returns mean squarred error for top k
	# recipes of each user predicted ratings
	""" Parameters: 	
		predictions: Dataframe of true and predicted ratings, default 20
		k: Top k predicted ratings to evaluate with mse
	"""
	@staticmethod
	def top_k_evaluator(predictions, k = 20):
		users = list(predictions.drop_duplicates(subset = ['userId'])['userId'])
		
		top_k_predictions = []
		for user in users:
			user_ratings = predictions[(predictions['userId'] == user)]
			top_k_user_ratings =  user_ratings.sort_values(by = ['prediction'], ascending = False).head(k)
			top_k_predictions.append(top_k_user_ratings)
		top_k_predictions_df = pd.concat(top_k_predictions, ignore_index = True)

		k_mse = mean_squared_error(list(top_k_predictions_df['rating']), list(top_k_predictions_df['prediction']))
		
		return(k_mse) 


	# Takes in user-recipe rating predictions dataframe, returns percent of recipes that ended
	# up in someone's top k.  Larger value means more personalization
	""" Parameters:
		predictions: Dataframe of true and predicted ratings
		k: Top k recipes to count in percentage, default 20
	"""
	@staticmethod
	def percent_in_top_ratings(predictions, k = 20):
		total_recipes = len(predictions.drop_duplicates(subset = ['recipeId']))
		users = list(predictions.drop_duplicates(subset = ['userId'])['userId'])

		top_k_predictions = set()
		for user in users:
			user_ratings = predictions[(predictions['userId'] == user)]
			user_pred_ordered = list(user_ratings.sort_values(by = ['prediction'], ascending = False)['recipeId'])
			top_k_user_recipes = user_pred_ordered[:k]
			top_k_predictions.update(top_k_user_recipes)
		
		top_recipes_count = len(top_k_predictions)
		
		return(top_recipes_count/total_recipes)


	# Take in user-recipe rating predictios dataframe, returns ranked biased overlap
	# between top k predicted ratings and top k actual ratings
	# Refer to: https://github.com/changyaochen/rbo
	""" Parametes:
		predictions: Dataframe of true and predicted ratings
		k: Number of k recipes in the ranked list to evaluate with RBO, default 20
	"""
	@staticmethod
	def rbo_evaluation(predictions, k = 20):
		users = list(predictions.drop_duplicates(subset = ['userId'])['userId'])

		rbos = []
		for user in users:
			user_ratings = predictions[(predictions['userId'] == user)]
			user_actual_ordered = list(user_ratings.sort_values(by = ['rating'], ascending = False)['recipeId'])
			user_pred_ordered = list(user_ratings.sort_values(by = ['prediction'], ascending = False)['recipeId'])
			top_k_user_actual = user_actual_ordered[:k]
			top_k_user_pred = user_pred_ordered[:k]
			user_rbo = rbo.RankingSimilarity(top_k_user_actual, top_k_user_pred).rbo()
			rbos.append(user_rbo)

		return(np.mean(rbos), np.median(rbos))


	# Takes in user-recipe rating predictions dataframe, returns the Kendalls Tau evaluation
	# between actual ratings and predicted ratings
	""" Parameters:
		predictions: Dataframe of true and predicted ratings	
	"""
	@staticmethod
	def kendalls_tau(predictions):
		users = list(predictions.drop_duplicates(subset = ['userId'])['userId'])

		tau = []
		for user in users:
			user_ratings = predictions[(predictions['userId'] == user)]
			user_actual_ordered = list(user_ratings.sort_values(by = ['rating'], ascending = False)['recipeId'])
			user_pred_ordered = list(user_ratings.sort_values(by = ['prediction'], ascending = False)['recipeId'])
			user_tau, user_p_value = kendalltau(user_actual_ordered, user_pred_ordered)
			tau.append(user_tau)

		return(np.mean(tau), np.median(tau))


	# Takes in user-recipe rating predictions dataframe, returns the normalized discounted cummulative gain
	# evaluation between actual ratings and predicted ratings
	""" Parameters:
		predictions: Dataframe of ture and predicted ratings
		k: Number of k recipes in the ranked list to evaluate, default None	
	"""
	@staticmethod
	def nDCG_evaluation(predictions, k = None):
		users = list(predictions.drop_duplicates(subset = ['userId'])['userId'])

		ndcg = []
		for user in users:
			user_ratings = predictions[(predictions['userId'] == user)]
			relevance = np.asarray([list(user_ratings['rating'])])
			preds = np.asarray([list(user_ratings['prediction'])])
			score = ndcg_score(relevance, preds, k=k)
			ndcg.append(score)

		return(np.mean(ndcg), np.median(ndcg))


	# Takes in user-recipe rating predictions dataframe, returns the personalizaion score, as dissimilarity
	# between all users list of recomendations.  Higher value means more pesonalization
	# Refer to: https://github.com/statisticianinstilettos/recmetrics
	""" Parameters:
		predictions: Dataframe of ture and predicted ratings
		k: Number of k recipes in the ranked list to compare among all users for personalization, default 10
	"""
	@staticmethod
	def personalization(predictions, k = 10):
		users = list(predictions.drop_duplicates(subset = ['userId'])['userId'])

		recs = []
		for user in users:
			user_ratings = predictions[(predictions['userId'] == user)]
			user_pred_ordered = list(user_ratings.sort_values(by = ['prediction'], ascending = False)['recipeId'])
			top_k_user_recipes = user_pred_ordered[:k]
			recs.append(top_k_user_recipes)

		score = recmetrics.personalization(predicted = recs)

		return(score)









		

		
