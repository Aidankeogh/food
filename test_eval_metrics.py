import sys
import pandas as pd
from eval_metrics import RecEvalMetrics as rem

eval_test = sys.argv[1]

eval_test = pd.read_csv(eval_test)

print("K MSE:")
print(rem.top_k_evaluator(eval_test, 3))
print('\n')

print("Percent in K:")
print(rem.percent_in_top_ratings(eval_test, 1))
print('\n')

print("RBO")
print(rem.rbo_evaluation(eval_test, 2))
print('\n')

print("Kendall's Tau:")
print(rem.kendalls_tau(eval_test))
print('\n')

print("nDCG:")
print(rem.nDCG_evaluation(eval_test))
print('\n')

print("Personalization:")
print(rem.personalization(eval_test, 1))