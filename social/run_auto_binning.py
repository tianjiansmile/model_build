import pandas as pd
import numpy as np
from Stepwise.Stepwise import Stepwise
from utils.auto_binning import auto_binning
from analysis.analysis import analysis


features = ['model_score_pd7']

#features = ['phone_apply', 'high_risk_total']
# features = ['last_1_week_total', 'last_2_week_total', 'last_3_week_total']

label_col = 'slpd7'

data = pd.read_csv('data/all_online_model_score.csv')

del data['create_time']

r = Stepwise(data, 'order_id', 'label')

data_in = r.data_to_woe(data, features, label_col, 20, False)
print(data_in)
    
