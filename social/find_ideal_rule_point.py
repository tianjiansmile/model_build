import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 找到特征的最佳切割点

features = ['model_score_pd7', 'model_score_pd7_m'
            ]

label_col = 'slpd7'


tb = pd.read_csv('data/all_online_model_score_map.csv')

for feature in features :
    feature_label = []
    feature_label.append(feature)
    feature_label.append(label_col) 

    tb_feature_label = tb[feature_label]

    tb_good = tb_feature_label[tb_feature_label[label_col] == 0]
    tb_bad = tb_feature_label[tb_feature_label[label_col] == 1]

    nr_good = len(tb_good)
    nr_bad = len(tb_bad)

    avg_bad_rate = nr_bad/(nr_good + nr_bad)

    feature_min = tb_feature_label[feature].min()
    feature_max = tb_feature_label[feature].max()

    nr_bin = 100

    width = (feature_max - feature_min)/(nr_bin - 1)

    #assume the feature is positive correlation with label
    best_bad_rate = -1
    best_ks = -1
    rule_cutpoint = -1
    rule_badrate = -1
    for i in range (0, nr_bin - 1) :
        cut_point = feature_max - width * i

        nr_good_cut = len(tb_good[tb_good[feature] > cut_point])
        nr_bad_cut = len(tb_bad[tb_bad[feature] > cut_point])

        nr_good_ratio = nr_good_cut/nr_good
        nr_bad_ratio = nr_bad_cut/nr_bad

        if nr_good_ratio == 0 :
           continue

        bad_rate = nr_bad_cut/(nr_good_cut + nr_bad_cut)

        #skip extreme point while sample is too short
        if nr_good_ratio < 0.01 : 
           continue

        if best_bad_rate == -1 or bad_rate > best_bad_rate :
           best_cut = cut_point
           best_bad_rate = bad_rate 

        if bad_rate > 2 * avg_bad_rate :
           if rule_cutpoint  == -1 or cut_point < rule_cutpoint :
              rule_cutpoint = cut_point
              rule_badrate = bad_rate

    
    if best_cut != -1 :
       print ("** feature %s, best_cut = %.3f, best_bad_rate = %.3f, nr_bad = %d, nr_good = %d"   
              % (feature, best_cut, best_bad_rate, len(tb_bad[tb_bad[feature] > best_cut]), len(tb_good[tb_good[feature] > best_cut])))
              #len(tb_bad[tb_bad[feature] > cut_point])/len(tb_feature_label[tb_feature_label[feature] > cut_point]) ))    
    if rule_cutpoint == -1 :
       print ("** feature %s , no rule cut point" % (feature)) 
    else :
       print ("** feature %s, rule_cut_point = %.3f, bad_rate = %.3f, nr_bad = %d, nr_good = %d" 
              % (feature, rule_cutpoint, rule_badrate, len(tb_bad[tb_bad[feature] > rule_cutpoint]), len(tb_good[tb_good[feature] > rule_cutpoint])))   

    if best_cut != -1 and rule_cutpoint != -1:
       rule_bound_cutpoint = -1
       rule_bound_badrate = -1
       for i in range (0, nr_bin - 1) :
           cut_point = feature_max - width * i
           if cut_point > best_cut :
              continue

           nr_good_cut = len(tb_good[(tb_good[feature] > cut_point) & (tb_good[feature] < best_cut)])
           nr_bad_cut = len(tb_bad[(tb_bad[feature] > cut_point) & (tb_bad[feature] < best_cut)])

           nr_good_ratio = nr_good_cut/nr_good
           nr_bad_ratio = nr_bad_cut/nr_bad

           if nr_good_ratio == 0 :
              continue
           
           bad_rate = nr_bad_cut/(nr_good_cut + nr_bad_cut)

           #skip extreme point while sample is too short
           #if nr_good_ratio < 0.01 :
           #   continue

           if bad_rate > 2 * avg_bad_rate :
              if rule_bound_cutpoint  == -1 or cut_point < rule_bound_cutpoint :
                 rule_bound_cutpoint = cut_point
                 rule_bound_badrate = bad_rate

       if rule_bound_cutpoint != -1 :
           print ("** feature %s, rule_bound_cut_point = %.3f, bad_rate = %.3f, nr_bad = %d, nr_good = %d"
              % (feature, rule_bound_cutpoint, rule_bound_badrate, len(tb_bad[(tb_bad[feature] > rule_bound_cutpoint) & (tb_bad[feature] < best_cut)]), len(tb_good[(tb_good[feature] > rule_bound_cutpoint) & (tb_good[feature] < best_cut)])))
           

          
        
        


    
