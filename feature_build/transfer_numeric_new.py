import os
import pandas as pd
import numpy as np


def get_object_column(df_all) :
    ls_object_colname = []
    for cname in df_all.columns:
        val = df_all[cname].dtype
        if(val == np.float64 or val == np.int64):
           pass
        else:
           print(cname, val)
           ls_object_colname.append(cname)
    return ls_object_colname


def transfer_numeric_new(df_all, ls_object_colname) :

    df_all_object =df_all[['order_id']]
    for colname in ls_object_colname:
        df_just_dummies = pd.get_dummies(df_all[colname])

        new_colname = []
        for val in df_just_dummies.columns:
            new_colname.append('{colname}_{val}'.format(colname=colname, val=val))

        df_just_dummies.columns = new_colname
        df_just_dummies = df_just_dummies.assign(order_id = df_all.order_id)

        df_all_object = pd.merge(df_all_object, df_just_dummies, on=['order_id'], how='inner')
        print(colname, df_all_object.shape)


    for colname in ls_object_colname:
        del df_all[colname]

    df_all_numeric = pd.merge(df_all, df_all_object, on=['order_id'], how='inner')

    return df_all_numeric


def change_type_category(df, feature) :

    df[feature] = df[feature].map(lambda type: 'C_%s' % (str(type)))
