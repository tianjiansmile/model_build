from pandas import *
from numpy import *

def Equal_Freqency(var,n):
    var = DataFrame(var)
    var.columns=['var']
    var['f']=1
    var = var[var['var'].isnull()==False]


    var_new = var.groupby(by='var').count()
    var_new['var']=var_new.index
    var_new['F'] = var_new['f'].cumsum()
    # print (var_new)
    f_avg = var.shape[0]/n
    bin = [-inf]
    for i in range(n):
        bin_cut = var_new.loc[abs((var_new['F']-(i+1)*f_avg)).idxmin(),'var']
        if bin_cut not in bin:
            bin.append(bin_cut)
    bin = bin[:bin.__len__()-1]
    bin.append(inf)
    # print (bin)
    var['bin'] = cut(var['var'],bin)
    # d2 = var.groupby('bin')
    # print (d2.count())
    return cut(var['var'],bin)

if __name__ == '__main__':
    data = read_csv('sample_file_v5.txt')
    var = data['call']
    # print (var)

    bin = Equal_Freqency(var,5)
    # print (bin)

