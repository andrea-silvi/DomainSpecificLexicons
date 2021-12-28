about liblinear : (https://www.csie.ntu.edu.tw/~cjlin/papers/guide/guide.pdf)

guide for python : https://github.com/cjlin1/liblinear/tree/master/python 

"""
For one-class SVM models, label_idx is ignored and b=-rho is
    returned from get_decfun(). That is, the decision function is
    w*x+b = w*x-rho.
"""

"""
Note that in get_decfun_coef, get_decfun_bias, and get_decfun, feat_idx
    starts from 1, while label_idx starts from 0. If label_idx is not in the
    valid range (0 to nr_class-1), then a NaN will be returned; and if feat_idx
    is not in the valid range (1 to nr_feature), then a zero value will be
    returned. For regression models, label_idx is ignored.
"""
