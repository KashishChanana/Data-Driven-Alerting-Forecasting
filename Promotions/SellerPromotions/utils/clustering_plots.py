import numpy as np
import matplotlib.pyplot as plt


def comparison_plots(X_HAC_0, X_HAC_1):
    """
    Compares the two plots on different features and displays plots
    :param X_HAC_0: found cluster 0
    :param X_HAC_1: found cluster 1
    :return: None, displays plots
    """
    info_0 = X_HAC_0.describe()

    info_0 = info_0[['dec_promo_discount', 'dec_base_charge',
                     'dec_list_items', 'pre_dec_promo_discount', 'pre_dec_base_charge',
                     'pre_dec_list_items', 'post_dec_promo_discount', 'post_dec_base_charge',
                     'post_dec_list_items']]

    info_1 = X_HAC_1.describe()

    info_1 = info_1[['dec_promo_discount', 'dec_base_charge',
                     'dec_list_items', 'pre_dec_promo_discount', 'pre_dec_base_charge',
                     'pre_dec_list_items', 'post_dec_promo_discount', 'post_dec_base_charge',
                     'post_dec_list_items']]


    ind = np.arange(2)
    idx = 1
    fig = plt.figure(figsize=(20, 20))
    for x in ("dec_promo_discount", "dec_base_charge",  "dec_list_items", "pre_dec_promo_discount","pre_dec_base_charge",
              "pre_dec_list_items", "post_dec_promo_discount", "post_dec_base_charge", "post_dec_list_items" ):

        plt.subplot(3, 3, idx)
        width = 0.3
        plt.title(x)
        plt.bar(ind, info_0.loc["mean"][x], width, label="Cluster 0")
        plt.bar(ind+width, info_1.loc["mean"][x], width, label=" Cluster 1")
        plt.xlabel("Clusters")
        plt.legend()
        idx+= 1

    plt.show()