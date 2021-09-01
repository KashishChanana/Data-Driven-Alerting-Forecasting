from sklearn.preprocessing import normalize

features_scaled = ['dec_promo_discount', 'dec_base_charge',
       'dec_list_items', 'pre_dec_promo_discount', 'pre_dec_base_charge',
       'pre_dec_list_items', 'post_dec_promo_discount', 'post_dec_base_charge',
       'post_dec_list_items']

def scale(df_clusters):
       """
       Normalizes/Scales features in the dataframe
       :param df_clusters: dataframe
       :return: scaled dataframe
       """
       df_clusters[features_scaled] = normalize(df_clusters[features_scaled])
       return df_clusters