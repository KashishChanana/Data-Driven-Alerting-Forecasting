
def clustering_fs(df):
    """
    Selects necessary features from the master dataframe
    :param df: master dataframe
    :return: df_clusters: subset of master dataframe with selected features
    """

    features = ['SELLER_RESIDENCE_COUNTRY_ID', 'IS_BUSINESS_SELLER' ,'IS_SELLER_VAT_EXEMPT',
                'HAS_USER_GROUP', 'HAS_EBAY_PLUS', 'IS_DIRECT_CHARITY_SELLER', 'IS_LMO_SELLER',
                'TOP_SELLER_LEVEL', 'SUBS_TIER', 'SELLER_SPS_LEVEL', 'PROMO_LEVEL_ONE_PROPOSAL_ID',
                'IS_P20_SELLER', 'dec_promo_discount', 'dec_base_charge', 'dec_list_items',
                'pre_dec_promo_discount', 'pre_dec_base_charge', 'pre_dec_list_items',
                'post_dec_promo_discount', 'post_dec_base_charge', 'post_dec_list_items']

    df_clusters = df[features]

    return df_clusters