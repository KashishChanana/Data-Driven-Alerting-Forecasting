import pandas as pd
import sys

sys.path.append('../pricingml/utils/')


def initialize_mapping():
    """
    Maps IDs to names for fee, country and substier

    Returns
    ----------------
    Mapping dictionaries of fee, countries, substiers
    """
    fee_code_name_map = pd.read_csv('../pricingml/utils/feecode-map.csv', header=None, index_col=0,
                                    squeeze=True).to_dict()

    country_id_name_map = {'0': "US"
                           }
    substier_map = {
        '1.0': 'SUBSTIER-NAME'

    }
    return fee_code_name_map, country_id_name_map, substier_map
