import argparse
from connection import *
from analysis.DoD import *
from analysis.HoH import *
from analysis.WoW import *
from analysis.substier import *
from utils.dashboard import *
from query.run_query import *
from utils.preprocess import *
from query.generate_query import *
from run import *

# Parse command line arguments

parser = argparse.ArgumentParser(description='Connecting to Database via Proxy')
parser.add_argument('--connect', help='Do you want to set up a new connection [y/n] ?')
parser.add_argument('--feecode', help='Enter Fee Code to be Analyzed', type=str)
parser.add_argument('--siteid', help='Enter Listing Site ID', type=str)
parser.add_argument('--multi', help='Do you want to perform multivariate forecasting [y/n] ?')
parser.add_argument('--hourly', help='The number of hours you want to aggregate the data on', type=str)
parser.add_argument('--WoW', help='Week on Week Analysis [y/n]', type=str)
parser.add_argument('--substier', help='Analysis by SUBS_TIER [y/n]', type=str)
parser.add_argument('--model_name', help='Specify the ML model you want to run')
args = parser.parse_args()

# Set up a connection
if args.connect == 'y':
    print("\n Connecting to Database... \n")
    connect()

# Initialize variables
fee_code = args.feecode
listing_site_id = args.siteid
multi = True if args.multi == 'y' else False
hours = args.hourly
model_name = args.model_name

# Query the data, create and clean dataframes

substier = {}
DOD = {}
WOW = {}
HOH = {}

# Performs substier-wise analysis
if args.substier:
    substier = substier_analysis(listing_site_id, fee_code, model_name)

# Performs Day-on-day multivariate analysis
if args.multi == 'y':
    DOD = DOD_analysis(listing_site_id, fee_code, model_name)

# Performs week-on-week and/or hourly multivariate analysis
if args.WoW or args.hourly:

    query, fee_column = generate_query_multi_hour(listing_site_id, fee_code)
    df = run_query(query, columns=['datetime', 'hour', fee_column, 'total_sale_price', 'total_promo_discount'])
    df = clean(df, columns=[fee_column, 'hour', "total_sale_price", 'total_promo_discount'])

    if args.WoW:
        df_week = df.copy()
        WOW = WOW_analysis(df_week, fee_column, listing_site_id, fee_code, "Lasso")

    if args.hourly:
        df_hourly = df.copy()
        HOH = HOH_analysis(df_hourly, fee_column, listing_site_id, fee_code, model_name, args.hourly)

# Launches dashboard
if model_name != "Prophet-Multi":
    dash_plot(fee_code, listing_site_id, DOD, HOH, WOW, substier)
