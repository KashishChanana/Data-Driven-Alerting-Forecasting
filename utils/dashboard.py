import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import sys
sys.path.append('../pricingml/utils')
from utils.mapping import *
import base64

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)


feecode_map, countryid_map, substier_map = initialize_mapping()
encoded_image = base64.b64encode(open('utils/component_plots.png', 'rb').read())
hourly_encoded_image = base64.b64encode(open('utils/hourly_component_plots.png', 'rb').read())

def create_substier_dcc_graph(fee_code, listing_site_id, substier, uni):

    """
    Creates graph component for a particular substier
    Parameters
    ----------------
    :param fee_code: code of fee_type being analyzed
    :param listing_site_id: listing site id on eBay
    :param substier : Dictionary with substier-wise trends, alerts and forecasts
    :param uni : substier ID

    Returns
    ----------------
    Dash graph component

    """

    graph = dcc.Graph(
        figure={
            'data': [
                {'x': substier["substier_data"][uni]["train_dates"], 'y': substier["substier_data"][uni]["y_train"],
                 'type': 'scatter', 'name': 'Train'},
                {'x': substier["substier_data"][uni]["test_dates"], 'y': substier["substier_data"][uni]["y_test"],
                 'type': 'scatter', 'name': 'Test'},
                {'x': substier["substier_data"][uni]["test_dates"],
                 'y': substier["substier_data"][uni]["predictions"],
                 'type': 'scatter', 'name': 'Predicted'},
            ],
            'layout': {
                'title':  "Analysis of " + feecode_map[int(fee_code)] + " on site " + countryid_map[listing_site_id] + " for " + substier_map[uni] + " Substier",
                'xaxis': {
                    'title': 'Datetime'
                },
                'yaxis': {
                    'title': 'Fee' + fee_code
                }
            }
        }
    )

    return graph

def get_children(fee_code, listing_site_id, substier):
    """
    Generates graph components for all substiers
    Parameters
    ----------------
    :param fee_code: code of fee_type being analyzed
    :param listing_site_id: listing site id on eBay
    :param substier : Dictionary with substier-wise trends, alerts and forecasts

    Returns
    ----------------
    List of all graph components

    """
    children = []
    for key in substier["substier_data"].keys():
        children.append(create_substier_dcc_graph(fee_code, listing_site_id, substier,key))

    return children


def dash_plot(fee_code, listing_site_id, DOD, HOH, WOW, substier):
    """
    Generates dashboard on localhost

    Parameters
    ----------------
    :param fee_code: code of fee_type being analyzed
    :param listing_site_id: listing site id on eBay
    :param DOD : Dictionary with daily trends, alerts and forecasts.
    :param HOH : Dictionary with hourly trends, alerts and forecasts.
    :param WOW : Dictionary with weekly trends, alerts and forecasts.
    :param substier : Dictionary with substier-wise trends, alerts and forecasts

    Returns
    ----------------
    None

    """

    app.layout = html.Div([
        html.Center(
            html.H4(children='Machine Learning Based Alerts For ' + feecode_map[int(fee_code)] + " On Site " + countryid_map[listing_site_id])),
        dcc.Tabs([
            dcc.Tab(label='Day on Day Analysis', children=[

                dcc.Graph(
                    figure={
                        'data': [
                            {'x': DOD["DOD_data"]["train_dates"], 'y': DOD["DOD_data"]["y_train"],
                             'type': 'scatter', 'name': 'Train'},
                            {'x': DOD["DOD_data"]["test_dates"], 'y': DOD["DOD_data"]["y_test"],
                             'type': 'scatter', 'name': 'Test'},
                            {'x': DOD["DOD_data"]["test_dates"], 'y': DOD["DOD_data"]["predictions"],
                             'type': 'scatter', 'name': 'Predicted'},
                            {'x': DOD["DOD_forecast"]["ds"], 'y': DOD["DOD_forecast"]["yhat"],
                             'type': 'scatter', 'name': 'Forecast'},
                            {'x': DOD["DOD_data"]["test_dates"], 'y': DOD["DOD_data"]["lower_bound"],
                             'type': 'scatter', 'name': 'Lower Bound'},
                            {'x': DOD["DOD_data"]["test_dates"], 'y': DOD["DOD_data"]["upper_bound"],
                             'type': 'scatter', 'name': 'Upper Bound', 'fill': 'tonexty',
                             'fillcolor': 'rgb(255, 245, 189)', 'mode': 'None', 'opacity': '0.2'},

                        ],
                        'layout': {
                            'title': "Daily Trends of " + feecode_map[int(fee_code)] + " on site " + countryid_map[listing_site_id],
                            'xaxis': {
                                'title': 'Datetime'
                            },
                            'yaxis': {
                                'title': 'Fee' + fee_code
                            }
                        }
                    }
                ),
                html.Center(
                html.Img(src='data:image/png;base64,{}'.format(encoded_image.decode()), style={'Align': 'center'})
                ),
                html.Div(children='''
                Alerts raised in last 30 days:
                '''),
                html.Div(
                    children=html.Ul([html.Li(x) for x in DOD["DOD_alerts"]])
                )
            ]),

            dcc.Tab(label='12 Hourly', children=[
                dcc.Graph(
                    figure={
                        'data': [
                            {'x': HOH["HOH_data"]["train_dates"], 'y': HOH["HOH_data"]["y_train"],
                             'type': 'scatter', 'name': 'Train'},
                            {'x': HOH["HOH_data"]["test_dates"], 'y': HOH["HOH_data"]["y_test"],
                             'type': 'scatter', 'name': 'Test'},
                            {'x': HOH["HOH_data"]["test_dates"], 'y': HOH["HOH_data"]["predictions"],
                             'type': 'scatter', 'name': 'Predicted'},
                            {'x': HOH["HOH_forecast"]["ds"], 'y': HOH["HOH_forecast"]["yhat"],
                             'type': 'scatter', 'name': 'Forecast'},
                            {'x': HOH["HOH_data"]["test_dates"], 'y': HOH["HOH_data"]["lower_bound"],
                             'type': 'scatter', 'name': 'Lower Bound'},
                            {'x': HOH["HOH_data"]["test_dates"], 'y': HOH["HOH_data"]["upper_bound"],
                             'type': 'scatter', 'name': 'Upper Bound', 'fillcolor': 'rgb(255, 245, 189)', 'mode': 'None', 'opacity': '0.5'},
                        ],
                        'layout': {
                            'title': "Hourly analysis of " + feecode_map[int(fee_code)] + " on site " + countryid_map[listing_site_id],
                            'xaxis': {
                                'title': 'Datetime'
                            },
                            'yaxis': {
                                'title': 'Fee' + fee_code
                            }
                        }
                    }
                ),
                html.Center(
                html.Img(src='data:image/png;base64,{}'.format(hourly_encoded_image.decode()), style={'Align': 'center'})
                ),
                html.Div(
                    children=html.Ul([html.Li(x) for x in HOH["HOH_alerts"]])
                )
            ]),
            dcc.Tab(label='Week on Week', children=[
                dcc.Graph(
                    figure={
                        'data': [
                            {'x': WOW["WOW_data"]["train_dates"], 'y': WOW["WOW_data"]["y_train"],
                             'type': 'scatter', 'name': 'Train'},
                            {'x': WOW["WOW_data"]["test_dates"], 'y': WOW["WOW_data"]["y_test"],
                             'type': 'scatter', 'name': 'Test'},
                            {'x': WOW["WOW_data"]["test_dates"], 'y': WOW["WOW_data"]["predictions"],
                             'type': 'scatter', 'name': 'Predicted'},

                        ],
                        'layout': {
                            'title': 'Week-on-Week analysis of ' + feecode_map[int(fee_code)] + " on site " + countryid_map[listing_site_id],
                            'xaxis': {
                                'title': 'Datetime'
                            },
                            'yaxis': {
                                'title': 'Fee' + fee_code
                            }
                        }

                    }
                ),
                html.Div(
                    children=html.Ul([html.Li(x) for x in WOW["WOW_alerts"]])
                )
            ]),
            dcc.Tab(label='Substier', children=
                get_children(fee_code, listing_site_id, substier),
            ),

        ]),

    ])

    app.run_server(host='127.0.0.1', port=1234)
