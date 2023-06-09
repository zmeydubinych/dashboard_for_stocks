import pandas as pd
import dash
from dash import html
from dash import dcc
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import plotly.express as px
from dash import no_update
import numpy as np
import matplotlib.pyplot as plt
import requests
from bs4 import BeautifulSoup

app = dash.Dash(__name__)

app.config.suppress_callback_exceptions = True

app.layout = html.Div(children=[html.H1('Dashboard for stockmarket with regression analysis',
                                        style={'textAlign': 'center',
                                               'color': '#503D36',
                                               'front-sixe': 24}),

                                html.Div([
                                    html.Div([
                                        html.Div(
                                            [
                                                html.H2('Report Type:', style={'margin-right': '2em'}),
                                            ]
                                        ),
                                        dcc.Dropdown(id='input-type',
                                                     options=[{'label': 'Intrinsic Value',
                                                               'value': 'OPT1'},
                                                              {'label': 'Real Price',
                                                               'value': 'OPT2'}],
                                                     placeholder='Select a report type',
                                                     style={'width': '80%', 'padding': '3px', 'front-size': '20px',
                                                            'text-align-last': 'center'}
                                                     )
                                    ], style={'display': 'flex'}),
                                    html.Div([
                                        html.Div(
                                            [
                                                html.H2('Choose Year:', style={'margin-right': '2em'})
                                            ]
                                        ),
                                        dcc.Dropdown(id='input-year',
                                                     options=[{'label': 'i', 'value': 'Year'}],
                                                     placeholder="Select a year",
                                                     style={'width': '80%', 'padding': '3px', 'font-size': '20px',
                                                            'text-align-last': 'center'}),
                                    ], style={'display': 'flex'}),
                                ]),
                                html.Div([], id='plot1'),
                                html.Div([
                                    html.Div([], id='plot2'),
                                    html.Div([], id='plot3')
                                ], style={'display': 'flex'}),
                                html.Div([
                                    html.Div([], id='plot4'),
                                    html.Div([], id='plot5')
                                ], style={'display': 'flex'}),

                                ])
if __name__ == '__main__':
    app.run_server()
