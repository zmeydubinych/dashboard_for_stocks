import pandas as pd
import numpy as np
import dash
from dash import dcc, html, Input, Output, callback
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf

# Блок дашборда

app = dash.Dash(__name__)

app.config.suppress_callback_exceptions = True

app.layout = html.Div(children=[html.H1('Dashboard: market price and inner price of stocks',
                                        style={'textAlign': 'center',
                                               'color': '#8a2be2',
                                               'front-sixe': 20}),
                                html.Div([
                                    html.Div(
                                        [
                                            html.H2('Ticket_module:', style={
                                                'margin-right': '2em'}),
                                        ]
                                    ),
                                    html.Div([
                                        dcc.Input(
                                            id='input_text',
                                            placeholder='input ticket'
                                        )])
                                ]),
                                html.Br(),
                                html.Div([], id='article'),
                                html.Div([
                                    html.Div(
                                        [
                                            html.H2('Date_range:', style={
                                                'margin-right': '2em'}),
                                        ]
                                    ),
                                    html.Div([
                                        dcc.Dropdown(
                                            id='date-dropdown',
                                            options=[
                                                {'label': '3 MONTHS',
                                                    'value': '3mo'},
                                                {'label': '6 MONTHS',
                                                    'value': '6mo'},
                                                {'label': '9 MONTHS',
                                                    'value': '9mo'},
                                                {'label': '12 MONTHS',
                                                    'value': '12mo'},
                                                {'label': '2 YEARS',
                                                    'value': '2y'},
                                                {'label': '3 YEARS',
                                                    'value': '3y'},
                                                {'label': '4 YEARS',
                                                    'value': '4y'},
                                                {'label': '5 YEARS',
                                                    'value': '5y'},
                                                {'label': 'MAX', 'value': 'max'}],
                                            value='max',
                                            style={'width': '30%'}
                                        )
                                    ])
                                ]),
                                html.Div([], id='plot'),
                                html.Br(),
                                html.Div([
                                    html.Button('Make a prediction, this may take several minutes',
                                                id='submit-val', n_clicks=0)
                                ]),
                                html.Div([], id='prediction')
                                ])

# Callback для ввода данных в нашу функцию и возврат графика в дашбоард


@app.callback(
    [
        Output(component_id='article', component_property='children'),
        Output(component_id='plot', component_property='children'),
        Output(component_id='prediction', component_property='children')
    ],
    [
        Input(component_id='input_text', component_property='value'),
        Input(component_id='submit-val', component_property='n_clicks'),
        Input(component_id='date-dropdown', component_property='value')
    ]
)
def update_output_div(input_value, submit_val, date_range):
    ctx = dash.callback_context
    if not ctx.triggered:
        button_id = 'No clicks yet'
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if input_value is None or input_value == '':
        return '', '', ''

    if button_id == 'submit-val':
        return get_output(input_value, date_range)
    else:
        return get_output(input_value, date_range)[0], \
            get_output(input_value, date_range)[1], \
            ''


def get_output(entered_ticket, date_range):
    data_stock = yf.Ticker(entered_ticket)
    stock_df = data_stock.history(
        period=date_range, interval='1mo')
    stock_df.reset_index(inplace=True)
    close_price = stock_df[['Date', 'Close']]
    low_price = stock_df[['Date', 'Low']]
    close_price.columns = ['Date', 'Close_price']
    max_close = close_price['Close_price'].max()
    min_low = low_price['Low'].min()
    max_close_index = close_price['Close_price'].idxmax()
    min_low_index = low_price['Low'].idxmin()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=close_price['Date'],
                             y=close_price['Close_price'],
                             name='close')
                  )
    fig.add_trace(go.Scatter(x=low_price['Date'],
                             y=low_price['Low'],
                             name='low')
                  )
    fig.add_trace(go.Scatter(x=[close_price['Date'][max_close_index]],
                             y=[max_close],
                             mode='markers',
                             marker=dict(color='red', size=10),
                             name='max_close')
                  )
    fig.add_trace(go.Scatter(x=[low_price['Date'][min_low_index]],
                             y=[min_low],
                             mode='markers',
                             marker=dict(color='green', size=10),
                             name='min_low'))

    fig.add_annotation(x=close_price['Date'][max_close_index],
                       y=max_close,
                       text=f'Max close: {max_close:.2f}',
                       showarrow=False
                       )
    fig.add_annotation(x=low_price['Date'][min_low_index],
                       y=min_low,
                       text=f'Min low: {min_low:.2f}',
                       showarrow=False
                       )

    fig.update_layout(legend_orientation="h")
    fig.update_yaxes(tickprefix="$", showgrid=True)

    info = data_stock.info

    # def get_block(label, value): 
    #     return html.Div([label, value])

    text = html.Div([
        html.Div(['Name: ', info['longName']]),
        html.Div(['Country: ', info['longName']]),
        html.Div(['City: ', info['city']]),
        html.Div(['ZIP: ', info['zip']]),
        html.Div(['Address: ', info['address1']]),
        html.Div(['Industry: ', info['industry']]),
        html.Div(['Industry: ', info['longName']]),
        html.Div(['Website: ', html.A(info['website'], href=info['website'])]),
        html.Br(),
        html.Div(['Bussiness Summary:', html.P(info['longBusinessSummary'])])
    ])

    text2 = html.Div([
        html.P(info['recommendationKey'])
    ])

    return [html.Article(text),
            dcc.Graph(figure=fig),
            html.Article(text2)
            ]


if __name__ == '__main__':
    app.run_server()
