import pandas as pd
import dash
from dash import dcc, html, Input, Output, callback
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf
from scrapping import df_new


# Блок дашборда

app = dash.Dash(__name__)

app.config.suppress_callback_exceptions = True

app.layout = html.Div(children=[html.H1('Dashboard: market price and evaluated price of stocks',
                                        style={'textAlign': 'center',
                                               'color': '#480607',
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
                                        dcc.RangeSlider(
                                            id='date-slider',
                                            min=0,
                                            max=10,
                                            step=None,
                                            marks={
                                                0: '2013-12-31',
                                                1: '2015-12-31',
                                                2: '2016-12-31',
                                                3: '2017-12-31',
                                                4: '2018-12-31',
                                                5: '2019-12-31',
                                                6: '2020-12-31',
                                                7: '2021-12-31',
                                                8: '2022-12-31',
                                                9: '2023-12-31',
                                                10: '2024-12-31',
                                            },
                                            value=[0, 10]
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
        Input(component_id='date-slider', component_property='value')
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
    # переписать, чтобы учитывалась вся граница доступного интервала
    start_date = pd.Timestamp('2013-12-31') + \
        pd.DateOffset(years=date_range[0])
    end_date = pd.Timestamp('2013-12-31') + pd.DateOffset(years=date_range[1])
    data_stock = yf.Ticker(entered_ticket)
    info = data_stock.info
    stock_df = data_stock.history(
        start=start_date, end=end_date, interval='1mo')
    stock_df.reset_index(inplace=True)
    close_price = stock_df[['Date', 'Close']]
    close_price.columns = ['Date', 'Close_price']
    iv_price = df_new(entered_ticket)[
        ['Date', 'Intrinsic Value']]
    filtered_iv_price = iv_price.loc[(iv_price['Date'] >= start_date) & (
        iv_price['Date'] <= end_date)]

    max_close_index = close_price['Close_price'].idxmax()
    max_close = close_price['Close_price'].max()
    max_iv_index = filtered_iv_price['Intrinsic Value'].idxmax()
    max_iv = filtered_iv_price['Intrinsic Value'].max()

    fig = go.Figure()

    fig.update_layout(
        autosize=True,
        height=1000
    )

    fig.add_trace(go.Scatter(x=close_price['Date'],
                             y=close_price['Close_price'],
                             mode='lines',
                             line=dict(color='black'),
                             name='Close market price')
                  )
    fig.add_trace(go.Bar(x=filtered_iv_price['Date'],
                         y=filtered_iv_price['Intrinsic Value'],
                         name='Intrinsic Value')
                  )

    fig2 = px.scatter(filtered_iv_price, x='Date', y='Intrinsic Value',
                      trendline='expanding', title='Intrinsic Value Mean')
    trendline = fig2.data[1]
    fig.add_trace(trendline)

    fig.add_trace(go.Scatter(x=[close_price['Date'][max_close_index]],
                             y=[max_close],
                             mode='markers',
                             marker=dict(color='red', size=10),
                             name='Max close price')
                  )
    fig.add_trace(go.Scatter(x=[filtered_iv_price['Date'][max_iv_index]],
                             y=[max_iv],
                             mode='markers',
                             marker=dict(color='green', size=10),
                             name='Max Intrinsic Value'))

    fig.add_annotation(x=close_price['Date'][max_close_index],
                       y=max_close,
                       text=f'{max_close:.2f} $',
                       showarrow=True
                       )
    fig.add_annotation(x=filtered_iv_price['Date'][max_iv_index],
                       y=max_iv,
                       text=f'{max_iv:.2f} $',
                       showarrow=True
                       )

    fig.update_layout(legend_orientation="h")
    fig.update_yaxes(tickprefix="$", showgrid=True)

    # def get_block(label, value):
    #     return html.Div([label, value])

    text = html.Div([
        html.Div(['Name: ', info['longName']]),
        html.Div(['Country: ', info['country']]),
        html.Div(['City: ', info['city']]),
        html.Div(['ZIP: ', info['zip']]),
        html.Div(['Address: ', info['address1']]),
        html.Div(['Industry: ', info['industry']]),
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
