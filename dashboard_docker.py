import pandas as pd
import dash
from dash import Dash, dcc, html, Input, Output
import plotly.graph_objects as go
import plotly.express as px
from predicter import intrinsic_value_curr
from predicter import intrinsic_value_next
from sqlalchemy import create_engine
import warnings
import json


warnings.filterwarnings("ignore", category=FutureWarning)

engine = create_engine(
    "mysql://python_admin:wofawzfa1984@95.163.243.20:3306/SP500")
engine2 = create_engine(
    "mysql://python_admin:wofawzfa1984@95.163.243.20:3306/YFINANCE")

with open('stock_dict.json', 'r') as f:
    company_names = json.load(f)

data_info = pd.read_csv('INFO.csv', index_col='symbol')
top10 = pd.read_sql_table('TOP10', engine)

options_dropdown = [{'label': f'{key} : {value.capitalize()}', 'value': key}
                    for key, value in company_names.items()]


list_of_years = [2009, 2010, 2011, 2012, 2013, 2014,
                 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]
years_keys = {}

for key in range(15):
    years_keys.update({key: list_of_years[key]})

options_slider = [{'label': value, 'value': key}
                  for key, value in years_keys.items()]

# Блок дашборда

app = Dash(__name__)

app.config.suppress_callback_exceptions = True

app.layout = html.Div(children=[html.H2('Dashboard: market price and evaluated price of stocks',
                                        style={'textAlign': 'center',
                                               'font-family': 'Arial'}),
                                html.Div([
                                    html.Div(
                                        [
                                            html.H3('Choose company:',
                                                    style={
                                                        'font-family': 'Arial',
                                                        'margin-left': '20px'}
                                                    ),
                                        ]
                                    ),
                                    html.Div([
                                        dcc.Dropdown(
                                            id='input-company',
                                            options=options_dropdown,
                                            style={'width': 250,
                                                   'margin-left': '10px',
                                                   'font-family': 'Arial'},
                                            value='',
                                        )
                                    ])
                                ]),
                                html.Br(),
                                html.Div([], id='article', style={
                                         'margin-left': '20px',
                                         'font-family': 'Arial'}),
                                html.Div([
                                    html.Div(
                                        [
                                            html.H3('Clarify year series:',
                                                    style={
                                                        'font-family': 'Arial',
                                                        'margin-left': '20px'}
                                                    ),
                                        ]
                                    ),
                                    html.Div([
                                        dcc.RangeSlider(
                                            id='date-slider',
                                            min=0,
                                            max=14,
                                            step=None,
                                            marks=options_slider,
                                            value=[0, 14]
                                        )
                                    ])
                                ]),
                                html.Br(),
                                html.Div([
                                    html.Button(children='Make a prediction, this may take several minutes',
                                                id='submit-val', n_clicks=0,
                                                style={
                                                    'font-family': 'Arial',
                                                    'background-color': 'light gray',
                                                    'color': 'dark gray',
                                                    'border-radius': '5px',
                                                    'transition-duration': '0.4s',
                                                    'margin-left': '20px'
                                                }
                                                )]),
                                html.Br(),
                                html.Div([], id='plot'),
                                html.Br(),
                                html.Div([], id='plot2'),
                                ])

# Callback для ввода данных в нашу функцию и возврат графика в дашбоард

df = None
data_stock = None


@app.callback(
    [
        Output(component_id='article', component_property='children'),
        Output(component_id='plot', component_property='children'),
        Output(component_id='plot2', component_property='children'),
    ],
    [
        Input(component_id='input-company', component_property='value'),
        Input(component_id='submit-val', component_property='n_clicks'),
        Input(component_id='date-slider', component_property='value')
    ]
)
def update_output_div(input_value, submit_val, date_range):
    global df
    global data_stock

    ctx = dash.callback_context
    if not ctx.triggered:
        button_id = 'No clicks yet'
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if input_value is None or input_value == '' or input_value not in company_names:
        df = None
        data_stock = None
        return '', '', ''
    else:
        df = pd.read_sql_table(input_value, engine)
        data_stock = pd.read_sql_table(input_value, engine2)

    if button_id == 'submit-val':
        df_prediction = intrinsic_value_next(df)
        return get_info(data_info, input_value), \
            get_graph(data_stock, df, date_range, df_prediction, input_value), \
            get_top_graph()

    else:
        df_prediction = None
        return get_info(data_info, input_value), \
            get_graph(data_stock, df, date_range, df_prediction, input_value), \
            get_top_graph()


def get_info(data_info, input_value):
    text = html.Div([
        html.Div(['Name: ', data_info.loc[input_value, 'name']]),
        html.Div(['Address: ', f'{data_info.loc[input_value, "address1"]}, \
                  {data_info.loc[input_value, "address2"]}']),
        html.Div(['Industry: ', data_info.loc[input_value, 'industry']]),
        # html.Div(['Website: ', html.A(data_info.loc[input_value,
        #          'website'], href=data_info.loc[input_value, 'website'])]),
        html.Br(),
        html.Div(['Bussiness Summary:', html.P(
            data_info.loc[input_value, 'summary'])])
    ])
    return html.Article(text)


def get_graph(data_stock, df, date_range, df_prediction, input_value):
    start_date = pd.Timestamp('2009') + \
        pd.DateOffset(years=date_range[0])
    end_date = pd.Timestamp('2010') + pd.DateOffset(years=date_range[1])
    stock_df = data_stock.loc[(data_stock['Date'] >= start_date) & (
        data_stock['Date'] <= end_date)]
    stock_df.reset_index(inplace=True)
    close_price = stock_df[['Date', 'Close']]
    close_price.columns = ['Date', 'Close_price']
    df = intrinsic_value_curr(df)
    iv_price = df[['Date', 'Intrinsic Value']]
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
                         name='Intrinsic value')
                  )

    fig2 = px.scatter(filtered_iv_price, x='Date', y='Intrinsic Value',
                      trendline='expanding', title='Intrinsic value mean')
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
                             name='Max intrinsic value'))

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
    if df_prediction is not None:
        next_intrinsic_value = df_prediction.loc[0, 'Intrinsic value']
        last_close = data_stock.loc[0, 'Close']
        difference = next_intrinsic_value-last_close
        recomendations = f"Future intrinsic value evaluated using RNN neural network.<br> \
            The difference between intrinsic value {next_intrinsic_value:.02f}$ and last close price {last_close:.02f}$ is {difference:.02f}$, <br>"
        if last_close > next_intrinsic_value*1.5:
            recomendations += f"so you shoudn't buy or do it with some precautions"
        elif last_close < next_intrinsic_value*1.5 and last_close >= next_intrinsic_value:
            recomendations += f"so you could buy it, but don't forget about diversification"
        elif last_close < next_intrinsic_value*0.9:
            recomendations += f"so you should definitely buy it"
        fig.add_trace(go.Bar(x=df_prediction['Date'],
                             y=df_prediction['Intrinsic value'],
                             name='Intrinsic value next quarter by LSTM Model',
                             marker_color='green',
                             ))
        fig.add_annotation(text=recomendations,
                           xref="paper",
                           yref="paper",
                           x=1,
                           y=1,
                           showarrow=False,
                           bordercolor='green',
                           borderwidth=1,
                           bgcolor='white',
                           align='center'
                           )
    fig.update_layout(
        title_text=f'Difference between intrinsic value and close market price for {company_names[input_value].capitalize()}')
    fig.update_layout(legend_orientation="h")
    fig.update_yaxes(tickprefix="$", showgrid=True)

    return dcc.Graph(figure=fig)


def get_top_graph():
    symbol_to_company = top10['symbol'].apply(
        lambda x: f'{x}: {company_names[x]}')
    fig = go.Figure()
    fig.add_trace(go.Bar(x=symbol_to_company,
                         y=top10['Close price'],
                         name='Close market price'))
    fig.add_trace(go.Bar(x=symbol_to_company,
                         y=top10['Intrinsic Value'],
                         name='Intrinsic value'))
    fig.update_layout(barmode='stack')

    fig.update_layout(
        title_text='Top 10 stocks based on difference between intrinsic value and close market price')
    fig.update_yaxes(tickprefix="$", showgrid=True)

    return dcc.Graph(figure=fig)


if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port=3000)
