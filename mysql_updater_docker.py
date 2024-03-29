import re
import json
import time
import os
from gc import collect
from datetime import datetime
from warnings import filterwarnings
import joblib
import pandas as pd
from sqlalchemy import create_engine
from selenium import webdriver
import yfinance as yf

print('All libraries were succesfully imported')

filterwarnings('ignore')

os.system('mkdir problems_db_updater')

current_date = datetime.now().date()

# create engine drivers for mysql

engine = create_engine(
    "mysql://############")
engine2 = create_engine(
    "mysql://############")

# load dict for stocks

with open('stock_dict.json', 'r') as f:
    company_names = json.load(f)

# intializing chromedriver

options = webdriver.ChromeOptions()
options.add_argument('--headless')
options.add_argument('--no-sandbox')
options.add_argument('--disable-dev-shm-usage')
driver = webdriver.Chrome(options=options)

# load latest interest_rate.csv statistics

interest_rate = pd.read_csv(
    'https://fred.stlouisfed.org/graph/fredgraph.csv?id=FEDFUNDS')
interest_rate = interest_rate.set_axis(['Date', 'Discount_rate'], axis=1)
interest_rate['Date'] = pd.to_datetime(interest_rate['Date'])
interest_rate['Discount_rate'] = interest_rate['Discount_rate']/100
interest_rate['Date'] = interest_rate['Date'].dt.to_period('M').dt.end_time
interest_rate['Date'] = interest_rate['Date'].dt.strftime('%Y-%m-%d')
interest_rate['Date'] = pd.to_datetime(interest_rate['Date'])
print('Table interest_rate.csv was successfully downloaded and transform')

# load sp500 statistics

sp500 = pd.read_csv('sp500history.csv')
sp500.drop(columns=['Open', 'High', 'Low', 'Vol.', 'Change %'], inplace=True)
sp500['Date'] = pd.to_datetime(sp500['Date'])
sp500['Date'] = sp500['Date'].dt.to_period('M').dt.end_time
sp500['Date'] = sp500['Date'].dt.strftime('%Y-%m-%d')
sp500['Date'] = pd.to_datetime(sp500['Date'])
print('Table sp500.csv was successfully downloaded and transform')


# get company name from dict

def get_company_name(entered_ticket):
    """return company name from dict"""
    company_name = company_names[entered_ticket]
    return company_name


# re pattern compile, script

pattern = re.compile(r'var originalData = (\[.*?\]);', re.DOTALL)
script = 'return document.querySelector("body > div.main_content_container.container-fluid > script:nth-child(7)").textContent;'


# functions for scrapping

def get_df_from_url(url, webdriver_):
    """return DataFrame from url
    :type arg1: str
    :type arg2: webdriver_ object
    :return: dataframe
    :rtype: pd.DataFrame
    """

    webdriver_.get(url)
    data_raw = webdriver_.execute_script(script)
    data = pattern.search(data_raw).group(1)  # type: ignore
    data = json.loads(data)
    df = pd.DataFrame(data)
    df.drop(columns=['popup_icon'], inplace=True)
    df['field_name'] = df['field_name'].str.extract(r'>([^<]*)</a>$')
    df.dropna(inplace=True)
    return df


def df_for_msql(entered_ticket, webdriver_):
    """ Return DataFrame from URLs, concatenating three DataFrame in one"""

    company_name = get_company_name(entered_ticket)
    urls = {
        'url1': f'https://www.macrotrends.net/stocks/charts/{entered_ticket}/{company_name}/balance-sheet?freq=Q',
        'url2': f'https://www.macrotrends.net/stocks/charts/{entered_ticket}/{company_name}/income-statement?freq=Q',
        'url3': f'https://www.macrotrends.net/stocks/charts/{entered_ticket}/{company_name}/cash-flow-statement?freq=Q'
    }
    df_concat = pd.DataFrame()
    for _, values in urls.items():
        df = get_df_from_url(values, webdriver_)
        df_concat = pd.concat([df_concat, df], axis=0)
    df_concat = df_concat.transpose().reset_index()
    df_concat = df_concat.set_axis(df_concat.iloc[0], axis=1).drop(0)
    df_concat = df_concat.rename(columns={'field_name': 'Date'})
    df_concat['Date'] = pd.to_datetime(df_concat['Date'])
    for col in df_concat.columns:
        df_concat[col] = df_concat[col].apply(lambda x: 0 if x == '' else x)
    if 'Net Change In Property, Plant, And Equipment' in df_concat.columns:
        df_concat['FCF'] = df_concat['Cash Flow From Operating Activities'].astype(float) + \
            df_concat['Net Change In Property, Plant, And Equipment'].astype(
                float)
    else:
        df_concat['FCF'] = df_concat['Cash Flow From Operating Activities'].astype(
            float)

    df = df_concat

    # add statistics from web
    df = df.merge(interest_rate, left_on='Date',
                  right_on='Date', how='left')
    df = df.merge(sp500, left_on='Date',
                  right_on='Date', how='left')
    df.rename(columns={'Price': 'SP500'}, inplace=True)
    df['SP500'] = df['SP500'].str.replace(',', '').astype(float)

    return df


def yfinance_updater(ticket):
    """yfinance_updater"""

    data_ticket = yf.Ticker(ticket)
    df = data_ticket.history(period="max", interval='1mo').reset_index()
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(
        by='Date', ascending=False).reset_index()
    df.drop(columns=['index'], inplace=True)

    return df


print('All functions, dicts and drivers were succesfully launched')

# Updating for SP500 Database

problems_with_scraping = []

for ticket in company_names:
    success = False
    while not success:
        try:
            df_new = df_for_msql(ticket, driver)
            df_new.to_sql(name=f'{ticket}', con=engine,
                          if_exists='replace', index=False)
            print(f'load of {ticket} to SP500 was successful')
            collect()
            success = True
        except Exception:
            time.sleep(600)  # Wait for 10 minutes before retrying
            collect()
            try:
                df_new = df_for_msql(ticket, driver)
                df_new.to_sql(name=f'{ticket}', con=engine,
                              if_exists='replace', index=False)
                print(f'load of {ticket} to SP500 was successful')
                collect()
                success = True
            except Exception as e:
                problems_with_scraping.append(ticket)
                print(f'load of {ticket} failed with error: {e}')
                success = True
                collect()
joblib.dump(problems_with_scraping,
            f'./problems_db_updater/scrapping_problems_{current_date}.txt')

print('Updating for SP500 Databasee was finished succesfully')


# Updating for YFINANCE Datebase

problems_with_yfinance = []

for ticket in company_names:
    success = False
    while not success:
        try:
            stock_df = yfinance_updater(ticket)
            stock_df.to_sql(name=f'{ticket}', con=engine2,
                            if_exists='replace', index=False)
            print(f'load of {ticket} to YFINANCE was successfull')
            collect()
            success = True
        except Exception:
            time.sleep(600)  # Wait for 10 minutes before retrying
            collect()
            try:
                stock_df = yfinance_updater(ticket)
                stock_df.to_sql(name=f'{ticket}', con=engine2,
                                if_exists='replace', index=False)
                print(f'load of {ticket} to YFINANCE was successfull')
                collect()
                success = True
            except Exception as e:
                problems_with_yfinance.append(ticket)
                print(f'load of {ticket} failed with error: {e}')
                success = True
                collect()
joblib.dump(problems_with_yfinance,
            f'./problems_db_updater/yfinance_problems_{current_date}.txt')

print('Updating for YFINANCE Datebase was finished succesfully')


print('All updates were fineshed')
