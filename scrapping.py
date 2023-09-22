import pandas as pd
import yfinance as yf
import json
from selenium import webdriver
import re
from predicter import intrinsic_value

company_names = {'AAPL': 'apple',
                 'NVDA': 'nvidia',
                 'AMD': 'amd',
                 'INTC': 'intel',
                 'KO': 'cocacola',
                 'TSLA': 'tesla',
                 'PFE': 'pfizer'
                 }


def get_df_from_url(url, driver):
    driver.get(url)
    script = 'return document.querySelector("body > div.main_content_container.container-fluid > script:nth-child(7)").textContent;'
    pattern = re.compile(r'var originalData = (\[.*?\]);', re.DOTALL)
    data_raw = driver.execute_script(script)
    data = pattern.search(data_raw).group(1)
    data = json.loads(data)
    df = pd.DataFrame(data)
    df.drop(columns=['popup_icon'], inplace=True)
    df['field_name'] = df['field_name'].str.extract(r'>([^<]*)</a>$')
    df.dropna(inplace=True)
    return df


def df_new(entered_ticket):
    company_name=company_names[entered_ticket]
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    driver = webdriver.Chrome(options=options)
    urls = {
        'url1': f'https://www.macrotrends.net/stocks/charts/{entered_ticket}/{company_name}/balance-sheet?freq=Q',
        'url2': f'https://www.macrotrends.net/stocks/charts/{entered_ticket}/{company_name}/income-statement?freq=Q',
        'url3': f'https://www.macrotrends.net/stocks/charts/{entered_ticket}/{company_name}/cash-flow-statement?freq=Q'
    }
    df_concat = pd.DataFrame()
    for url in urls:
        df = get_df_from_url(urls[url], driver)
        df_concat = pd.concat([df_concat, df], axis=0)
    df_concat = df_concat.transpose().reset_index()
    df_concat = df_concat.set_axis(df_concat.iloc[0], axis=1).drop(0)
    df_concat = df_concat.rename(columns={'field_name': 'Date'})
    df_concat['Date'] = pd.to_datetime(df_concat['Date'])
    df_concat['FCF'] = df_concat['Cash Flow From Operating Activities'].astype(
        float)+df_concat['Net Change In Property, Plant, And Equipment'].astype(float)
    df = df_concat
    return intrinsic_value(df)

# df=df_new('KO')
# print(df)