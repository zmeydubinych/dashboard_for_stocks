import json
import time
import os
from gc import collect
from datetime import datetime
from warnings import filterwarnings
import joblib
import pandas as pd
from sqlalchemy import create_engine
from predicter import intrinsic_value_curr

print('All libraries were succesfully imported')

filterwarnings('ignore')

os.system('mkdir problems_db_updater')

current_date = datetime.now().date()

# create engine drivers for mysql

engine = create_engine(
    "mysql://################")
engine2 = create_engine(
    "mysql://################")

# load dict for stocks

with open('stock_dict.json', 'r') as f:
    company_names = json.load(f)


#TOP10 dunction

def top10_creator(ticket):
    """Create top10 functions from MySQL DB"""

    data_stock = pd.read_sql_table(ticket, engine2)
    df_iv_price = intrinsic_value_curr(
        pd.read_sql_table(ticket, engine), data_stock)
    data_stock = data_stock.loc[data_stock['Date']
                                <= df_iv_price.loc[0, 'Date']].reset_index()
    last_close_price = data_stock.loc[0, 'Close']
    last_iv_value = df_iv_price.loc[0, 'Intrinsic Value']
    diff_price = df_iv_price.loc[0, 'Intrinsic Value'].astype(  # type: ignore
        float) - data_stock.loc[0, 'Close'].astype(float)  # type: ignore
    df_compare = pd.DataFrame.from_dict({'symbol': ticket, 'Date': df_iv_price.loc[0, 'Date'],
                                            'Close price': last_close_price, 'Intrinsic Value': last_iv_value,
                                            'Difference': diff_price}, orient='index').transpose()
    return df_compare


# Creating TOP10 table for YFINANCE Datebase
problems_with_top10 = []
df_ = pd.DataFrame()
for ticket in company_names:
    success = False
    while not success:
        try:
            df_compare=top10_creator(ticket)
            df_ = pd.concat([df_, df_compare], axis=0)
            print(f'Add ticket {ticket} while creating top10 table was succesfull')
            success=True
            collect()
        except Exception:
            time.sleep(600)  # Wait for 10 minutes before retrying
            collect()
            try:
                df_compare=top10_creator(ticket)
                df_ = pd.concat([df_, df_compare], axis=0)
                print(f'Add ticket {ticket} while creating top10 table was succesfull')
                success=True
                collect()
            except Exception as e:
                problems_with_top10.append(ticket)
                print(f'Add ticket {ticket} while creating top10 failed with error {e}')
                success=True
                collect()

joblib.dump(problems_with_top10,
            f'./problems_db_updater/top10_problems_{current_date}.txt')

#Sort values, transform index
top10 = df_.sort_values(by='Difference', ascending=False).head(10)
top10['symbol'].apply(lambda x: f'{x}: {company_names[x]}')

#upload predefined table to DB
top10.to_sql(name='TOP10', con=engine, if_exists='replace', index=False)


print('Proccess of downloading and uploadeding for top10 table to DataBase successfully finished')
