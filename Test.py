import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
from bs4 import BeautifulSoup

# ticket=input()
# url1='https://www.marketwatch.com/investing/stock/'+ticket+'/financials?mod=mw_quote_tab'
# url2='https://www.marketwatch.com/investing/stock/'+ticket+'/financials/cash-flow'
# #print(url)
# data1=pd.read_html(url1)[4]
# data2=pd.read_html(url2)[5]
# net_income=data1.iloc[47,5]
# depreciation=data1.iloc[6,5]
# capital_expenditure=data2.iloc[0,5]
# # Owners_earnings=net_income+depreciation-capital_expenditure
# print('net income:',net_income, '\n'
#       'capital expenditure:', capital_expenditure,'\n'
#       'depriciation:',depreciation)
# print(data1.head())

# cell_df = cell_df[pd.to_numeric(cell_df['BareNuc'], errors='coerce').notnull()]

# data_clean=data.iloc[6,5]
# data= requests.get(url).text
# soup = BeautifulSoup(data, 'html5lib')
# netflix_data = pd.DataFrame(columns=["Date", "Open", "High", "Low", "Close", "Volume"])
#
# # First we isolate the body of the table which contains all the information
# # Then we loop through each row and find all the column values for each row
# for row in soup.find("tbody").find_all('tr'):
#     col = row.find_all("td")
#     date = col[0].text
#
#     # Finally we append the data of each row to the table
#     netflix_data = netflix_data._append(
#         {"Date": date}, ignore_index=True)
#

