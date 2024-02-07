# dashboard_for_stocks

The aim of the project is to calculate the Intrinsic Value of stocks, which is used in financial statistics. Additionally, the project predicts this indicator on a time series as a trend, highlights the most profitable securities from the point of view of this metric, and displays the requested information about the security through a GUI (Dashboard).
The project was written in Python, and the Dash and Plotly libraries were used for the dashboard.
Web scraping of financial data for each company is carried out using selenium webdriver and chromedriver, and data is loaded/unloaded into the MySQL database on a schedule via Cron.
Under the hood: calculation of financial statistics using Pandas/Numpy, forecasting of data on a time series using NN - LSTM from the Keras package, validation on MAE. Deployment using Docker compose with a web server on nginx.
The project is running on a live server, the database is updated regularly, and the model is being validated.
