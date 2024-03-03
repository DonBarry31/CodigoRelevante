import yfinance as yf
import pandas as pd
import requests
from bs4 import BeautifulSoup
import plotly.graph_objects as go
from plotly.subplots import make_subplots

tesla = yf.Ticker("TSLA")
tesla_data = pd.DataFrame(tesla.history(period = 'max')).reset_index()
#print(tesla_data.head())


url='https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-PY0220EN-SkillsNetwork/labs/project/revenue.htm'
data_tesla_revenue = pd.read_html(url,match='Tesla Annual Revenue')
df = data_tesla_revenue[0]
df = df.rename(columns={'Tesla Annual Revenue (Millions of US $)':'Date', 'Tesla Annual Revenue (Millions of US $).1':'Revenue'})
df['Revenue'] = df['Revenue'].str.replace(r'\W', '', regex=True)
df.dropna(inplace=True)
df = df[df['Revenue'] != ""]
df['Date'] = pd.to_datetime(df['Date'].astype(str), format='%Y')
#print(df)


gameS = yf.Ticker("GME")
gameS_data = pd.DataFrame(gameS.history(period = 'max')).reset_index()
#print(tesla_data.head())


url1 = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-PY0220EN-SkillsNetwork/labs/project/stock.html'
data_gameS_revenue = pd.read_html(url1, match='GameStop Annual Revenue')
df1 = data_gameS_revenue[0]
df1 = df1.rename(columns={'GameStop Annual Revenue (Millions of US $)':'Date', 'GameStop Annual Revenue (Millions of US $).1':'Revenue'})
df1['Revenue'] = df1['Revenue'].str.replace(r'\W', '', regex=True)
df1.dropna(inplace=True)
df1= df1[df1['Revenue'] != ""]
df1['Date'] = pd.to_datetime(df1['Date'].astype(str), format='%Y')
#print(df1.tail())


def make_graph(stock_data, revenue_data, stock):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=("Historical Share Price", "Historical Revenue"), vertical_spacing = .3)
    stock_data_specific = stock_data[stock_data.Date <= '2021--06-14']
    revenue_data_specific = revenue_data[revenue_data.Date <= '2021-04-30']
    fig.add_trace(go.Scatter(x=pd.to_datetime(stock_data_specific.Date, infer_datetime_format=True), y=stock_data_specific.Close.astype("float"), name="Share Price"), row=1, col=1)
    fig.add_trace(go.Scatter(x=pd.to_datetime(revenue_data_specific.Date, infer_datetime_format=True), y=revenue_data_specific.Revenue.astype("float"), name="Revenue"), row=2, col=1)
    fig.update_xaxes(title_text="Date", row=1, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Price ($US)", row=1, col=1)
    fig.update_yaxes(title_text="Revenue ($US Millions)", row=2, col=1)
    fig.update_layout(showlegend=False,
    height=900,
    title=stock,
    xaxis_rangeslider_visible=True)
    fig.show()


#plot tesla stock


#make_graph(tesla_data, df, 'Tesla')


#plot gamestop stock


make_graph(gameS_data, df1, 'GameStop')
