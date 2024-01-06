# -*- coding: utf-8 -*-
"""
Created on Sat Jan  6 15:18:52 2024

@author: Chalermwong
"""

import dash
import dash_html_components as html
import dash_core_components as dcc
import requests
import datetime as dt
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import pandas as pd
import river
from river.compose import Pipeline, FuncTransformer
from river.linear_model import LinearRegression
from river.preprocessing import StandardScaler 
from river.utils import Rolling
from river.metrics import RMSE
from river import tree

import csv

field_names = ['time','price','predict','error']
               
               
model_name = 'tree'
 
# Model
def get_ordinal_date(x):
    return {'closeTime': x['closeTime'].toordinal()}

# pipeline เป็นกระบวนการจัดการข้อมูลตั้งแต่การเก็บและรวบรวมข้อมูล จนถึงการสร้างโมเดลทำนายและนำโมเดลขึ้น production 
# ทำการ tranform date และ scale ข้อมูล ให้อยู่ใน scale เดียวกัน และ เลือก HoeffdingTreeRegressor ML Model
# สุดท้ายนำ pipeline ที่จัดเตรียมมา ไปใช้ในการ learn และ predict
def create_pipeline():
    pl = Pipeline(
        ('ordinal_date', FuncTransformer(get_ordinal_date)),
        ('scale', StandardScaler()),
        ('tree', tree.HoeffdingTreeRegressor()),
       
    )
    return pl
 
def create_metric():
    return Rolling(RMSE(), 12)
 
def learn_pred(x, y, pl, metric):
    y_pred_old = pl.predict_one(x)
    pl = pl.learn_one(x, y)
    metric = metric.update(y, y_pred_old)
    return x, y, y_pred_old, pl, metric
 
my_pl = create_pipeline()
my_mt = create_metric()
 
coin = 'ETHUSDT'
tick_interval = '1m'
key = f"https://api.binance.com/api/v3/ticker?symbol={coin}"
 
figure = dict(
    data=[{'x': [], 'y': []}], 
    layout=dict(
        xaxis=dict(range=[]), 
        yaxis=dict(range=[])
        )
    )
 
app = dash.Dash(__name__, update_title=None)
 
app.layout = html.Div(
    [
        dcc.Graph(id='graph', figure=figure), 
        dcc.Interval(id="interval", interval=1*1000)
    ]
)
 
dateList = []
priceList = []
predictList = []
 
historical_data = requests.get("https://api.binance.com/api/v3/klines", 
                               params={'symbol': 'ETHUSDT', 'interval': '1m', 'limit': 1000}).json()
 
historical_rows = []
for candle in historical_data:
    row = {
        'openTime': dt.datetime.fromtimestamp(candle[0] / 1000),
        'closeTime': dt.datetime.fromtimestamp(candle[6] / 1000),
        'openPrice': float(candle[1]),
        'highPrice': float(candle[2]),
        'lowPrice': float(candle[3]),
        'closePrice': float(candle[4])
    }
    historical_rows.append(row)
 
for row in historical_rows:
    x_historical = row
    y_historical = row['closePrice']
    del x_historical['openTime']
    del x_historical['closePrice']
    
    _, _, _, my_pl, my_mt = learn_pred(x_historical, y_historical, my_pl, my_mt)
 
@app.callback(
    Output('graph', 'figure'), 
    [Input('interval', 'n_intervals')])
def update_data(n_intervals):
    global my_mt
    global my_pl
 
    print("interval ", n_intervals)
    data = requests.get(key).json()
    price = float(data['lastPrice'])
 
    x = {
        'closeTime': dt.datetime.fromtimestamp(data['closeTime']/1000),
        'openPrice': float(data['openPrice']),
        'highPrice': float(data['highPrice']),
        'lowPrice': float(data['lowPrice'])
    }
    y = float(data['lastPrice'])
    lst_x, lst_y, lst_y_pred, my_pl, my_mt = learn_pred(x, y, my_pl, my_mt)
    # print(f"Pred price {lst_y_pred} \n {my_mt}")

    predicted = lst_y_pred
    closeTime = data['closeTime']
    my_datetime = dt.datetime.fromtimestamp(closeTime / 1000)
 
    dateList.append(my_datetime)
    priceList.append(price)
    predictList.append(predicted)
    
    with open(model_name+'.csv', 'a', newline='') as csv_file:
        dict_object = csv.DictWriter(csv_file, fieldnames=field_names) 
        dict_data = {'time':dt.datetime.fromtimestamp(data['closeTime']/1000),'price': float(price),'predict':float(predicted),'error':my_mt.get()}
        dict_object.writerow(dict_data)
        print (dict_data)
    
    
    df = pd.DataFrame(list(zip(dateList, priceList, predictList)), columns=['datetime', 'price', 'predicted'])
    df = df.iloc[-30:].reset_index(drop=1)
    df = [
        go.Scatter(
            x=df['datetime'],
            y=df['price'],
            name='price',
            mode='lines+markers'
        ),
        go.Scatter(
            x=df['datetime'],
            y=df['predicted'],
            name='predicted',
            mode='lines+markers'
        ),
    ]
 
    return {'data': df } 
if __name__ == '__main__':
    app.run_server()