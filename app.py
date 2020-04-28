from flask import Flask, render_template
import json
import plotly
import requests
import requests
import pickle
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import gzip


app = Flask(__name__)
app.debug = True


@app.route('/')
def index():
    headers = {'accept': 'application/json','apikey': 'TfvwMAIRVIF7NkNuB7RAEJIEJA437sdM'}
    response1 = requests.get('https://api.climacell.co/v3/weather/forecast/hourly?lat=35&lon=-97&unit_system=si&fields=temp%3AF&start_time=now', headers=headers)
    response2 = requests.get('https://api.climacell.co/v3/weather/forecast/hourly?lat=35&lon=-97&unit_system=us&fields=humidity&start_time=now', headers=headers)
    response3 = requests.get('https://api.climacell.co/v3/weather/forecast/hourly?lat=35&lon=-97&unit_system=us&fields=wind_speed%3Amph&start_time=now', headers=headers)
    response4 = requests.get('https://api.climacell.co/v3/weather/forecast/hourly?lat=35&lon=-97&unit_system=us&fields=wind_direction%3Adegrees&start_time=now', headers=headers)
    response5 = requests.get('https://api.climacell.co/v3/weather/forecast/hourly?lat=35&lon=-97&unit_system=us&fields=baro_pressure%3AhPa&start_time=now', headers=headers)
    response6 = requests.get('https://api.climacell.co/v3/weather/forecast/hourly?lat=35&lon=-97&unit_system=us&fields=precipitation_type&start_time=now', headers=headers)
    response7 = requests.get('https://api.climacell.co/v3/weather/forecast/hourly?lat=35&lon=-97&unit_system=us&fields=precipitation%3Ain%2Fhr&start_time=now', headers=headers)
    response8 = requests.get('https://api.climacell.co/v3/weather/forecast/hourly?lat=35&lon=-97&unit_system=us&fields=precipitation_probability%3A%25&start_time=now', headers=headers)
    R1 = response1.content
    R2 = response2.content
    R3 = response3.content
    R4 = response4.content
    R5 = response5.content
    R6 = response6.content
    R7 = response7.content
    R8 = response8.content
    R1 = json.loads(R1)
    R2 = json.loads(R2)
    R3 = json.loads(R3)
    R4 = json.loads(R4)
    R5 = json.loads(R5)
    R6 = json.loads(R6)
    R7 = json.loads(R7)
    R8 = json.loads(R8)
    date = [(R1[0]['observation_time']['value']),(R1[1]['observation_time']['value']),(R1[2]['observation_time']['value']),(R1[3]['observation_time']['value']),(R1[4]['observation_time']['value']),(R1[5]['observation_time']['value']),(R1[6]['observation_time']['value']),(R1[7]['observation_time']['value']),(R1[8]['observation_time']['value']),(R1[9]['observation_time']['value']),(R1[10]['observation_time']['value']),(R1[11]['observation_time']['value']),(R1[12]['observation_time']['value']),(R1[13]['observation_time']['value']),(R1[14]['observation_time']['value']),(R1[15]['observation_time']['value']),(R1[16]['observation_time']['value']),(R1[17]['observation_time']['value']),(R1[18]['observation_time']['value']),(R1[19]['observation_time']['value']),(R1[20]['observation_time']['value']),(R1[21]['observation_time']['value']),(R1[22]['observation_time']['value']),(R1[23]['observation_time']['value']),(R1[24]['observation_time']['value'])]
    Temp = [(R1[0]['temp']['value']),(R1[1]['temp']['value']),(R1[2]['temp']['value']),(R1[3]['temp']['value']),(R1[4]['temp']['value']),(R1[5]['temp']['value']),(R1[6]['temp']['value']),(R1[7]['temp']['value']),(R1[8]['temp']['value']),(R1[9]['temp']['value']),(R1[10]['temp']['value']),(R1[11]['temp']['value']),(R1[12]['temp']['value']),(R1[13]['temp']['value']),(R1[14]['temp']['value']),(R1[15]['temp']['value']),(R1[16]['temp']['value']),(R1[17]['temp']['value']),(R1[18]['temp']['value']),(R1[19]['temp']['value']),(R1[20]['temp']['value']),(R1[21]['temp']['value']),(R1[22]['temp']['value']),(R1[23]['temp']['value']),(R1[24]['temp']['value'])]
    hum = [(R2[0]['humidity']['value']),(R2[1]['humidity']['value']),(R2[2]['humidity']['value']),(R2[3]['humidity']['value']),(R2[4]['humidity']['value']),(R2[5]['humidity']['value']),
        (R2[6]['humidity']['value']),(R2[7]['humidity']['value']),(R2[8]['humidity']['value']),(R2[9]['humidity']['value']),(R2[10]['humidity']['value']),(R2[11]['humidity']['value']),(R2[12]['humidity']['value']),(R2[13]['humidity']['value']),
        (R2[14]['humidity']['value']),(R2[15]['humidity']['value']),(R2[16]['humidity']['value']),(R2[17]['humidity']['value']),(R2[18]['humidity']['value']),(R2[19]['humidity']['value']),(R2[20]['humidity']['value']),(R2[21]['humidity']['value']),
        (R2[22]['humidity']['value']),(R2[23]['humidity']['value']),(R2[24]['humidity']['value'])]
    Wind_S = [(R3[0]['wind_speed']['value']),(R3[1]['wind_speed']['value']),(R3[2]['wind_speed']['value']),(R3[3]['wind_speed']['value']),(R3[4]['wind_speed']['value']),(R3[5]['wind_speed']['value']),
        (R3[6]['wind_speed']['value']),(R3[7]['wind_speed']['value']),(R3[8]['wind_speed']['value']),(R3[9]['wind_speed']['value']),(R3[10]['wind_speed']['value']),(R3[11]['wind_speed']['value']),(R3[12]['wind_speed']['value']),(R3[13]['wind_speed']['value']),
        (R3[14]['wind_speed']['value']),(R3[15]['wind_speed']['value']),(R3[16]['wind_speed']['value']),(R3[17]['wind_speed']['value']),(R3[18]['wind_speed']['value']),(R3[19]['wind_speed']['value']),(R3[20]['wind_speed']['value']),(R3[21]['wind_speed']['value']),
        (R3[22]['wind_speed']['value']),(R3[23]['wind_speed']['value']),(R3[24]['wind_speed']['value'])]
    Wind_D = [(R4[0]['wind_direction']['value']),(R4[1]['wind_direction']['value']),(R4[2]['wind_direction']['value']),(R4[3]['wind_direction']['value']),(R4[4]['wind_direction']['value']),(R4[5]['wind_direction']['value']),
        (R4[6]['wind_direction']['value']),(R4[7]['wind_direction']['value']),(R4[8]['wind_direction']['value']),(R4[9]['wind_direction']['value']),(R4[10]['wind_direction']['value']),(R4[11]['wind_direction']['value']),(R4[12]['wind_direction']['value']),(R4[13]['wind_direction']['value']),
        (R4[14]['wind_direction']['value']),(R4[15]['wind_direction']['value']),(R4[16]['wind_direction']['value']),(R4[17]['wind_direction']['value']),(R4[18]['wind_direction']['value']),(R4[19]['wind_direction']['value']),(R4[20]['wind_direction']['value']),(R4[21]['wind_direction']['value']),
        (R4[22]['wind_direction']['value']),(R4[23]['wind_direction']['value']),(R4[24]['wind_direction']['value'])]
    Press = [(R5[0]['baro_pressure']['value']),(R5[1]['baro_pressure']['value']),(R5[2]['baro_pressure']['value']),(R5[3]['baro_pressure']['value']),(R5[4]['baro_pressure']['value']),(R5[5]['baro_pressure']['value']),
        (R5[6]['baro_pressure']['value']),(R5[7]['baro_pressure']['value']),(R5[8]['baro_pressure']['value']),(R5[9]['baro_pressure']['value']),(R5[10]['baro_pressure']['value']),(R5[11]['baro_pressure']['value']),(R5[12]['baro_pressure']['value']),(R5[13]['baro_pressure']['value']),
        (R5[14]['baro_pressure']['value']),(R5[15]['baro_pressure']['value']),(R5[16]['baro_pressure']['value']),(R5[17]['baro_pressure']['value']),(R5[18]['baro_pressure']['value']),(R5[19]['baro_pressure']['value']),(R5[20]['baro_pressure']['value']),(R5[21]['baro_pressure']['value']),
        (R5[22]['baro_pressure']['value']),(R5[23]['baro_pressure']['value']),(R5[24]['baro_pressure']['value'])]
    Rain_T = [(R6[0]['precipitation_type']['value']),(R6[1]['precipitation_type']['value']),(R6[2]['precipitation_type']['value']),(R6[3]['precipitation_type']['value']),(R6[4]['precipitation_type']['value']),(R6[5]['precipitation_type']['value']),
        (R6[6]['precipitation_type']['value']),(R6[7]['precipitation_type']['value']),(R6[8]['precipitation_type']['value']),(R6[9]['precipitation_type']['value']),(R6[10]['precipitation_type']['value']),(R6[11]['precipitation_type']['value']),(R6[12]['precipitation_type']['value']),(R6[13]['precipitation_type']['value']),
        (R6[14]['precipitation_type']['value']),(R6[15]['precipitation_type']['value']),(R6[16]['precipitation_type']['value']),(R6[17]['precipitation_type']['value']),(R6[18]['precipitation_type']['value']),(R6[19]['precipitation_type']['value']),(R6[20]['precipitation_type']['value']),(R6[21]['precipitation_type']['value']),
        (R6[22]['precipitation_type']['value']),(R6[23]['precipitation_type']['value']),(R6[24]['precipitation_type']['value'])]
    Rain_I = [(R7[0]['precipitation']['value']),(R7[1]['precipitation']['value']),(R7[2]['precipitation']['value']),(R7[3]['precipitation']['value']),(R7[4]['precipitation']['value']),(R7[5]['precipitation']['value']),
        (R7[6]['precipitation']['value']),(R7[7]['precipitation']['value']),(R7[8]['precipitation']['value']),(R7[9]['precipitation']['value']),(R7[10]['precipitation']['value']),(R7[11]['precipitation']['value']),(R7[12]['precipitation']['value']),(R7[13]['precipitation']['value']),
        (R7[14]['precipitation']['value']),(R7[15]['precipitation']['value']),(R7[16]['precipitation']['value']),(R7[17]['precipitation']['value']),(R7[18]['precipitation']['value']),(R7[19]['precipitation']['value']),(R7[20]['precipitation']['value']),(R7[21]['precipitation']['value']),
        (R7[22]['precipitation']['value']),(R7[23]['precipitation']['value']),(R7[24]['precipitation']['value'])]
    Rain_E = [(R8[0]['precipitation_probability']['value']),(R8[1]['precipitation_probability']['value']),(R8[2]['precipitation_probability']['value']),(R8[3]['precipitation_probability']['value']),(R8[4]['precipitation_probability']['value']),(R8[5]['precipitation_probability']['value']),
        (R8[6]['precipitation_probability']['value']),(R8[7]['precipitation_probability']['value']),(R8[8]['precipitation_probability']['value']),(R8[9]['precipitation_probability']['value']),(R8[10]['precipitation_probability']['value']),(R8[11]['precipitation_probability']['value']),(R8[12]['precipitation_probability']['value']),(R8[13]['precipitation_probability']['value']),
        (R8[14]['precipitation_probability']['value']),(R8[15]['precipitation_probability']['value']),(R8[16]['precipitation_probability']['value']),(R8[17]['precipitation_probability']['value']),(R8[18]['precipitation_probability']['value']),(R8[19]['precipitation_probability']['value']),(R8[20]['precipitation_probability']['value']),(R8[21]['precipitation_probability']['value']),
        (R8[22]['precipitation_probability']['value']),(R8[23]['precipitation_probability']['value']),(R8[24]['precipitation_probability']['value'])]

    X_test = pd.DataFrame({'Date_Time': date,'Humidity': hum,'Wind speed': Wind_S, 'Wind Direction':Wind_D,'Air Pressure':Press,'Rain intensity':Rain_T,'Rain Type':Rain_I,'Rain event':Rain_E,'temperature':Temp}, columns=['Date_Time','temperature','Humidity','Wind speed',
                                                                                                                         'Wind Direction','Air Pressure','Rain Type','Rain intensity','Rain event'])
    X_test['Rain intensity'] = X_test['Rain intensity'].apply({'rain':1, 'none':0}.get)
    X_test['Rain event'] = np.where(X_test['Rain event'] <= 15, 0, X_test['Rain event'])
    X_test['Rain event'] = np.where(X_test['Rain event'] >= 15, 1, X_test['Rain event'])
    from datetime import datetime
    X_test['Date_Time'] = pd.to_datetime(X_test.Date_Time)
    X_test['Date_Time'] = X_test['Date_Time'].dt.tz_convert('US/Central')
    X_test['Date_Time'] = X_test['Date_Time'].dt.strftime('%m-%d-%Y %H:%M:%S')
    X_test = X_test.set_index('Date_Time')
    
    #model1 = pickle.load(open('model_RF.pkl', 'rb'))
    f = gzip.open('model_RF.pklz','rb')
    model1 = pickle.load(f)
    f.close()
    model2 = load_model('model_DNN.h5')
    model3 =load_model('LSTM_MLDeployment.h5')
    scaler = pickle.load(open('scaler.pkl', 'rb'))

    prediction = model1.predict(X_test)
    prediction = pd.DataFrame(prediction)
    prediction = prediction.set_index(X_test.index) 
    prediction = prediction.rename(columns = {0:"predicted temp"}) 
    prediction1 = model2.predict(X_test)
    prediction1 = pd.DataFrame(prediction1)
    prediction1 = prediction1.set_index(X_test.index) 
    prediction1 = prediction1.rename(columns = {0:"predicted temp"})
   
    ####New Code####
    
    headers = {'accept': 'application/json','apikey': 'TfvwMAIRVIF7NkNuB7RAEJIEJA437sdM'}
    response1 = requests.get('https://api.climacell.co/v3/weather/forecast/hourly?lat=35&lon=-97&unit_system=si&fields=temp%3AF&start_time=now', headers=headers)
    RR1 = response1.content
    RR1 = json.loads(RR1)
    m= len(RR1)
    date = [] 
    Temp = []
    for x in range(m):
        date.append(RR1[x]['observation_time']['value'])
        Temp.append(RR1[x]['temp']['value'])
    
    Online_df_24 = pd.DataFrame({'Date_Time': date,'temperature':Temp}, columns=['Date_Time','temperature'])
    from datetime import datetime, timedelta
    Online_df_24['Date_Time'] = pd.to_datetime(Online_df_24.Date_Time)
    Online_df_24['Date_Time'] = Online_df_24['Date_Time'].dt.tz_convert('US/Central')
    Online_df_24['Date_Time'] = Online_df_24['Date_Time'].dt.strftime('%m-%d-%Y %H:%M:%S')
    Online_df_24['Date_Time'] = pd.to_datetime(Online_df_24.Date_Time)
    Online_df_24 = Online_df_24.set_index('Date_Time')
 
    headers = """Host: rwis.tulsa.ou.edu
User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:76.0) Gecko/20100101 Firefox/76.0
Accept: application/json, text/javascript, */*; q=0.01
Accept-Language: en-US,en;q=0.5
Accept-Encoding: gzip, deflate, br
Content-Type: application/x-www-form-urlencoded; charset=UTF-8
X-Requested-With: XMLHttpRequest
Content-Length: 29
Origin: https://rwis.tulsa.ou.edu
Connection: keep-alive
Referer: https://rwis.tulsa.ou.edu/rwis/report/showReport/32
Cookie: ci_session=a%3A5%3A%7Bs%3A10%3A%22session_id%22%3Bs%3A32%3A%22d0d9ac66e8928495ce09cc2ee64eb79e%22%3Bs%3A10%3A%22ip_address%22%3Bs%3A7%3A%220.0.0.0%22%3Bs%3A10%3A%22user_agent%22%3Bs%3A78%3A%22Mozilla%2F5.0+%28Windows+NT+10.0%3B+Win64%3B+x64%3B+rv%3A76.0%29+Gecko%2F20100101+Firefox%2F76.0%22%3Bs%3A13%3A%22last_activity%22%3Bi%3A1587987999%3Bs%3A9%3A%22user_data%22%3Bs%3A0%3A%22%22%3B%7Df634e42b8baef1d1803e00fb53718bcf49c53ed0"""    
    headers = dict([[s.strip() for s in line.split(':', 1)]
                    for line in headers.strip().split('\n')])
    
    from datetime import datetime
    import pytz
    # Current date time in local system
    tz = pytz.timezone('US/Central')
    x = datetime.now(tz)
    x = x.replace(tzinfo=None)
    TodayDate = datetime.date(x)
    TodayDate = TodayDate.strftime("%d-%m-%Y" )
    body = """station_id=32&date=%s""" % (TodayDate)
    import httplib2 
    h = httplib2.Http()
    url = 'https://rwis.tulsa.ou.edu/rwis/report/ajaxGetTempOverDay/'
    resp, content = h.request(url, 'POST', body=body, headers=headers)
    Q = json.loads(content)
    n= len(Q)
    date_time = [] 
    date_hour = []
    Temp = []
    Temp_Sur1 = []
    Temp_Sur2 = []
    Probe1 = []
    Probe2 = []
    for x in range(n):
        date_time.append(Q[x]['date_time'])
        date_hour.append(Q[x]['hour'])
        Temp.append(Q[x]['temp_avg'])
        Temp_Sur1.append(Q[x]['temp1_surface_temperature'])
        Temp_Sur2.append(Q[x]['temp2_surface_temperature'])
        Probe1.append(Q[x]['temp_probe_1'])
        Probe2.append(Q[x]['temp_probe_2'])

    RWIS_DF = pd.DataFrame({'Date': date_time,'Hour': date_hour,'Temperature': Temp, 'IR1':Temp_Sur1,'IR2':Temp_Sur1,'Probe 1':Probe1,'Probe 2':Probe1}, columns=['Date','Hour','Temperature','Probe 1','Probe 2','IR1','IR2'])
    #Create database to be shifted
    database1 = RWIS_DF
    database1  = database1.drop(columns=['Date', 'Hour'])
    
    from pandas import DataFrame
    from pandas import concat
# Function to convert series to supervised learning
    def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
        n_vars = 1 if type(data) is list else data.shape[1]
        df = DataFrame(data)
        cols, names = list(), list()
        for i in range(n_in, 0, -1):
            cols.append(df.shift(i))
            names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
        for i in range(0, n_out):
            cols.append(df.shift(-i))
            if i == 0:
                names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
            else:
                names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
        agg = concat(cols, axis=1)
        agg.columns = names
        if dropnan:
            agg.dropna(inplace=True)
        return agg
    from sklearn.preprocessing import LabelEncoder
    values1 = database1.values
    values1 = values1.astype('float32')
    scaled = scaler.transform(values1)
    n_steps = 2
    n_features = 5
    reframed1 = series_to_supervised(scaled, n_steps, 1)
    reframed1 = reframed1.iloc[-3:]
         
    from sklearn.preprocessing import MinMaxScaler
    database_API = Online_df_24['temperature']
    database_API = database_API.iloc[:3]
    database_API = database_API.values
    database_API = database_API.reshape(len(database_API),1)
    scaler_API = MinMaxScaler(feature_range=(0, 1))
    minmax = np.array([15.62, 84.92])
    minmax = minmax.reshape(len(minmax),1)
    scaler_API.fit_transform(minmax)
    database_API = scaler_API.transform(database_API)
    
    # Input data preparation for LSTM RNN
    test_X = reframed1.values
    test_X_input2 = database_API #Current Ambient Temperature
    # reshape input to be 3D [samples, timesteps, features]
    test_X = test_X.reshape((test_X.shape[0], 3, 5))
    test_X_input2 = test_X_input2.reshape((test_X_input2.shape[0], 1, 1))
    print(test_X.shape, test_X_input2.shape)
    
    # make a prediction
    yhat = model3.predict([test_X, test_X_input2])
    from numpy import concatenate
    test_X = test_X.reshape((test_X.shape[0], 15))
    inv_yhat = concatenate((test_X[:, -12:-9], yhat), axis=1)
    inv_yhat = concatenate ((inv_yhat, test_X[:, -1:]), axis=1)
    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:,3] # Predicted values in fahrenheit
    firsthour = RWIS_DF['Hour'].iloc[-1] +1
    secondhour = firsthour+1
    thirdhour = secondhour+1
    # Current date time in local system
    tz = pytz.timezone('US/Central')
    x = datetime.now(tz)
    x = x.replace(tzinfo=None)
    x = pd.to_datetime(x)
    x = x.replace(minute=0, second=0)
    firststep = x.replace(hour=firsthour)
    secondstep = x.replace(hour=secondhour)
    thirdstep = x.replace(hour=thirdhour)
    datastep = np.array([firststep, secondstep, thirdstep])
    dataframe = pd.DataFrame({'Date_Time' : datastep, 'Road Surface Temperature' : inv_yhat}, columns= ['Date_Time', 'Road Surface Temperature'])
    
    rng = pd.date_range('1/1/2011', periods=7500, freq='H')
    ts = pd.Series(np.random.randn(len(rng)), index=rng)

    graphs = [
        dict(
            data=[
                dict(
                   x=dataframe['Date_Time'],
                    y=dataframe['Road Surface Temperature'],
                    type='scatter'
                ),
            ],
            layout=dict(
                title='<b>Forecasting RNN LSTM Road Surface Temp For the Next Three hours</b>',
                family="verdana",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
            )
        ),

        dict(
            data=[
                dict(
                    x= prediction1.index,
                    y=prediction1['predicted temp'],
                    type='scatter'
                ),
            ],
            layout=dict(
                title='<b>DNN Model Road Surface Temp for the next 24 hours</b>',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
        ),

        dict(
            data=[
                dict(
                    x= prediction.index,
                    y=prediction['predicted temp'],
                    type='scatter'
                ),
            ],
            layout=dict(
                title='<b>Random Forest Model Road Surface Temp for the next 24 hours</b>',
                yaxis_title="<b>Temperature °F</b>",
                     paper_bgcolor='rgba(0,0,0,0)',
                     plot_bgcolor='rgba(0,0,0,0)'
            )
        )
    ]

    # Add "ids" to each of the graphs to pass up to the client
    # for templating
    ids = ['graph-{}'.format(i) for i, _ in enumerate(graphs)]

    # Convert the figures to JSON
    # PlotlyJSONEncoder appropriately converts pandas, datetime, etc
    # objects to their JSON equivalents
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template('index.html',
                           ids=ids,
                           graphJSON=graphJSON)

if __name__ == '__main__':
    app.run()