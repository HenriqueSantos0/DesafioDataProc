import pandas as pd
from fbprophet import Prophet
from fbprophet.plot import plot_plotly, plot_components_plotly
from sklearn.metrics import mean_squared_error
from math import sqrt

#Lendo arquivos em parquet
df_customer=pd.read_parquet('gs://stack-labs-list/processing/df_customer')
df_geolocation=pd.read_parquet('gs://stack-labs-list/processing/df_geolocation')
df_order_items=pd.read_parquet('gs://stack-labs-list/processing/df_order_items')
df_order_payments=pd.read_parquet('gs://stack-labs-list/processing/df_order_payments')
df_order_reviews=pd.read_parquet('gs://stack-labs-list/processing/df_order_reviews')
df_orders=pd.read_parquet('gs://stack-labs-list/processing/df_orders')
df_products=pd.read_parquet('gs://stack-labs-list/processing/df_products')
df_sellers=pd.read_parquet('gs://stack-labs-list/processing/df_sellers')
df_category_name=pd.read_parquet('gs://stack-labs-list/processing/df_category_name')

#Executando merge
df = df_orders.merge(df_order_items, on='order_id', how='left')
df = df.merge(df_order_payments, on='order_id', how='outer', validate='m:m')
df = df.merge(df_order_reviews, on='order_id', how='outer')
df = df.merge(df_products, on='product_id', how='outer')
df = df.merge(df_customer, on='customer_id', how='outer')
df = df.merge(df_sellers, on='seller_id', how='outer')

df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp']).dt.date
df1 = df.groupby('order_purchase_timestamp')['product_id'].count().reset_index()

split_point = len(df1)-190
df_model = df1[0:split_point]
validation = df1[split_point:]

df_model = df_model[['order_purchase_timestamp', 'product_id']].rename(columns = {'order_purchase_timestamp': 'ds', 'product_id': 'y'})
validation = validation[['order_purchase_timestamp', 'product_id']].rename(columns = {'order_purchase_timestamp': 'ds', 'product_id': 'y'})

df_validation = pd.DataFrame({'ds': validation['ds']})

model = Prophet()
model.fit(df_model)

saida = model.predict(df_validation)

rmse = sqrt(mean_squared_error(validation['y'], saida.yhat))
print('Test RMSE: %.3f' % rmse)

future = model.make_future_dataframe(periods=50, freq='M')  
forecast = model.predict(future)

def transformar_estado(valor):
    if valor == 'AC':
        return 'Norte'
    elif valor == 'AP':
        return 'Norte'
    elif valor == 'AM':
        return 'Norte'
    elif valor == 'PA':
        return 'Norte'
    elif valor == 'RO':
        return 'Norte'
    elif valor == 'RR':
        return 'Norte'
    elif valor == 'TO':
        return 'Norte'
    elif valor == 'AL':
        return 'Nordeste'
    elif valor == 'BA':
        return 'Nordeste'
    elif valor == 'CE':
        return 'Nordeste'
    elif valor == 'MA':
        return 'Nordeste'
    elif valor == 'PB':
        return 'Nordeste'
    elif valor == 'PE':
        return 'Nordeste'
    elif valor == 'PI':
        return 'Nordeste'
    elif valor == 'RN':
        return 'Nordeste'
    elif valor == 'SE':
        return 'Nordeste'  
    elif valor == 'DF':
        return 'Centro-Oeste'
    elif valor == 'GO':
        return 'Centro-Oeste'
    elif valor == 'MT':
        return 'Centro-Oeste'
    elif valor == 'MS':
        return 'Centro-Oeste'
    elif valor == 'ES':
        return 'Sudeste'
    elif valor == 'RJ':
        return 'Sudeste'
    elif valor == 'MG':
        return 'Sudeste'
    elif valor == 'SP':
        return 'Sudeste'
    else:
        return 'Sul'

df['customer_region'] = df['customer_state'].map(transformar_estado)
df['seller_region'] = df['seller_state'].map(transformar_estado)

df_region_1 = df[(df['customer_region'] == 'Nordeste') | (df['customer_region'] == 'Norte') | (df['customer_region'] == 'Centro-Oeste')]


df_region_1['order_purchase_timestamp'] = pd.to_datetime(df_region_1['order_purchase_timestamp']).dt.date
df_region = df_region_1.groupby('order_purchase_timestamp')['product_id'].count().reset_index()

split_point = len(df_region)-180
df_model_region = df_region[0:split_point]

validation_region = df_region[split_point:]

df_model_region = df_model_region[['order_purchase_timestamp', 'product_id']].rename(columns = {'order_purchase_timestamp': 'ds', 'product_id': 'y'})
validation_region = validation_region[['order_purchase_timestamp', 'product_id']].rename(columns = {'order_purchase_timestamp': 'ds', 'product_id': 'y'})

df_validation = pd.DataFrame({'ds': validation_region['ds']})

model = Prophet()

model.fit(df_model_region)

saida = model.predict(df_validation)

rmse = sqrt(mean_squared_error(validation_region['y'], saida.yhat))
print('Test RMSE: %.3f' % rmse)

future = model.make_future_dataframe(periods=50, freq='M')  
forecast = model.predict(future)

figura = model.plot(saida, xlabel = 'Data', ylabel = 'Vendas');

model.plot_components(saida)

future = model.make_future_dataframe(periods=50, freq='M')  
forecast = model.predict(future)

model.plot(forecast, xlabel = 'Date', ylabel = 'Vendas');





#DATA_PATH = "gs://stack-labs-list/curated/"
#with open(DATA_PATH+"test.pickle", 'rb') as f:
     #pickle.dump(model, f)

