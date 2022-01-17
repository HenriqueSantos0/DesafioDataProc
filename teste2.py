import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import cufflinks as cf
cf.go_offline()

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.linear_model import LinearRegression
#%matplotlib inline

from folium import Choropleth, Circle, Marker
from folium.plugins import HeatMap, MarkerCluster
import folium
from folium.plugins import FastMarkerCluster

import plotly.offline as py
import plotly.graph_objs as go

import plotly.express as px
           

df_customer = pd.read_csv('gs://stack-labs-list/landig/olist_customers_dataset.csv', sep=";")
df_geolocation = pd.read_csv('gs://stack-labs-list/landig/olist_geolocation_dataset.csv', sep=";")
df_order_items = pd.read_csv('gs://stack-labs-list/landig/olist_order_items_dataset.csv', sep=";")
df_order_payments = pd.read_csv('gs://stack-labs-list/landig/olist_order_payments_dataset.csv', sep=";")
df_order_reviews = pd.read_csv('gs://stack-labs-list/landig/olist_order_reviews_dataset.csv', sep=";")
df_orders = pd.read_csv('gs://stack-labs-list/landig/olist_orders_dataset.csv', sep=";")
df_products = pd.read_csv('gs://stack-labs-list/landig/olist_products_dataset.csv', sep=";")
df_sellers = pd.read_csv('gs://stack-labs-list/landig/olist_sellers_dataset.csv', sep=";")
df_category_name = pd.read_csv('gs://stack-labs-list/landig/product_category_name_translation.csv', sep=";")


df = df_orders.merge(df_order_items, on='order_id', how='left')
df = df.merge(df_order_payments, on='order_id', how='outer', validate='m:m')
df = df.merge(df_order_reviews, on='order_id', how='outer')
df = df.merge(df_products, on='product_id', how='outer')
df = df.merge(df_customer, on='customer_id', how='outer')
df = df.merge(df_sellers, on='seller_id', how='outer')

df=df[['customer_state', 'customer_city', 'customer_id', 'customer_unique_id', 'seller_state', 'seller_id', 'order_id', 'order_item_id', 'order_status', 'order_purchase_timestamp', 'order_approved_at', 'order_estimated_delivery_date', 'freight_value', 'price']]

df.to_parquet('gs://stack-labs-list/processing/df')

df=pd.read_parquet('gs://stack-labs-list/processing/df')


# Total Clientes por estado
df_customer['customer_state'].value_counts()

plt.figure(figsize=(20,10))
g = sns.countplot(x='customer_state', 
                  data=df_customer,
                 order = df_customer['customer_state'].value_counts().index,
                 palette='icefire_r')
g.set_title("Clientes por Estado", fontsize = 15, weight='bold')
g = plt.xticks(fontsize = 13)


total_cities = df_customer.groupby('customer_city')['customer_id'].count().sort_values(ascending=[False])[:15]

plt.figure(figsize=(20,10))
sns.barplot(x = total_cities.index,
            y = total_cities.values,
                palette='icefire_r');


round((df_geolocation['geolocation_state'].value_counts()/len(df_geolocation['geolocation_state']))*100,2
      
#Status dos pedidos
#tive que rodar a função configure_plotly_browser_state para executar o gráfico do plotly no colab
configure_plotly_browser_state()
df_orders['order_status'].value_counts().iplot(kind='bar')
      

df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])
df['order_purchase_month'] = df['order_purchase_timestamp'].dt.to_period('M').astype(str)
      
     
configure_plotly_browser_state()
data_vendas_mes = [go.Bar(x=vendas_por_mes.index,
                          y=vendas_por_mes.values,
                          marker = {'color': 'lightblue',
                                    'line':{'color': '#333',
                                            'width': 2}})]

# Layout Gráfico

config_layout = go.Layout(title='Vendas no Periodo',
                          yaxis = {'title':'Valores em Vendas'},
                          xaxis={'title':'Periodo'})

#Objeto Figura

fig = go.Figure(data=data_vendas_mes, layout=config_layout)

#Plot grafico
py.iplot(fig)
      
configure_plotly_browser_state()
trace = go.Scatter(x=df['freight_value'],
                   y=df['price'],
                   mode = 'markers')

#armazenando gráfico em lista
data=[trace]

#layout
layout = go.Layout(title='Valor do Frete x Valor do Produto',
                   yaxis={'title':'Valor do Produto'},
                   xaxis={'title':'Valor do Frete'})

#Criando a figura que será exibida
fig = go.Figure(data=data, layout=layout)

py.iplot(fig)
      
      
      
      
 #Calculo das informações
freightAvgState = (df.groupby('customer_state')['freight_value'].sum() /
                   df.groupby('customer_state')['order_id'].nunique()).sort_values(ascending=False)

#Plotando o gráfico
plt.figure(figsize = (15,5))
g = sns.barplot(x=freightAvgState.index, 
                y=freightAvgState,
                palette='icefire_r'
                )
g.set_title('Média do Frete por Estado em R$')

#Incluindo a linha horizontal com a média do frete pago

g.axhline(df['freight_value'].mean(), color = 'red')
      
      
df['order_freight_ratio'] = df.freight_value / df.price
      
      
# Conversão das colunas para dateto,e

df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])
df['order_approved_at'] = pd.to_datetime(df['order_approved_at'])
df['order_estimated_delivery_date'] = pd.to_datetime(df['order_estimated_delivery_date'])
df['order_delivered_customer_date'] = pd.to_datetime(df['order_delivered_customer_date'])

# Cálculo das diferenãs em horas

df['delivery_time'] = (df['order_delivered_customer_date'] - df['order_approved_at']).dt.total_seconds() / 86400
df['estimated_delivery_time'] = (df['order_estimated_delivery_date'] - df['order_approved_at']).dt.total_seconds() / 86400

# Plotando o gráfico o tempo estimado x o tempo real de entrega

plt.figure(figsize=(10,4))
plt.title("Tempo de entrega em Dias")
ax1 = sns.kdeplot(df['delivery_time'].dropna(), color="red", label='Delivery time')
ax2 = sns.kdeplot(df['estimated_delivery_time'].dropna(), color="gray", label='Estimated delivery time')
      
      
df['diff_delivery_estimated'] = df['delivery_time'] - df['estimated_delivery_time']
      
data_delivery_produtcs = df[df['diff_delivery_estimated']> 0 ]
      
plt.figure(figsize=(20,10))
g = sns.countplot(x='customer_state', 
                  data=data_delivery_produtcs,
                 order = data_delivery_produtcs['customer_state'].value_counts().index,
                 palette='icefire_r')
g.set_title("Encomendas que Atrasaram por Estado", fontsize = 15, weight='bold')
g = plt.xticks(fontsize = 13)
      
data_delivery_produtcs.groupby('customer_state')['diff_delivery_estimated'].sum().sort_values(ascending=False)
      
data_porcent_atraso = (data_delivery_produtcs.groupby('customer_state')['order_id'].nunique().sort_values(ascending=False) / df.groupby('customer_state')['order_id'].nunique().sort_values(ascending=False)) * 100
      
data_porcent_atraso.sort_values(ascending=False)
      
      
#Função para criar a feature de região
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
      
df_1 = df[(df['customer_region'] == 'Nordeste') | (df['customer_region'] == 'Norte')]
      
df_customer_region = df_1[(df_1.seller_region != "Nordeste") & (df_1.seller_region != "Norte")]
      
#Gráfico de pizza mostrando a % de compras das regiões norte e nordeste em outras regiões, neste gráfico
# não está sendo considerada as compras feitas nas próprias regiões

df_customer_region['seller_region'].value_counts().plot.pie()
      
      
#Gráfico de pizza mostrando a % de compras das regiões norte e nordeste incluindo as compras na regiões norte e nordeste

df_1['seller_region'].value_counts().plot.pie()
      
      
