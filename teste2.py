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

df_customer = pd.read_csv('gs://stack-labs-list/olist_customers_dataset.csv', sep=";")
df_geolocation = pd.read_csv('gs://stack-labs-list/olist_geolocation_dataset.csv', sep=";")
df_order_items = pd.read_csv('gs://stack-labs-list/olist_order_items_dataset.csv', sep=";")
df_order_payments = pd.read_csv('gs://stack-labs-list/olist_order_payments_dataset.csv', sep=";")
df_order_reviews = pd.read_csv('gs://stack-labs-list/olist_order_reviews_dataset.csv', sep=";")
df_orders = pd.read_csv('gs://stack-labs-list/olist_orders_dataset.csv', sep=";")
df_products = pd.read_csv('gs://stack-labs-list/olist_products_dataset.csv', sep=";")
df_sellers = pd.read_csv('gs://stack-labs-list/olist_sellers_dataset.csv', sep=";")
df_category_name = pd.read_csv('gs://stack-labs-list/product_category_name_translation.csv', sep=";")


df = df_orders.merge(df_order_items, on='order_id', how='left')
df = df.merge(df_order_payments, on='order_id', how='outer', validate='m:m')
df = df.merge(df_order_reviews, on='order_id', how='outer')
df = df.merge(df_products, on='product_id', how='outer')
df = df.merge(df_customer, on='customer_id', how='outer')
df = df.merge(df_sellers, on='seller_id', how='outer')

df=df[['customer_state', 'customer_city', 'customer_id', 'customer_unique_id', 'seller_state', 'seller_id', 'order_id', 'order_item_id', 'order_status', 'order_purchase_timestamp', 'order_approved_at', 'order_estimated_delivery_date', 'order_estimated_delivery_date', 'freight_value', 'price']]

df.to_parquet('gs://stack-labs-list/df.parquet')

