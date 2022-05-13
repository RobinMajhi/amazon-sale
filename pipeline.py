"""
Spyder Editor

This is a temporary script file.
"""


import sqlalchemy as sql
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import combinations
from collections import Counter
import statsmodels.api as sm
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso
from sklearn.feature_selection import SelectFromModel

pd.set_option("float_format",lambda x :"%.2f" %x)



myconn=sql.create_engine("mysql+pymysql://root:@localhost:3306/e-commerce")

a=myconn.table_names()

data=pd.DataFrame()
for i in a:
    df=pd.read_sql_table(i,myconn)
    data=pd.concat([data,df])
    

data.reset_index(drop=True,inplace=True)


data_nm=data.loc[~(data["OrderID"].isnull()==True)]

data_nm.reset_index(drop= True,inplace=True)


data_nm=data_nm.loc[~(data_nm["PriceEach"]=="Price Each")]

data_nm.reset_index(drop= True,inplace=True)

data_nm["PriceEach"]=data_nm["PriceEach"].astype("float")


data_nm["QuantityOrdered"]=data_nm["QuantityOrdered"].astype("int")

data_nm["OrderDate"]=pd.to_datetime(data_nm["OrderDate"])

data_nm["sales"]=data_nm["QuantityOrdered"]*data_nm["PriceEach"]

data_pre=data_nm.copy()


data_pre["month"]=data_pre["OrderDate"].dt.month

data_pre["day"]=data_pre["OrderDate"].dt.day

data_sales=pd.DataFrame(data_pre.groupby("month")[["sales","QuantityOrdered"]].sum())

a=plt.figure(figsize=(24,8))
x=range(1,13)
y=data_sales["sales"]
plt.bar(x,y,color="RBGCMYK")
plt.xticks(x)
plt.ylabel("SALES in USD($)")
plt.xlabel("months")


a.savefig("visaulisation month vs sales")



fig, ax1 = plt.subplots(figsize=(30,10))
ax2 = ax1.twinx()
ax1.bar(x,y,color="RBGCMYK")
ax2.plot(x,data_sales["QuantityOrdered"], color="K")
plt.xticks(x)
ax1.set_xlabel('months')
ax1.set_ylabel('sales in USD($)', color='R')
ax2.set_ylabel('Quantity Orderded', color="G")
plt.show()


fig.savefig("visualisation month vs sales vs quantity ordered")


data_pre["Hour"]=data_pre["OrderDate"].dt.hour

data_pre["day"]=data_pre["OrderDate"].dt.day
data_pre.head()

peak_hour=pd.DataFrame(data_pre.groupby("Hour")[["sales","QuantityOrdered"]].sum())

peak_hour.reset_index(inplace=True)

fig1, ax1 = plt.subplots(figsize=(25,8))
x=peak_hour["Hour"]
y=peak_hour["sales"]
ax2 = ax1.twinx()
ax1.bar(x,y,color="RBGCMYK")
ax2.plot(x,peak_hour["QuantityOrdered"], color="K")
plt.xticks(x)
ax1.set_xlabel('Hour')
ax1.set_ylabel('sales in USD($)', color='R')
ax2.set_ylabel('Quantity Orderded', color="G")
plt.show()


fig1.savefig("visualisation hour vs sales vs quantity ordered")


data_pre["City"]=data_pre["Address"].apply(lambda x : x.split(",")[1] + " " + "({})".format(x.split(",")[2][1:3]))

city_sales=pd.DataFrame(data_pre.groupby("City")[["sales","QuantityOrdered"]].sum())
city_sales.reset_index(inplace=True)

fig2, ax1 = plt.subplots(figsize=(18,4))
x=city_sales["City"]
y=city_sales["sales"]
ax2 = ax1.twinx()
ax1.bar(x,y,color="RBGCMYK")
ax2.plot(x,city_sales["QuantityOrdered"], color="g")
plt.xticks(x,rotation="vertical")
ax1.set_xlabel('Cities')
ax1.set_ylabel('sales in USD($)', color='R')
ax2.set_ylabel('Quantity Orderded', color="G")
plt.show()


fig2.savefig("visualisation city vs sales vs quantity ordered")


top_city=data_pre.groupby(["City","Product"]).agg({"sales":sum})

g = top_city["sales"].groupby('City',group_keys=False)

product_city=int(input("enter your value to select the no of products in terms of city :"))
res = g.apply(lambda x: x.sort_values(ascending=False).head(product_city))
top5_product=pd.DataFrame(res)
top5_product.reset_index(inplace=True)


sns.set(rc = {'figure.figsize':(24,8)})
new_fig=sns.barplot(top5_product["City"],top5_product["sales"],hue=top5_product["Product"],palette="magma_r")
new_fig = new_fig.get_figure()

new_fig.savefig("visualisation top5 product as per city  vs sales vs quantity ordered.png",dpi=800)


df_asso = data_pre[data_pre['OrderID'].duplicated(keep=False)]
df_asso["grouped"]=df_asso.groupby('OrderID')['Product'].transform(lambda x : ",".join(x))
df_asso2=df_asso.drop_duplicates("OrderID")




count = Counter()

asso_product=int(input( "enter your value for the products to association with each other "))
top_product=int(input("how many products would you like to see for association "))

for row in df_asso2['grouped']:
    row_list = row.split(',')
    count.update(Counter(combinations(row_list, asso_product)))

products={}
for key,value in count.most_common(top_product):
    products[key]=value
    

df_top_product=pd.DataFrame(products.items(),columns=["products","frequency"])


sold_most=pd.DataFrame(data_pre.groupby("Product")[["sales","QuantityOrdered"]].sum())
sold_most.reset_index(inplace=True)


fig5, ax1 = plt.subplots(figsize=(18,4))
x=sold_most["Product"]
y=sold_most["sales"]
ax2 = ax1.twinx()
ax1.bar(x,y,color="RBGCMYK")
ax2.plot(x,sold_most["QuantityOrdered"], color="g")
ax1.set_xlabel('Products',color="M")
ax1.set_ylabel('sales in USD($)', color='R')
ax2.set_ylabel('Quantity Orderded', color="y")
ax1.set_xticklabels(sold_most["Product"], rotation='vertical', size=12,color="y")
plt.show()



    
data_fe=data_pre.drop(['OrderID', 'OrderDate',"Address"],axis=1)



print(data_fe["Product"].value_counts())

top_labels=int(input("how many labels would you like to select"))
nom_product=[ i for i in data_fe["Product"].value_counts().head(top_labels).index]



for i,j in zip(nom_product,range(len(data_fe["Product"]))):
    data_fe[i]=np.where(data_fe["Product"][j]==i,1,0)

print(data_fe["City"].value_counts())

top_labels_city=int(input("how many city labels would you like to select"))
nom_city=[ i for i in data_fe["City"].value_counts().head(top_labels_city).index]

for i,j in zip(nom_city,range(len(data_fe["City"]))):
    data_fe[i]=np.where(data_fe["City"][j]==i,1,0)
    
data_fe.drop(["Product","City","sales"],axis=1,inplace=True)    


with pd.ExcelWriter("AMAZON_PROJECT.xlsx") as writer:
    data_nm.to_excel(writer,sheet_name="cleanse_data",index=False)
    data_pre.to_excel(writer,"preprocess_data",index=False)
    data_sales.to_excel(writer,"sales_data_with_month",index=False)
    peak_hour.to_excel(writer,sheet_name="sales_data_with_month",index=False)
    city_sales.to_excel(writer,sheet_name="sales_data_with_city",index=False)
    top5_product.to_excel(writer,sheet_name="top5_product_sale_with_city",index=False)
    df_asso2.to_excel(writer,sheet_name="grouped products",index=False)
    sold_most.to_excel(writer,sheet_name="sold_most products",index=False)
    data_fe.to_excel(writer,sheet_name="data after feature engineering",index=False)


data_fe.columns


x_fs=data_fe[['QuantityOrdered', 'month', 'day', 'Hour',
       'USB-C Charging Cable', 'Lightning Charging Cable',
       'AAA Batteries (4-pack)', 'AA Batteries (4-pack)', 'Wired Headphones',
       'Apple Airpods Headphones', 'Bose SoundSport Headphones',
       '27in FHD Monitor', 'iPhone', '27in 4K Gaming Monitor',
       '34in Ultrawide Monitor', 'Google Phone', 'Flatscreen TV',
       'Macbook Pro Laptop', 'ThinkPad Laptop', ' San Francisco (CA)',
       ' Los Angeles (CA)', ' New York City (NY)', ' Boston (MA)',
       ' Atlanta (GA)', ' Dallas (TX)', ' Seattle (WA)', ' Portland (OR)',
       ' Austin (TX)', ' Portland (ME)']]

y_fs=data_fe['PriceEach']

reg = LassoCV()
reg.fit(x_fs, y_fs)
print("Best alpha using built-in LassoCV: %f" % reg.alpha_)
print("Best score using built-in LassoCV: %f" %reg.score(x_fs,y_fs))
coef = pd.Series(reg.coef_, index = x_fs.columns)
print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")
imp_coef = coef.sort_values()



imp_coef.plot(kind = "barh")
plt.title("Feature importance using Lasso Model")

    