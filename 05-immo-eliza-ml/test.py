import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import hstack

data = pd.read_csv("C:\\Users\\jyoti\\Immo-Eliza-Model-Building\\05-immo-eliza-ml\\data\\properties.csv")

data.head()

data.columns

y = data["price"]

num_values = data[["price","total_area_sqm", "latitude", "longitude",'surface_land_sqm', "garden_sqm", "primary_energy_consumption_sqm", "construction_year", "cadastral_income", 'nbr_frontages', 'nbr_bedrooms', "terrace_sqm" ]]

cat_values = data[["property_type", "subproperty_type", "region", "province", "locality", "zip_code", "state_building", "epc", "heating_type", 'equipped_kitchen']]

bin_values = data[["fl_garden", "fl_furnished", "fl_open_fire", "fl_terrace","fl_swimming_pool", "fl_floodzone", "fl_double_glazing"]]

print(type(num_values))

imputer = SimpleImputer(strategy="mean")
imputer.fit(num_values)
imp_num_values = imputer.transform(num_values)

print(type(imp_num_values))

imp_num_val = pd.DataFrame(imp_num_values, columns = ["price","total_area_sqm", "latitude", "longitude",'surface_land_sqm', "garden_sqm", "primary_energy_consumption_sqm", "construction_year", "cadastral_income", 'nbr_frontages', 'nbr_bedrooms', "terrace_sqm"])

print(type(imp_num_val))

print(imp_num_val.head())

heatmap = sns.heatmap((imp_num_val).corr()[['price']].sort_values(by='price', ascending=False), vmin=-1, vmax=1, annot=True, cmap='BrBG')

cat_values["equipped_kitchen"].value_counts().to_frame() #--> lot of missing values
cat_values["property_type"].value_counts().to_frame() # --> OK (house and apartment), BINARY?
cat_values["subproperty_type"].value_counts().to_frame() # -->OK but many subtypes, how many to include?
cat_values["region"].value_counts().to_frame()  # --> OK


encoded_df = pd.DataFrame()

label_encoders = {}

for column in cat_values.columns:
    le = LabelEncoder()
    encoded_df[column] = le.fit_transform(cat_values[column])
    label_encoders[column] = le

encoded_df['price'] = data['price']

plt.figure(figsize=(10, 8))
sns.heatmap(encoded_df.corr()[["price"]].sort_values(by='price', ascending=False), vmin=-1, vmax=1, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()