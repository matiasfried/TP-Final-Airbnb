#A continuacion, realizamos los modelos predictivos para el precio del alojamiento. El objetivo es encontrar si con los datos que disponemos se puede obtener un bueno modelo y poder saber como pricear un alojamiento de manera óptima
#Utilizamos tantos modelos de arboles de regresión logística como regresiones lineales con posteriores correcciones Ridge y Lasso
#Se obtuvieron modelos con R2 cercanos a 0,4 

#Luego, se realizo otro modelo predictivo para identificar dependiendo del precio y de las caracteristicas del alojamiento, si este iba a ser reservado dentro del proximo mes (availavility_30), con el objetivo de que el anfitrión pueda forecastear sus ingresos futuros

pip install catboost
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.impute import SimpleImputer
import xgboost as xgb
from sklearn.ensemble import BaggingRegressor
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import lightgbm as lgb
from catboost import CatBoostRegressor


#Modelo 1 - Prediccion de Precio

#ÁRBOL DE DECICIÓN DecisionTreeRegressor

#Definimos X e y
X = dflimpia.drop('price',axis=1)
y = dflimpia['price']

#Dividimos los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=33)

#Elijimos los hiperparámetros y defino la regresión
regr1 = DecisionTreeRegressor(max_depth=6, min_samples_split=6)
#Entrenamos el modelo
regr1.fit(X_train, y_train)

#Hacemos la predicción
y_pred1 = regr1.predict(X_test)

#Calculamos las métricas de evaluación
mse_1 = mean_squared_error(y_test, y_pred1)
mae_1 = mean_absolute_error(y_test, y_pred1)
r2_1 = r2_score(y_test, y_pred1)

print("MSE del Decision Tree Regressor:", mse_1)
print("MAE del Decision Tree Regressor:", mae_1)
print("R2 Score del Decision Tree Regressor:", r2_1)

#Realizamos validación cruzada con 10-fold
scores = cross_val_score(regr1,  X, y, cv=10, scoring='r2')
#Convertimos las puntuaciones negativas a positivas y calculo la media
mse_1_cv = scores.mean()
print("R2 Score dela validación cruzada es:", mse_1_cv)


#Modelo 2 - Prediccion de Precio

#ÁRBOL DE DECICIÓN DecisionTreeRegressor con datos normalizados

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#Dividimos los datos en entrenamiento y prueba
X = dflimpia.drop('price',axis=1)

y = dflimpia['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Normalizamos X_train y X_test
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

#Normalizamos y_train y y_test si es necesario (aunque no es común para el objetivo)
scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()
y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1)).flatten()

#Reentrenamos el modelo con los datos normalizados
print("Pruebo el árbol de decisión de DecisionTreeRegressor")

regr1 = DecisionTreeRegressor(max_depth=10, min_samples_split=10)
regr1.fit(X_train_scaled, y_train_scaled)

y_pred1 = regr1.predict(X_test_scaled)

mse_1 = mean_squared_error(y_test_scaled, y_pred1)
mae_1 = mean_absolute_error(y_test_scaled, y_pred1)
r2_1 = r2_score(y_test_scaled, y_pred1)

print("MSE del Decision Tree Regressor:", mse_1)
print("MAE del Decision Tree Regressor:", mae_1)
print("R2 Score del Decision Tree Regressor:", r2_1)

#Realizamos validación cruzada con 5-fold

# La distribucion de precio está sesgado hacia un extremo (la mayoría de los precios están en el rango bajo), esto indica un desbalanceo.
# Tiene una cola larga hacia un lado y muchos outliers, una señal de desbalanceo.
# Por lo tanto hacemos la validacion cruzada en 5 partes en vez de 10. 

scores = cross_val_score(regr1, X, y, cv=10, scoring='r2')
mse_1_cv = scores.mean()
print("R2 Score de la validación cruzada es:", mse_1_cv)


#Modelo 3 - Prediccion de Precio

#Grid
#Se intento probar con este modelo, sin embargo estuvimos 1h esperando y no conseguimos que termine de correr. El codigo no esta dentro de este archivo. Lo mismo nos paso con otros modelos mas complejos.


#Modelo 4 - Prediccion de Precio

#Regresión Lineal

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.feature_selection import f_regression

X = dflimpia.drop('price', axis=1)
y = dflimpia['price']

#Dividimos los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Creamos el modelo de regresión lineal
linear_reg = LinearRegression()

#Ajustamos el modelo a los datos de entrenamiento
linear_reg.fit(X_train, y_train)

#Hacemos predicciones en el conjunto de prueba
y_pred = linear_reg.predict(X_test)

#Calculamos las métricas de evaluación
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("MSE de la Regresión Lineal:", mse)
print("MAE de la Regresión Lineal:", mae)
print("R2 Score de la Regresión Lineal:", r2)

#Coeficientes del modelo
coefficients = pd.DataFrame(linear_reg.coef_, X.columns, columns=['Coefficient'])
print(coefficients)

#Graficamos los coeficientes
plt.figure(figsize=(10, 6))
sns.barplot(x=coefficients['Coefficient'], y=coefficients.index)
plt.title('Importancia de las Características (Coeficientes de Regresión Lineal)')
plt.show()


#Análisis de Varianza (ANOVA) Para ver la significancia de los tests. Todos los estadisticos dan dentro del rango de aceptacion
X = dflimpia.drop('price', axis=1)
y = dflimpia['price']
f_values, p_values = f_regression(X, y)
anova_results = pd.DataFrame({
    'Feature': X.columns,
    'F-Value': f_values,
    'P-Value': p_values
})
print(anova_results.sort_values(by='F-Value', ascending=False))


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import f_regression
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

#Corrección Modelo Regresión Lineal con Regresión Ridge
ridge_reg = Ridge(alpha=1.0)
ridge_reg.fit(X_train, y_train)
y_pred_ridge = ridge_reg.predict(X_test)
mse_ridge = mean_squared_error(y_test, y_pred_ridge)
mae_ridge = mean_absolute_error(y_test, y_pred_ridge)
r2_ridge = r2_score(y_test, y_pred_ridge)
print("MSE de la Regresión Ridge:", mse_ridge)
print("MAE de la Regresión Ridge:", mae_ridge)
print("R2 Score de la Regresión Ridge:", r2_ridge)
ridge_coefficients = pd.DataFrame(ridge_reg.coef_, X.columns, columns=['Coefficient'])
print(ridge_coefficients)

#Corrección Modelo Regresión Lineal con Regresión Lasso
lasso_reg = Lasso(alpha=0.1)
lasso_reg.fit(X_train, y_train)
y_pred_lasso = lasso_reg.predict(X_test)
mse_lasso = mean_squared_error(y_test, y_pred_lasso)
mae_lasso = mean_absolute_error(y_test, y_pred_lasso)
r2_lasso = r2_score(y_test, y_pred_lasso)
print("MSE de la Regresión Lasso:", mse_lasso)
print("MAE de la Regresión Lasso:", mae_lasso)
print("R2 Score de la Regresión Lasso:", r2_lasso)
lasso_coefficients = pd.DataFrame(lasso_reg.coef_, X.columns, columns=['Coefficient'])
print(lasso_coefficients)

#Para ambos modelos se obtiene un r2 muy bajo de 0,2 aproximadamente


#Modelo 5 - Prediccion de Disponibilidad dependiendo de todas las demás variables. Objetivo ver si se va a reservar el alojamiento.
#Se obtiene un modelo con un r2 de 0,57

#DecisionTreeRegressor

#Instalación de bibliotecas necesarias
!pip install catboost xgboost lightgbm

#Importación de bibliotecas
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor, BaggingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

#Definimos X e y
X = dflimpia.drop('availability_30', axis=1)
y = dflimpia['availability_30']

#Dividimos los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=33)

#Normalizamos X_train y X_test
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

#Normalizamos y_train y y_test si es necesario
scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()
y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1)).flatten()


print("Pruebo el árbol de decisión de DecisionTreeRegressor")
regr1 = DecisionTreeRegressor(max_depth=10, min_samples_split=10)
regr1.fit(X_train_scaled, y_train_scaled)
y_pred1 = regr1.predict(X_test_scaled)
mse_1 = mean_squared_error(y_test_scaled, y_pred1)
mae_1 = mean_absolute_error(y_test_scaled, y_pred1)
r2_1 = r2_score(y_test_scaled, y_pred1)
print("MSE del Decision Tree Regressor:", mse_1)
print("MAE del Decision Tree Regressor:", mae_1)
print("R2 Score del Decision Tree Regressor:", r2_1)

# Realizamos validación cruzada con 10-fold
scores = cross_val_score(regr1, X, y, cv=5, scoring='r2')
mse_1_cv = scores.mean()
print("R2 Score de la validación cruzada es:", mse_1_cv)

