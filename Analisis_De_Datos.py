#Inicio Analisis de Datos
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Mostramos del 10% de los datos porque los codigos se nos traba la computadora. Es una muestra representativa para poder entender los datos a priori
sample_df = df.sample(frac=0.1, random_state=42)
small_sample_df = df.sample(n=10000, random_state=42)


#Removemos valores superiores al percentil 90, para sacar outliers
upper_limit = small_sample_df['price'].quantile(0.90)
filtered_df = small_sample_df[small_sample_df['price'] <= upper_limit]

#Grafico precios por noche de las propiedades de Airbnb sin outliers
#Podemos identificar una distribución lognormal, con más alojamientos con valores de precios bajos (menos de 100) y disminuyendo a medida que sube el precio
sns.histplot(data=filtro, x='price', kde=True)
plt.title('Distribución de Precio por Noche (sin outliers)')
plt.xlabel('Precio por Noche')
plt.ylabel('Frecuencia')
plt.show()


#Gráfico de la distribución de la capacidad de alojamiento
#Podemos ver como la mayor cantidad de alojamientos tienen capacidad para 2 personas (parejas) o 4 personas (familia promedio)
sns.histplot(data=filtro, x='accommodates', kde=True)
plt.title('Distribución de la Capacidad de Alojamiento')
plt.show()


#Comparación de precios entre Superhost y No Superhost, se ve una tendencia de precios menores para los no superhost
upper_limit = small_sample_df['price'].quantile(0.90)
filtered_df = small_sample_df[small_sample_df['price'] <= upper_limit]
sns.boxplot(data=filtro, x='host_is_superhost', y='price')
plt.title('Precios: Superhost vs No Superhost')
plt.show()


#Relación entre la tasa de respuesta y la tasa de aceptación del anfitrión
#Se puede ver como hay mas probabilidades de que el anfitrión tenga una alta tasa tanto de respuesta como de aceptación
sns.scatterplot(data=small_sample_df, x='host_response_rate', y='host_acceptance_rate')
plt.title('Tasa de Respuesta vs Tasa de Aceptación')
plt.show()


#Sacamos las variables que tiene mucha correlación entre si y dejamos las mas relevantes para el análisis. Esto pasa principalmente con las variables de disponibilidad y de reviews
#Dejamos las variables de availability_30 (disponibilidad en el proximo mes), reviews promedio (métrica de calidad del alojamiento en general) y de reviews de locación (factor clave en los alojamientos de aribnb)
dflimpia = filtro.drop(['availability_60', 'availability_90','number_of_reviews_ltm','number_of_reviews_l30d','host_total_listings_count','review_scores_accuracy','review_scores_cleanliness','review_scores_checkin','review_scores_communication','review_scores_value','calculated_host_listings_count_entire_homes','Entire home/apt'], axis=1)


#Hacemos una matriz de correlación para poder identificar que variables están relacionadas entre si
#En principio podemos ver como variables como accomodates, bedrooms, beds y bathrooms estan relacionadas entre si, lo cual tiene lógica
#En cuanto a la variable que posteriormente (precio) podemos ver que no hay ninguna variable con una alta correlacion >0.8, lo cual es bueno. 
#Por otro lado, podemos encontrar varias variables con correlación entre 0,1 y 0,4, lo cual pueden ser útiles para aportar información a la hora de armar un modelo predictivo
#Estas variables son: availavility_30, accomodates, cantidad de reviews, si tiene descripción o no, cantidad de ammenities y si es un cuarto privado
corr = dflimpia.corr()
plt.figure(figsize=(24, 16))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Mapa de Calor de Correlaciones')
plt.show()

#Matriz de correlación más prolija, para identificar por colores las correlaciones a simple vista
plt.figure(figsize = (9,5))
corr=dflimpia.corr()
sns.heatmap(corr,
               xticklabels=corr.columns.values,
               yticklabels=corr.columns.values, cmap=sns.color_palette("Blues"))



#Comparación de precios por ammenities
#Mediante un gráfico de violin, podemos que si a mayor cantidad de ammenities aumenta levemente el precio del alojamiento
plt.figure(figsize=(16, 10))
#Gráfico de violín
sns.violinplot(data=dflimpia, x='amenities_count', y='price')
plt.title('Precios por Cantidad de Amenidades', fontsize=16)
plt.xticks(rotation=90, fontsize=12)
plt.yticks(fontsize=12)
plt.show()


#Para propiedades mas grandes parece haber una mayor influencia en el precio sobre si el host es superhost o no.
#Se puede ver en los boxplots en la linea de la mediana.
plt.figure(figsize=(10, 10))
# Generar el gráfico de violín con los ajustes necesarios
sns.violinplot(x='bedrooms', y='price', hue="host_is_superhost", data=dflimpia, palette="Blues", bw_method=0.2,
               cut=2, linewidth=2, split=True)
sns.despine(left=True)
plt.title('Bedrooms vs Price vs Is Superhost', weight='bold')
plt.show()



