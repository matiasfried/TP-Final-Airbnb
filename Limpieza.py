#Valores faltantes, mostrar de todas las filas
pd.options.display.max_rows = None
pd.options.display.max_columns = None


#Vemos el tamaño del Dataset
combined_df.shape

#Vemos que columnas tienen, para ver que podríamos analizar
print(combined_df.columns)

combined_df.sample(5).transpose()

combined_df.info()

#Sacamos todas las columnas que no nos interesan para el análisis
df = combined_df.drop(['id', 'listing_url','scrape_id','last_scraped','source','picture_url','host_id','host_url','host_name','host_thumbnail_url','host_picture_url','host_neighbourhood','neighbourhood','neighbourhood_cleansed','neighbourhood_group_cleansed','latitude','longitude','calendar_updated','has_availability','calendar_last_scraped','minimum_minimum_nights','maximum_minimum_nights','minimum_maximum_nights','maximum_maximum_nights','minimum_nights_avg_ntm','maximum_nights_avg_ntm','name'], axis=1)

df.describe(include= "all").transpose()

#Vemos la cantidad de datos nulos por columna

datosna=df.isna().sum()

datosna.columns = ['column_name','# NA']
datosna.sort_values(ascending = False)

missings=df.isnull().sum()

df.isna().sum() / df.shape[0] *100

#Borramos license porque tiene 70% de datos nulos y host response time porque esta en texto y no tiene un valor numerico
df = df.drop(['license','host_response_time'], axis=1)

#Convertimos las columnas de fecha al tipo datetime
df['first_review'] = pd.to_datetime(df['first_review'])
df['last_review'] = pd.to_datetime(df['last_review'])

#Calculamos la diferencia en días entre first_review y last_review para convertirla en una variable que aporte valor.
#Mide la frecuencia entre reviews

df['days_between_reviews'] = (df['last_review'] - df['first_review']).dt.days
df['days_between_reviews'].fillna(0, inplace=True)

#Las sacamos
df = df.drop(['first_review','last_review'], axis=1)

df['days_between_reviews'].head()

#Convertimos la columna host_since al tipo datetime
df['host_since'] = pd.to_datetime(df['host_since'])

#Definimos la fecha específica (en este caso, '2024-07-01')
specific_date = pd.Timestamp('2024-07-01')

#Calculamos la diferencia en días desde host_since hasta la fecha específica 
#Construimos una variable que muestra el tiempo que ha sido host el usuario

df['days_as_host'] = (specific_date - df['host_since']).dt.days
df['days_as_host'].fillna(0, inplace=True)

#La Sacamos
df = df.drop(['host_since'], axis=1)

df['days_as_host'].head()

#Para empezar a llenar los nulos, cambiamos por 1 si es nulo en la cantidad de propiedades del host asumiendo que tiene 1 sola propiedad ya que 0 propiedades es absurdo

df['host_listings_count'].fillna(1, inplace=True)
df['host_total_listings_count'].fillna(1, inplace=True)


#Cambiamos por la cantidad maxima de huespedes dividido 2 los nulos para bedrooms, para los nulos de beds, cambiamos por el numero de accomodates
df['bedrooms'].fillna((df['accommodates']/2).round().astype(int), inplace=True)
df['beds'].fillna(df['accommodates'], inplace=True)


#Transformamos la variable precio a numerica porque con % esta como string

df['price'] = df['price'].str.replace('$', '')
df['price'] = df['price'].str.replace(',', '')
df['price'] = df['price'].astype(float)

#Pasamos host response rate y host acceptance rate a numericas y cambiamos sus nulos por el promedio

df['host_response_rate'].replace('N/A', np.nan, inplace=True)
df['host_acceptance_rate'].replace('N/A', np.nan, inplace=True)

df['host_response_rate'] = df['host_response_rate'].astype(str).str.rstrip('%')
df['host_response_rate'] = df['host_response_rate'].astype(float)
df['host_response_rate'].mean()

df['host_acceptance_rate'] = df['host_acceptance_rate'].astype(str).str.rstrip('%')
df['host_acceptance_rate'] = df['host_acceptance_rate'].astype(float)
df['host_acceptance_rate'].mean()

#Imputacion de los faltantes con la media

df['host_response_rate'].fillna(93, inplace=True)
df['host_acceptance_rate'].fillna(85, inplace=True)


#Imputacion de los faltantes con la media

df['review_scores_value'].fillna(df['review_scores_value'].mean(), inplace=True)
df['review_scores_location'].fillna(df['review_scores_location'].mean(), inplace=True)
df['review_scores_checkin'].fillna(df['review_scores_checkin'].mean(), inplace=True)
df['review_scores_accuracy'].fillna(df['review_scores_accuracy'].mean(), inplace=True)
df['review_scores_communication'].fillna(df['review_scores_communication'].mean(), inplace=True)
df['review_scores_cleanliness'].fillna(df['review_scores_cleanliness'].mean(), inplace=True)
df['reviews_per_month'].fillna(df['reviews_per_month'].mean(), inplace=True)
df['review_scores_rating'].fillna(df['review_scores_rating'].mean(), inplace=True)

#creamos una nueva variable para ver si tiene o no descripcion el host

df['has_host_about'] = df['host_about'].notna().astype(int)

df = df.drop(['host_about'], axis=1)

#Creamos una nueva variable categórica que diga si tiene o no descripcion de la propiedad y del barrio y sacamos las otras

df['has_description'] = df['description'].notna().astype(int)
df['has_neighborhood_overview'] = df['neighborhood_overview'].notna().astype(int)

df=df.drop(['description', 'neighborhood_overview'], axis=1)

#Cambiamos los tipos de variables a categoricas
df = df.astype({'has_description':'category','has_neighborhood_overview':'category','has_host_about':'category'})

#Cambiamos valores 't' y 'f' a 1 y 0
df['host_is_superhost'] = df['host_is_superhost'].replace({'t': 1, 'f': 0})
#Cambiamos valores nulos por 0
df['host_is_superhost'].fillna(0, inplace=True)
df = df.astype({'host_is_superhost':'category'})

#Cambiamos valores 't' y 'f' a 1 y 0
df['host_has_profile_pic'] = df['host_has_profile_pic'].replace({'t': 1, 'f': 0})
#Cambiamos valores nulos por 0
df['host_has_profile_pic'].fillna(0, inplace=True)
df = df.astype({'host_has_profile_pic':'category'})

#Cambiamos valores 't' y 'f' a 1 y 0
df['host_identity_verified'] = df['host_identity_verified'].replace({'t': 1, 'f': 0})
#Cambiamos valores nulos por 0
df['host_identity_verified'].fillna(0, inplace=True)
df = df.astype({'host_identity_verified':'category'})

#Cambiamos valores 't' y 'f' a 1 y 0
df['instant_bookable'] = df['instant_bookable'].replace({'t': 1, 'f': 0})
#Cambiamos valores nulos por 0
df['instant_bookable'].fillna(0, inplace=True)
df = df.astype({'instant_bookable':'category'})


#La variable bathrooms tiene demasiados faltantes. Esto es porque la informacion mayormente esta en bathrooms_text, entonces vamos a extraer el numero de aqui.

import re
#Definimos una función para extraer el número de baños
def extract_bathrooms(text):
    if pd.isna(text):
        return 0
    match = re.search(r'(\d+(\.\d+)?)', str(text))
    if match:
        return float(match.group(1))
    return 0

df['bathrooms_nuevo'] = df.apply(
    lambda row: row['bathrooms'] if not pd.isna(row['bathrooms']) else extract_bathrooms(row['bathrooms_text']),
    axis=1
)

#Sacamos bathrooms y bathrooms_text ya que vamos a utilizar bathrooms_nuevo
df = df.drop(['bathrooms','bathrooms_text'], axis=1)
df['bathrooms_nuevo'].head()

df['host_verifications'].head()

#Las verificaciones estan en una lista de la forma [Email,Phone], entonces creamos una nueva variable de cantidad de verificaciones

df['host_verifications_count'] = df['host_verifications'].str.count(',')+1

#Llenamos los nulos con 0 verificaciones

df['host_verifications_count'].fillna(0, inplace=True)

df['host_verifications_count'].head()

df = df.drop(['host_verifications'], axis=1)

#Pasamos las ammenities a cantidad de amenities, era una variable object de la forma de la de verificaciones

df['amenities_count'] = df['amenities'].str.count(',')+1
df = df.drop(['amenities'], axis=1)

df['property_type'].unique()

#La sacamos porque no podemos clusterizar en base a esto, son descripciones
df = df.drop(['property_type'], axis=1)

df['room_type'].unique()

#Generamos variables dummies para cada tipo de room
dummies = pd.get_dummies(df['room_type'], dtype=int)

#Convertimos las columnas generadas a tipo categórico
for col in dummies.columns:
    dummies[col] = dummies[col].astype('category')

#Combinamos las columnas dummies con el DataFrame original
df = pd.concat([df, dummies], axis=1)

df = df.drop(['room_type'], axis=1)

#cambiamos la variable de ubicacion del host por la frecuencia que aparece cada ubicacion para poder utilizar la informacion

df['host_location'].fillna('Unclassified', inplace=True)
from sklearn.preprocessing import LabelEncoder

categorical_columns = df.select_dtypes(include=['object']).columns

#Iteramos sobre las columnas categóricas (solo queda location)
for column in categorical_columns:
    #Calculamos la frecuencia de cada categoría
    frequencies = df[column].value_counts(normalize=False)
    #Aplicamos la codificación de frecuencia a cada valor en la columna
    df[column] = df[column].map(frequencies).astype(int)

df.sample(15).transpose()

df.info()

#Ultimos Ajustes para definir el type de las variables

df = df.astype({'host_listings_count': int,'host_total_listings_count': int,'bedrooms': int,'beds': int,'days_between_reviews': int,'days_as_host': int,'bathrooms_nuevo': int,'host_verifications_count': int})

#verificamos que no queden nulos
datosna=df.isna().sum()

datosna.columns = ['column_name','# NA']
datosna.sort_values(ascending = False)

missings=df.isnull().sum()

missings.columns = ['column_name','# missings']
missings.sort_values(ascending = False)

df.isna().sum() / df.shape[0] *100

#vemmos que los nulos de precio estan a lo largo de todo el dataframe
missingno.matrix(df)
plt.xticks(ticks=np.arange(len(df.columns)), labels=df.columns, rotation=90)
plt.show()

#Parece haber muchos valores atipicos viendo los maximos de precio, bedrooms y beds. Nos metimos a los links de las publicaciones y parecen ser estafas
#No puede ser que hayan 100 bedrooms por ejemplo o que la cantidad de habitaciones sea exageradamente mayor a la cantidad de huespedes.

df.describe(include= "all").transpose()

#Agregamos un filtro para casos como los que vimos donde tenes 48 habitaciones pero solo acomoda a 2 personas, lo cual es absurdo
#Agregamos filtro de precio ya que muchas publicaciones con un precio tan alto son posiblemente scam o no hayan puesto el precio por noche por lo tanto no se puede analizar
#Tenemos en cuenta que agregar el filtro del precio saca los nulos tambien que son 150.000 aprox, establecemos el limite de precio por noche de 3000 dolares
#Propiedades de hasta 10 bedrooms, beds y bathrooms es logico y no filtra tantos datos (aprox 2000 filas de mas de 500.000)
filtro = df[(df['bedrooms'] < 10) & (df['beds'] < 10) & (df['bathrooms_nuevo'] < 10) & (df['price'] < 3000) ]

filtro.shape

filtro.describe(include= "all").transpose()
#Podemos ver valores mas logicos en la descripcion estadistica

filtro.sample(5).transpose()

