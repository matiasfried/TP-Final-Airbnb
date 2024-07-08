Airbnb (Kaggle)
Este repositorio presenta un análisis exploratorio y predictivo de la base de la datos de alojamientos de Airbnb realizado por Barreiro Francisco y Fried Matias para el SAT 81.75 - Fundamentos del desarrollo de software y análisis de datos en Python

Este repositorio contiene:

  Link_a_base.http: link a la carpeta de Google Drive donde se encuentran los archivos de cada ciudad relevada, utilizados para el análisis.

  Archivo Consolidacion_Base.py: Código en Python utilizado para la unión de todas las bases de datos utilizadas en una.

  Archivo Analisis_De_Datos.py : Código en Python que presenta el análisis exploratorio de datos, proporcionando insights detallados sobre la naturaleza de la base de datos.

  Archivo Modelo_Predictivo.py: Código en Python que presenta el proceso y el modelo final utilizado para realizar la predicción.

Conclusiones

El objetivo del trabajo final fue de proporcionar insights significativos sobre los factores que afectan los precios de las propiedades para proveer a los dueños y que de esta manera puedan determinar sus precios de la forma mas acertada posible. 

Partimos de un dataframe de 75 columnas, de las cuales la mayoria contenian valores nulos. Se realizo el descarte de diferentes columnas, ya sea por contener informacion parecida, por contener informacion dificil de interpretar en un analisis o por ser reemplazadas por variables dummies. Luego, llevamos a cabo el reemplazo de los nulos por valores de acuerdo a criterios interpretados segun la naturaleza de cada variable. De esta manera se obtuvo un dataframe de 45 columnas sin valores nulos. 

Del analisis de datos llevado a cabo, se puede interpretar que la variable 'precio' en su distribucion se encuentra en su mayoria en un rango de precio menor a $300, con una gran cantidad de outliers. El analisis de la distribucion de la variable 'accommodates' (huespedes) tiende en su mayoria a numeros pares, siendo 2 la cantidad mas repetida. Es evidente que a mayor cantidad de 'amenities', el precio tiende a ser mayor. Tambien, se observa un mayor precio de los listados de superhost contra quienes no son superhost, con una diferencia mas marcada mientras mas grande sea la propiedad. 

Con base en el análisis realizado, se recomienda a los administradores de propiedades de Airbnb considerar la optimización de las comodidades ofrecidas y la mejora del estatus de superhost para aumentar la competitividad y el valor percibido de las propiedades listadas. Además, se sugiere monitorear regularmente los precios en relación con la disponibilidad para ajustar estrategias de tarifas y maximizar el ingreso por propiedad. El feature importance llevado a cabo muestra que la ubicacion y la cantidad de amenities son por amplia ventaja las variables de mayor importancia en la formacion del precio. Luego, la cantidad de huespedes, la cantidad de baños y habitaciones, la cantidad minima de noches, la disponibilidad a 30 dias y el tipo de propiedad se encuentran en un segundo escalon. Sorprendentemente, las variables en relacion a las reseñas, no son de gran importancia, mas haya de que poseen cierto grado de relevancia. Las variables en relacion al comportamiento de los propietarios dentro de la plataforma, como la inclusion de descripciones, foto de perfil y verificaciones, no parecen tener incidencia en el precio de las propiedades.
