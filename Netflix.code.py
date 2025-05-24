import pandas as pd,numpy as np,matplotlib.pyplot as plt,seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df =  pd.read_csv("netflix_titles.csv")
df.info()
df.index
df.shape
df.isnull().sum()
df.head()
# Corregimos tipo de dato(date_added) y extraemos mes y año
df['date_added']= pd.to_datetime(df['date_added'],errors='coerce')
#Comprobamos que hemos cambiado el tipo
#print(df['date_added'].dtype)
#Separamos la columna date_added en 2 columnas: una para el año y otra para el mes, que queremos por su nombre
df['year_added']= df['date_added'].dt.year
df['month_added']= df['date_added'].dt.month_name()
df.head()
#convertimos la columna 'year_added' con posibles NaN a un numero entero
df['year_added'] = df['year_added'].dropna().astype('Int64')
#Comprobamos que ahora la columna está asociada al nuevo tipo e imprimimos el número de nulos 
print(df['year_added'].dtype)
print(df['year_added'].isnull().sum())
#Añadimos columna 'month_num' para obtener los meses por número y así poder ordenarlos y obtener métricas para gráficos, también cambiamos el tipo de datos de la columna month_num 
df['month_num'] = df['date_added'].dt.month
df['month_num'] = df['month_num'].dropna().astype('Int64')
#Separamos columnas complejas y las convertimos en listas
df['cast'] = df['cast'].fillna('').apply(lambda x: [i.strip() for i in x.split(',')] if x else [])
df['listed_in'] = df['listed_in'].fillna('').apply(lambda x: [i.strip() for i in x.split(',')] if x else [])
df['director'] = df['director'].fillna('').apply(lambda x: [i.strip() for i in x.split(',')] if x else [])
#Reemplazamos los NaN de las columnas solo tipo object(texto) y no a numéricas o fechas:
df[df.select_dtypes(include='object').columns] = df.select_dtypes(include='object').fillna('unknown')
#PROCESAMIENTO LA COLUMNA DURATION
#Comprobamos los valores que contiene la columna
df['duration'].unique()
#verificamos si hay elementos nulos
df['duration'].isna().sum()
#separamos contenido en números y unidad(min/temporadas)
#(\d+) → captura uno o más dígitos (el número)
#\s* → ignora espacios intermedios
#(\D+) → captura los caracteres no numéricos (el tipo)
df[['duration_int','duration_type']] = df['duration'].str.extract(r'(\d+)\s*(\D+)')
#convertimos la columna duration_int a numerico
df['duration_int'] = pd.to_numeric(df['duration_int'],errors='coerce')
#Comprobar el tipo de dato,valores únicos y los valores nulos de la columna:
#df['duration_int'].dtype
#df['duration_type'].unique()
#df['duration_type'].isna().sum()
#Normalizamos las unidades
df['duration_type'] = df['duration_type'].str.strip().str.lower()
df['duration_type'] = df['duration_type'].replace({
    'season':'seasons',
    'min':'minutes'
})
df['duration_int'] = df['duration_int'].dropna().astype('Int64')
#guardamos el nuevo dataframe modificado y comprobamos las modificaciones:
df = df.copy()
df.head(10)
#Obtenemos tipo de contenido(peli/serie)
type_counts= df['type'].value_counts()
print(type_counts)
#Obtenemos gráfico
plt.figure(figsize=(6,4))
sns.barplot(x=type_counts.index, y=type_counts.values, palette='pastel')
plt.title('Distribución por tipo de contenido')
plt.xlabel('Tipo de contenido')
plt.ylabel('Cantidad')
plt.show()
#Nos aseguramos que 'release_year' está en formato int
df['release_year'] = df['release_year'].astype(int)
#Listado de las producciones por año a partir del año 2000
yearly_counts = df['release_year'].value_counts().sort_index()
print(yearly_counts.tail(22))
#Gráfico
plt.figure(figsize=(12,6))
sns.lineplot(x=yearly_counts.index,y=yearly_counts.values, marker='o')
plt.title('Títulos añadidos por año')
plt.xlabel('Año de estreno')
plt.ylabel('Cantidad de títulos')
plt.grid(True)
plt.show()
#filtramos películas
movies_df = df[df['type']=='Movie']
#Estadísticas básicas:
print(movies_df['duration_int'].describe())
#Gráfico
plt.figure(figsize=(10,5))
sns.histplot(movies_df['duration_int'],bins=30,color='skyblue')
plt.title('Distribución de duraciónde películas')
plt.xlabel('Duración en minutos')
plt.ylabel('Cantidad de películas')
plt.show()
#Limpiamos el campo con los valores a una misma fila
df['country_clean'] = df['country'].fillna('unknown').apply(lambda x: x.split(',')[0].strip())
#Listado de los 10 paises con más producciones:
top_countries = df['country_clean'].value_counts().head(10)
print(top_countries)
plt.figure(figsize=(10,5))
sns.barplot(x=top_countries.values,y=top_countries.index, palette='viridis')
plt.title('Top 10 países con más contenido')
plt.xlabel('Número de producciones')
plt.ylabel('País')
plt.show()
#Como antes hemos convertido 'listed_in' en una lista de elementos, ahora la podemos explotar directamente:
genres_exploded = df.explode('listed_in')

#Limpiamos espacios por si acaso
genres_exploded['listed_in'] =  genres_exploded['listed_in'].str.strip()
#Recuento de géneros en el catálogo
top_genres = genres_exploded['listed_in'].value_counts().head(10)
print(top_genres)
#Gráfico

plt.figure(figsize=(10,5))
sns.barplot(x=top_genres.values, y=top_genres.index, palette='coolwarm')
plt.title('Top 10 géneros Netflix')
plt.xlabel('Nº de títulos')
plt.ylabel('Género')
plt.show()
#Listado de lanzamientos por mes
monthly_releases = df['month_added'].value_counts().sort_values(ascending=False)
print(monthly_releases)
#Gráfico
plt.figure(figsize=(12,6))
sns.barplot(x=monthly_releases.index,y=monthly_releases.values,palette='magma')
plt.title('Estacionalidad de lanzamientos en Netflix')
plt.xlabel('Mes')
plt.ylabel('Cantidad de lanzamientos')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
#convertir listas de generos a string por título
df['genre_str'] = df['listed_in'].apply(lambda x: ''.join(x))
#Vectorizamos:
#convertimos texto en valores numéricos
vectorizer = CountVectorizer()
X=vectorizer.fit_transform(df['genre_str'])
#Cluster Kmeans
#Agrupamos datos en clusters automáticos
kmeans = KMeans(n_clusters=5,random_state=42)
df['cluster'] = kmeans.fit_predict(X)
#Para ver los titulos de un cluster concreto, por ejemplo el 0:
df[df['cluster'] == 0][['title', 'listed_in']].head(10)
#para ver cuantos títulos hay en cada grupo
df['cluster'].value_counts()
#Reducir dimensiones del cluster para visualizar datos complejos y graficar resultados
pca = PCA(n_components=2)
components = pca.fit_transform(X.toarray())
#Generamos nuevo dataset para poder asignar un color a cada cluster
labels = kmeans.labels_
df_pca = pd.DataFrame(
    components,
    columns=['PC1','PC2'],              # nombres para los ejes
    index=df['cluster'].index             # opcional: hereda el mismo índice
)
df_pca['cluster'] = labels             # añadimos la columna de color


plt.figure(figsize=(10,6))
sns.scatterplot(
    data=df_pca,
    x='PC1', y='PC2',
    hue='cluster',                       # asigna un color por etiqueta
    palette='tab10',
    legend='full'
)
plt.title('Clusters de géneros en el catálogo de Netflix')
plt.show()
#Identificar generos predominantes por cluster
# Explota los géneros
df['genres_list'] = df['listed_in']
genres_clustered = df.explode('genres_list')

# Agrupa por cluster y cuenta géneros más comunes
top_genres_by_cluster = genres_clustered.groupby('cluster')['genres_list'].value_counts().groupby('cluster').head(5)

print(top_genres_by_cluster)
#Hay que unir description,listed_in y type en un solo campo:
df['combined'] = df['type'] + ' ' + df['description'].fillna('') + ' ' + df['genre_str']
#Vectorizamos con TdifVectorizer
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['combined'])
#vamos a usar NearestNeighbor,que permite encontrar, para un punto dado, 
#los puntos más cercanos (similares) en un conjunto de datos.
model = NearestNeighbors(metric='cosine', algorithm='brute')
model.fit(tfidf_matrix)
# Buscar similares a un título específico
idx=35 #suponemos que quermos la similitud del titulo en la fila i, en este caso 35
distances, indices = model.kneighbors(tfidf_matrix[idx], n_neighbors=6)
# Mostrar resultados excluyendo el mismo título
df.iloc[indices[0][1:]][['title', 'description']]
#Mapeo entre títulos y su índice
indices = pd.Series(df.index, index=df['title']).drop_duplicates()

#Función basada en NearestNeighbors
def recomendar(titulo, top_n=5):
    idx = indices[titulo]
    distances, neighbors = model.kneighbors(tfidf_matrix[idx], n_neighbors=top_n+1)  # +1 porque el más cercano es el mismo
    similares = neighbors[0][1:]  # Excluye el propio título
    return df.iloc[similares][['title', 'type', 'listed_in', 'description']]
recomendar('Breaking Bad', top_n=5)
