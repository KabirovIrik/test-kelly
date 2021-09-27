import pandas as pd
import numpy as np
import joblib
import sklearn 

# from sklearn.decomposition import PCA
# from sklearn.preprocessing import OneHotEncoder

from tensorflow import keras

# from sklearn.cluster import KMeans
# открываем преобразователи данных
ohe = joblib.load('data/cat-encoder.joblib')
pca = joblib.load('data/num-pca.joblib')
clustering = joblib.load('data/clusterize.joblib')

pca_columns = ['Component '+str(i) for i in range(pca.n_components_)]
missed_regions = ['Михайловск', 'Заречный', 'Киров']

# открываем модель
model_nn = keras.models.load_model('best_model')


# собственная метрика
def customScorer(y_true, y_pred):
  y_pred = y_pred.flatten()
  n_elems = y_true.shape[0]
  ratio_list = y_true/y_pred
  prefer_ratio = ratio_list[(ratio_list <= 1.2) & (ratio_list >= .8)]
  return prefer_ratio.shape[0]/n_elems

# Получение региона
def getRegionCode(city):
  city_df = city_coords[city_coords['Город'] == city]
  return city_df['Регион'].values[0]
 

# Получени координат
def getCoordinates(city):
  city_df = city_coords[city_coords['Город'] == city]
  return city_df['Широта'].values[0], city_df['Долгота'].values[0]


# Читаем файлы
city_coords = pd.read_csv('data/tableconvert_csv_uladfy.csv', sep=',')
df = pd.read_excel('test.xlsx')

# Не удалось получить данные для всех городов, потому некоторые придется исключить
df = df[~df['CITY'].isin(missed_regions)]

# Списки вещественных и категориальных данных
num_features = [col for col in df.columns if 'NUM' in col]
cat_features = [col for col in df.columns if 'CAT' in col]
encoding_features = cat_features+['CITY', 'REGION', 'CLUSTER']

# добавляем новые значения
df['REGION'] = df['CITY'].apply(getRegionCode)
df['LAT'], df['LNG'] = zip(*df['CITY'].apply(getCoordinates))
df['CLUSTER'] = clustering.predict(df[['LNG', 'LAT']])

# Создаём датасет для предикта
df.reset_index(inplace=True, drop=True)

# One hot encoding
test_ohe = ohe.transform(df[encoding_features])
df_test_ohe = pd.DataFrame(test_ohe, columns=ohe.get_feature_names()).reset_index(drop=True)

# PCA
test_pca = pca.transform(df[num_features])
df_test_pca = pd.DataFrame(test_pca, columns=pca_columns).reset_index(drop=True)

# Всё склеиваем 
X_test = pd.concat([df_test_ohe, df_test_pca], axis=1)

# Получчаем предсказания
predict_values = model_nn.predict(X_test)

# Сохраняем предсказания в файл
pd.DataFrame({'predict_values':predict_values.flatten()}).to_csv('predicted_values.csv', index=False, sep=';')
print('Предсказанные данные сохранены в файл predicted_values.csv \n')

# Выводим первые 10 значений
print('Предсказанные значения (первые 10 штук):')
print(predict_values[:10])