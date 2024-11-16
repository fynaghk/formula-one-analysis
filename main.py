import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix


races = pd.read_csv('races.csv')
drivers = pd.read_csv('drivers.csv')
constructors = pd.read_csv('constructors.csv')
results = pd.read_csv('results.csv')
pit_stops = pd.read_csv('pit_stops.csv')


print(races.head())
print(drivers.head())
print(constructors.head())
print(results.head())
print(pit_stops.head())

#bu verilənlər dəstini birləşdirəcəyik və istifadə edəcəyimiz sütunları ayrı bir cədvəl şəklində yazacağıq
race_results = pd.merge(races, results, on='raceId', how='inner')
data = pd.merge(race_results, drivers, on='driverId', how='inner')
data = data[['raceId', 'year', 'round', 'circuitId', 'constructorId', 'driverId',
             'positionOrder', 'fastestLapSpeed', 'milliseconds', 'points']]

numeric_columns = ['raceId', 'year', 'round', 'circuitId', 'constructorId', 'driverId',
                   'positionOrder', 'fastestLapSpeed', 'milliseconds', 'points']

#seçdiyimiz sütunlarda yerləşən verilənlərin tipini ədədə çeviririk
for col in numeric_columns:
    data[col] = pd.to_numeric(data[col], errors='coerce')

data.fillna(0, inplace=True)

#orta dövrə sürətini hesablayırıq
data['avg_lap_speed'] = data['milliseconds'] / (data['fastestLapSpeed'] * 1000)

#boş xanaları 0-a bərabər edəcəyik
data.fillna(0, inplace=True)

#Korrelyasiya xəritəsini qururuq
plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title("Korrelyasiya")
plt.show()


#modelin  qurulması və qiymətləndirilməsi - verilənləri təlim və sınaq dəstlərinə bölürük
X = data[['year', 'round', 'constructorId', 'fastestLapSpeed', 'stop', 'avg_lap_speed']]
y = data['positionOrder']
X = pd.get_dummies(X, columns=['constructorId'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


#verilmiş parametrlərə əsasən komandanın yarış mövqeyini proqnozlaşdıra bilərik
sample = pd.DataFrame([{
    'year': 2024,
    'round': 20,
    'constructorId': 1,
    'fastestLapSpeed': 220.5,
    'stop': 2,
    'avg_lap_speed': 200.0
}])
sample = pd.get_dummies(sample, columns=['constructorId'])
sample = sample.reindex(columns=X.columns, fill_value=0) 

predicted_position = model.predict(sample)
print(f"Predicted Position: {predicted_position[0]}")






