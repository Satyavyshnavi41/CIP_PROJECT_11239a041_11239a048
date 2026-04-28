import warnings
warnings.filterwarnings("ignore")

import pandas as pd

df = pd.read_csv("data/US_Accidents.csv", nrows=5000)

df = df[[
    'Severity',
    'Temperature(F)',
    'Humidity(%)',
    'Pressure(in)',
    'Visibility(mi)',
    'Wind_Speed(mph)',
    'Weather_Condition',
    'Sunrise_Sunset'
]]

df = df.dropna()
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

plt.figure()
sns.countplot(x='Severity', data=df)
plt.title("Severity Distribution")
plt.show()

plt.figure()
df['Weather_Condition'].value_counts().head(10).plot(kind='bar')
plt.title("Weather Conditions")
plt.show()

plt.figure()
sns.countplot(x='Sunrise_Sunset', data=df)
plt.title("Day vs Night")
plt.show()

from sklearn.preprocessing import LabelEncoder

le1 = LabelEncoder()
le2 = LabelEncoder()

df['Weather_Condition'] = le1.fit_transform(df['Weather_Condition'])
df['Sunrise_Sunset'] = le2.fit_transform(df['Sunrise_Sunset'])
X = df.drop('Severity', axis=1)
y = df['Severity'].apply(lambda x: 1 if x >= 3 else 0)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

from sklearn.metrics import accuracy_score

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

import pickle
pickle.dump(model, open("model.pkl", "wb"))

print("Model saved successfully")