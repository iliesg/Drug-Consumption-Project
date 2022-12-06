import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle
from sklearn.svm import SVC
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.model_selection import GridSearchCV

# Load the csv file
df = pd.read_csv("drug_consumption_ml.csv")

print(df.head())

# Select independent and dependent variable

features = ['Age', 'Gender', 'Education', 'Country', 'Ethnicity',
            'Nscore', 'Escore', 'Oscore', 'Ascore', 'Cscore', 'Impulsive', 'SS']

X = df.loc[:, features]
y = df["Drug_user"]

# Split the dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=50)

# Feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# grid parameters
param_grid_Random = [{'class_weight': ['balanced'], 'criterion': [
    'gini'], 'max_depth': [16], 'max_features': ['auto'], 'n_estimators': [128]}]


# Instantiate the model
scv = StratifiedKFold(n_splits=5)

classifier = GridSearchCV(RandomForestClassifier(
), param_grid=param_grid_Random, scoring='f1', cv=scv, verbose=True, n_jobs=-1)

# Fit the model
classifier.fit(X_train, y_train)

# Make pickle file of our model
pickle.dump(classifier, open("model.pkl", "wb"))
