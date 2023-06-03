# Importing the necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn import svm
from imblearn.over_sampling import ADASYN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, accuracy_score
from sklearn.model_selection import GridSearchCV

# Loading the data
df = pd.read_csv("F:\\capstone\\cancer.csv")
print(df.head())

# Find the shape of data
print(df.shape)

# Find the descriptive stats of the data
print(df.describe().T)

# Check for null and datatype of each feature
print(df.info())

# Drop the unwanted features
df.drop(['id', 'Unnamed: 32'], axis=1, inplace=True)

# Count the target variables
print(df['diagnosis'].value_counts())

# Visualize the count
sns.countplot(data=df, x='diagnosis')

# Transform target to integer
le = LabelEncoder()
df['diagnosis'] = le.fit_transform(df['diagnosis'])

# Check the distribution of data w.r.t some features
sns.pairplot(df.iloc[:, 1:11])
plt.show()

# Remove outliers using OneClassSVM algorithm


def handle_outlier(daf):
    out_model = svm.OneClassSVM(nu=0.05)
    out_model.fit(daf)
    outlier_score = out_model.predict(daf)
    return daf[outlier_score == 1]


# Storing data without outliers
df_wo_outliers = handle_outlier(df)
# df_wo_outliers = df[outlier_score == 1]
print(df_wo_outliers)

# Apply ADASYN for over sampling the data
adasyn = ADASYN()
X_resampled, y_resampled = adasyn.fit_resample(df_wo_outliers.iloc[:, 1:], df_wo_outliers.iloc[:, 0])

# Check whether the data balanced or not
print(np.bincount(y_resampled))

# Get the correlation of the features
df.iloc[:, 1:].corr()

# Visualize the correlation
plt.figure(figsize=(20, 20))
sns.heatmap(df.iloc[:, 1:].corr(), annot=True, fmt='.0%')
plt.show()

# Apply standardization using Standardscaler
scaler = StandardScaler()
x_scaled = scaler.fit_transform(X_resampled)

# Apply PCA for feature extraction
pca = PCA(n_components=10)
transformed_x = pca.fit_transform(x_scaled)

# Prepare data for model building
X_train, X_test, y_train, y_test = train_test_split(transformed_x, y_resampled, test_size=0.3, random_state=42)

# Train the SVC model
svc = SVC(probability=True)
svc.fit(X_train, y_train)

# Evaluating the prediction
y_predict = svc.predict(X_test)
cm = confusion_matrix(y_test, y_predict)
print(cm)
print(classification_report(y_test, y_predict))
sns.heatmap(cm, annot=True)
plt.show()

# Check the roc-auc curve
fpr, tpr, thresholds = roc_curve(y_test, y_predict)
auc_curve = auc(fpr, tpr)
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % auc_curve)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()

# Hyperparameter tuning
param_grid = {'C': [10, 15, 20, 40, 70, 75],
              'gamma': [0.1, 0.01, 0.001, 0.03, 0.035],
              'kernel': ['rbf', 'linear']}
grid_search = GridSearchCV(svc, param_grid, cv=5)
final_model = grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_
best_param = grid_search.best_params_
y_predict = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_predict)
print("Best Parameters:", best_param)
print("Test Accuracy:", accuracy)

pickle.dump(final_model, open('model.pkl', 'wb'))
# pickle.dump(scaler, open('scale.pkl', 'wb'))
# pickle.dump(pca, open('pca.pkl', 'wb'))