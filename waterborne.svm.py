#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# In[2]:


data = pd.read_csv(r"file:///C:\Users\customer\OneDrive\Desktop\dataset.csv")
data


# In[3]:


data.isnull().sum()


# In[4]:


data = data.dropna()
data.isnull().sum()


# In[5]:


data.describe()


# In[6]:


waterborne_diseases = ["Cholera","Giardiasis","Hepatitis A","Leptospirosis","Legionnaires' Disease","Cryptosporidiosis","Dysentery","Typhoid Fever"
                       ,"Schistosomiasis","E. coli infection","Amoebiasis","Norovirus"]
data["is_waterborne"] = data["disease_name"].isin(waterborne_diseases).astype(int)
data.head()


# In[7]:


data.dtypes


# In[8]:


data = data.drop(['id','disease_name', 'type', 'causes','infection_status','transmission_mode','treatment','prevention'], axis=1)


# In[9]:


data['fever'] = data['fever'].astype(int)


# In[10]:


data.dtypes


# In[11]:


data.size


# In[12]:


data.shape


# In[13]:


# Check class distribution
print(data['is_waterborne'].value_counts())


# In[14]:


sns.countplot(x='is_waterborne', data=data)
plt.title('Class Distribution')
plt.xlabel('Class')
plt.ylabel('Count')
plt.show()


# In[39]:


# # Select a subset of features to avoid clutter
selected_features = ['symptoms','heart_rate', 'blood_pressure_systolic',
        'blood_pressure_diastolic', 'body_temperature',
        'duration_of_infection', 'is_waterborne']

# # Plot pair plot
def visualize_data():
    sns.pairplot(data[selected_features], hue="is_waterborne", diag_kind="kde")  # correctly indented here

visualize_data()


# In[15]:


X = data.drop('is_waterborne', axis=1)
y = data['is_waterborne']


# In[16]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[17]:


text_column = 'symptoms'
numeric_columns = ['heart_rate', 'blood_pressure_systolic', 'blood_pressure_diastolic','fever', 'body_temperature', 
                   'duration_of_infection']


# In[18]:


# Vectorizer for the text column
text_transformer = TfidfVectorizer()


# In[19]:


# Scaler for numeric columns (optional, good for models like logistic regression)
numeric_transformer = StandardScaler()


# In[20]:


# === Step 6: Combine using ColumnTransformer ===
preprocessor = ColumnTransformer(transformers=[('text', text_transformer, text_column)
                                               ,('num', numeric_transformer, numeric_columns)])


# In[21]:


# Preprocess and transform the data
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)


# In[22]:


# === Step 5: Train Random Forest Model ===
print("--- Training Random Forest ---")
rf_model = RandomForestClassifier()
rf_model.fit(X_train_processed, y_train)


# In[23]:


# Predict and evaluate
y_pred_rf = rf_model.predict(X_test_processed)
print("Random Forest Classification Report:")
print(classification_report(y_test, y_pred_rf))
print()


# In[24]:


y_pred_rf = rf_model.predict(X_test_processed)
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))


# In[25]:


# === Step 6: Train Logistic Regression Model ===
print("--- Training Logistic Regression ---")
lr_model = LogisticRegression()
lr_model.fit(X_train_processed, y_train)


# In[26]:


# Predict and evaluate
y_pred_lr = lr_model.predict(X_test_processed)
print("Logistic Regression Classification Report:")
print(classification_report(y_test, y_pred_lr))
print()


# In[27]:


print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_lr))


# In[28]:


# === Step 7: Train Support Vector Machine (SVM) ===
print("--- Training Support Vector Machine (SVM) ---")
svm_model = SVC()
svm_model.fit(X_train_processed, y_train)


# In[29]:


# Predict and evaluate
y_pred_svm = svm_model.predict(X_test_processed)
print("SVM Classification Report:")
print(classification_report(y_test, y_pred_svm))
print()


# In[30]:


print("SVM Accuracy:", accuracy_score(y_test, y_pred_svm))


# In[41]:


# Display the confusion matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# Generate the confusion matrix
cm = confusion_matrix(y_test, y_pred_svm)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()


# In[32]:


# === Step 4: Combine into a pipeline ===
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', svm_model)
])


# In[33]:


# === Step 9: Train Decision Tree Model ===
print("--- Training Decision Tree ---")
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train_processed, y_train)


# In[34]:


# Predict and evaluate
y_pred_dt = dt_model.predict(X_test_processed)
print("Decision Tree Classification Report:")
print(classification_report(y_test, y_pred_dt))


# In[35]:


print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred_dt))


# In[37]:


import joblib

# Save the model (e.g., RandomForest)
joblib.dump(model, 'svm_pipeline.pkl')


# In[ ]:




