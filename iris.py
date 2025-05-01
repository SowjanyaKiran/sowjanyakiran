#!/usr/bin/env python
# coding: utf-8

# In[28]:


get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import sqlalchemy
import pymysql
import joblib
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*missing ScriptRunContext.*")


# In[18]:


connection = pymysql.connect(host='localhost', user='root', password='sa123', db='iris')

# Query the data
query = "SELECT * FROM iris;"
df = pd.read_sql(query, connection)

# Validate the data structure
print(df.head())
print(df.info())


# In[5]:


df.isnull().sum()
df.duplicated().sum()


# In[6]:


df.describe()


# In[19]:


plt.figure(figsize=(10,6))
sns.histplot(df['SepalLengthCm'], bins=30, kde=True)
plt.title('Sepal Length Distribution')

# Instead of plt.show(), use st.pyplot to render the plot in Streamlit
st.pyplot(plt)


# In[20]:


sns.boxplot(x=df['SepalLengthCm'])
plt.show()


# In[23]:


sns.pairplot(df, hue='Species')
plt.show()


# In[26]:


label_encoder = LabelEncoder()

# Apply label encoding to the species column
df['Species'] = label_encoder.fit_transform(df['Species'])


# In[ ]:





# In[27]:


sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.show()


# In[29]:


#Model Building
X = df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y = df['Species']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[30]:


models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(),
    "XGBoost": xgb.XGBClassifier(eval_metric='mlogloss')
}


# In[31]:


results = {
    "Model": [],
    "Accuracy": [],
    "Precision": [],
    "Recall": [],
    "F1 Score": []
}

# Train and evaluate each model
for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    # Store the results
    results["Model"].append(model_name)
    results["Accuracy"].append(accuracy)
    results["Precision"].append(precision)
    results["Recall"].append(recall)
    results["F1 Score"].append(f1)

# Convert results to a DataFrame
results_df = pd.DataFrame(results)

# Display the results
print(results_df)


# In[32]:


plt.figure(figsize=(10, 6))
sns.barplot(x="Model", y="Accuracy", data=results_df)
plt.title('Model Comparison - Accuracy')
plt.show()

# Plot F1 Score comparison
plt.figure(figsize=(10, 6))
sns.barplot(x="Model", y="F1 Score", data=results_df)
plt.title('Model Comparison - F1 Score')
plt.show()


# In[33]:


best_model = RandomForestClassifier()  # Use the best model based on your analysis
best_model.fit(X_train, y_train)
y_pred_best = best_model.predict(X_test)

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred_best)

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title("Confusion Matrix for Best Model")
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()


# In[34]:


#we will go ahead with randforest classifier
model = RandomForestClassifier()
model.fit(X_train, y_train)


# In[35]:


y_pred = model.predict(X_test)

# Performance Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
conf_matrix = confusion_matrix(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')
print(f'Confusion Matrix:\n{conf_matrix}')


# In[37]:


#Save pkl file for deployment
joblib.dump(model, 'd:\iris_model.pkl')


# In[38]:


joblib.dump(label_encoder, 'd:\\label_encoder.pkl')


# In[ ]:




