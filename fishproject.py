#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[3]:


df=pd.read_csv(r"C:\Users\dell\Desktop\Realtime-env2.csv")


# In[4]:


df


# In[5]:


from matplotlib import pyplot as plt
df['NITRATE(PPM)'].plot(kind='hist', bins=20, title='NITRATE(PPM)')
plt.gca().spines[['top', 'right',]].set_visible(False)


# In[ ]:





# In[6]:


df.isnull().sum()


# In[7]:


df.drop("label",axis=1,inplace=True)


# In[8]:


df


# In[9]:


df1=df.drop(index=range(74757, 74796),axis=0)


# In[10]:


df1


# In[11]:


fish_conditions = [ df1['NITRATE(PPM)'] >= 100, (df1['PH'] >= 4.8) & (df1['PH'] <= 6.5),
df1['AMMONIA(mg/l)'] >= 0.05, (df1['TEMP'] >= 12) & (df1['TEMP'] <= 15),
df1['DO'] <= 3, df1['TURBIDITY'] <= 12, df1['MANGANESE(mg/l)'] <= 1.0 ]
# Assign 1 if at least two conditions are met, 0 otherwise
df1['FISH_DISEASE'] = (np.sum(fish_conditions, axis=0) >= 2).astype(int)


# In[12]:


fish_conditions


# In[13]:


# df1=pd.DataFrame(fish_conditions)


# In[14]:


df1


# In[ ]:





# 

# In[15]:


# Convert Date and Time columns to datetime objects
df1['DateTime'] = pd.to_datetime(df1['Date'] + ' ' + df1['Time'])
df1 = df1.drop(['Date', 'Time'], axis=1)  # Drop separate Date and Time columns
# # Assuming 'DateTime' column contains timestamps in your X_train and X_test after splitting
# X_train['DateTime'] = pd.to_datetime(X_train['DateTime'])  # Convert to pandas DateTime format
# X_train['DateTime'] = X_train['DateTime'].astype('int64')  # Convert timestamps to numeric representation (Unix time)

# X_test['DateTime'] = pd.to_datetime(X_test['DateTime'])  # Convert to pandas DateTime format
# X_test['DateTime'] = X_test['DateTime'].astype('int64')


# In[16]:


df1.info()


# In[17]:


df1['DateTime'] = df1['DateTime'].astype('int64')


# In[18]:


df1.info()


# In[19]:


X = df1.drop('FISH_DISEASE', axis=1).values


# In[20]:


X


# In[21]:


y=df1["FISH_DISEASE"].values


# In[22]:


y


# In[23]:


from sklearn.model_selection import train_test_split


# In[24]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[25]:


from sklearn.ensemble import RandomForestClassifier

# Initialize the model
model = RandomForestClassifier()


# Train the model
model.fit(X_train, y_train)


# In[26]:


# Make predictions on the test set
predictions = model.predict(X_test)


# In[27]:


predictions


# In[28]:


# Evaluate the model performance
from sklearn.metrics import accuracy_score, classification_report

accuracy = accuracy_score(y_test, predictions)*100
print(f"Accuracy: {accuracy}")


# In[29]:


# Additional evaluation metrics
print(classification_report(y_test, predictions))


# In[30]:


from sklearn.metrics import confusion_matrix
import seaborn as sns


# In[31]:


# Get the confusion matrix
conf_matrix = confusion_matrix(y_test, predictions)


# In[32]:


# Plot confusion matrix
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


# In[33]:


from sklearn.metrics import roc_curve, roc_auc_score


# In[34]:


# Calculate probabilities for ROC curve
probs = model.predict_proba(X_test)
probs = probs[:, 1]  # Consider the positive class probabilities

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, probs)

# Plot ROC curve
plt.plot(fpr, tpr, label=f'AUC = {roc_auc_score(y_test, probs):.2f}')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.show()


# In[35]:


from sklearn.metrics import precision_recall_curve

# Calculate precision and recall
precision, recall, _ = precision_recall_curve(y_test, probs)

# Plot Precision-Recall curve
plt.plot(recall, precision, marker='.')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.show()


# In[ ]:




