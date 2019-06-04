# Generate my dataset, features are the variables, labels are the known outcomes

#########FEATURES#########
# First Feature
weather=['Sunny','Sunny','Overcast','Rainy','Rainy','Rainy','Overcast','Sunny','Sunny',
'Rainy','Sunny','Overcast','Overcast','Rainy']
# Second Feature
temp=['Hot','Hot','Hot','Mild','Cool','Cool','Cool','Mild','Cool','Mild','Mild','Mild','Hot','Mild']

#########LABELS#########
# Label our target varible
play=['No','No','Yes','Yes','Yes','No','Yes','No','Yes','Yes','Yes','Yes','Yes','No']

print(weather)
print(temp)
print(play)

#scikit-learn prefers numbers, the label encoder converts our string data to numbers the library likes

# Import LabelEncoder
from sklearn import preprocessing
#creating labelEncoder
le = preprocessing.LabelEncoder()

#Encode the data
# Converting string labels into numbers (0,1,2)
weather_encoded=le.fit_transform(weather)
temp_encoded=le.fit_transform(temp)
label=le.fit_transform(play)

#zip weather and temp into features variable
features=list(zip(weather_encoded,temp_encoded))

from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=3)
model.fit(features,label)
predicted= model.predict([[0,2]])
print(predicted)
