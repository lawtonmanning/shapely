import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from scipy import stats
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import GradientBoostingClassifier
import matplotlib.pyplot as plt

#read in 100 samples with shapley values
train_data_with_shapley_values = pd.read_csv("data/brunello_train_with_shapley.csv")
test_data = pd.read_csv("data/brunello_test_shap.csv")
num_test = 1000
new_data = test_data[:-num_test]
new_test_data_1000 = test_data[-num_test:]
regr = RandomForestRegressor(max_depth=2, random_state=0)
X = train_data_with_shapley_values.iloc[:,1:-2]
y = train_data_with_shapley_values.iloc[:,-1]
regr.fit(X, y)
predicted_vals = regr.predict(new_data.iloc[:,1:-1])
new_data["shapley vals"] = predicted_vals


#Sort data by shapley values
new_data = new_data.sort_values(by=['shapley vals'])
shapley_values = new_data["shapley vals"]
#new_data.to_csv("Data/new_data.csv",index = False)
print(stats.describe(shapley_values))

#plt.figure()

#plt.style.use('ggplot')
#plt.show
#plt.savefig("4000shap_exp5_hist.pdf")


train_data1 = train_data_with_shapley_values

clf = GradientBoostingClassifier()

#Adding most valuable data points
accuracy_values = []
for i in range(new_data.shape[0]):
    train_data1 = train_data1.append(new_data.iloc[(-i-1),:])
    X = train_data1.iloc[:,1:-2]
    y = train_data1.iloc[:,-2]
    clf.fit(X, y)
    accuracy = clf.score(new_test_data_1000.iloc[:,1:-1],new_test_data_1000.iloc[:,-1])
    #print(accuracy)
    accuracy_values.append(accuracy)

plt.figure()
plt.style.use('ggplot')
plt.xlabel('Number of added training points')
plt.ylabel('Prediction accuracy (%)')
plt.plot(range(0,4000), accuracy_values)
#plt.savefig("brunello_experiment5_add_highvalue.pdf")

train_data1 = train_data_with_shapley_values

clf = GradientBoostingClassifier()

#Adding least valuable data points
accuracy_values2 = []
for i in range(new_data.shape[0]):
    train_data1 = train_data1.append(new_data.iloc[i,:])
    X = train_data1.iloc[:,1:-2]
    y = train_data1.iloc[:,-2]
    clf.fit(X, y)
    accuracy = clf.score(new_test_data_1000.iloc[:,1:-1],new_test_data_1000.iloc[:,-1])
    accuracy_values2.append(accuracy)

plt.figure()
plt.style.use('ggplot')
plt.xlabel('Number of added training points')
plt.ylabel('Prediction accuracy (%)')
plt.plot(range(0,4000), accuracy_values2)
#plt.savefig("brunello_experiment5_add_lowvalue.pdf")
