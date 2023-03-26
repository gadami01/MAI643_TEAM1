import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv("risk_factors_cervical_cancer.csv")
print(data)

labels = data.columns

print("Total Count of null values", data.isna().sum().sum())

data = data.replace("?", np.NaN)
print(data.isna().sum())
plt.title("Null values per feature")
data.isna().sum().plot.bar()
plt.xticks(rotation='vertical')
plt.show()

nullPercentage = data.isnull().sum() * 100 / len(data)
print(nullPercentage)
plt.title("Percentage of null values per feature")
plt.bar(x=labels, height=nullPercentage)
plt.xticks(rotation='vertical')
plt.show()


# percentage of negatives/positives per target variable
hinselmannValues = data["Hinselmann"]
print("Count of Negative Hinselmann tests: ", len(np.where(hinselmannValues==0)[0]))
print("Count of Positive Hinselmann test: ", len(np.where(hinselmannValues==1)[0]))

plt.title("Hinselmann Class imbalance")
plt.pie([len(np.where(hinselmannValues==0)[0]), len(np.where(hinselmannValues==1)[0])], labels=["Negative", "Positive"],
        explode=[0.0, 0.2], autopct='%1.1f%%', colors=["purple", "pink"])
plt.show()

schillerValues = data["Schiller"]
print("Count of Negative Schiller tests: ", len(np.where(schillerValues==0)[0]))
print("Count of Positive Schiller tests: ", len(np.where(schillerValues==1)[0]))

plt.title("Schiller Class imbalance")
plt.pie([len(np.where(schillerValues==0)[0]), len(np.where(schillerValues==1)[0])], labels=["Negative", "Positive"],
        explode=[0.0, 0.2], autopct='%1.1f%%', colors=["orange", "bisque"])
plt.show()

citologyValues = data["Citology"]
print("Count of Negative Citology: ", len(np.where(citologyValues==0)[0]))
print("Count of Positive Citology: ", len(np.where(citologyValues==1)[0]))

plt.title("Citology Class imbalance")
plt.pie([len(np.where(citologyValues==0)[0]), len(np.where(citologyValues==1)[0])], labels=["Negative", "Positive"],
        explode=[0.0, 0.2], autopct='%1.1f%%', colors=["red", "mistyrose"])
plt.show()

biopsyValues = data["Biopsy"]
print("Count of Negative Biopsies: ", len(np.where(biopsyValues==0)[0]))
print("Count of Positive Biopsies: ", len(np.where(biopsyValues==1)[0]))

plt.title("Biopsy Class imbalance")
plt.pie([len(np.where(biopsyValues==0)[0]), len(np.where(biopsyValues==1)[0])], labels=["Negative", "Positive"],
        explode=[0.0, 0.2], autopct='%1.1f%%', colors=["blue", "lightblue"])
plt.show()

data = data.fillna(data.median())
data = data.apply(pd.to_numeric)


# Min and max values per feature
minList = []
maxList = []

for i in data.columns:
    temp = data[i].dropna().values
    minList.append(temp.min())
    maxList.append(temp.max())

minList = [float(val) for val in minList]
maxList = [float(val) for val in maxList]

plt.title("Min and Max values per feature")
plt.xticks(ticks=np.arange(len(minList)),labels=labels, rotation='vertical')
plt.plot(minList)
plt.plot(maxList)
plt.show()


# Outliers plot
outliers_labels = []
outliers_values = []

for column in data.columns:
    col = data[column].dropna().astype(float)

    if not (col.min()==0 and col.max()==1):
        outliers_labels.append(column)
        outliers_values.append(np.array(col))

print(np.array(outliers_values, dtype=object).shape)
print(np.array(outliers_labels, dtype=object).shape)
plt.title("Outliers")
plt.xticks(rotation='vertical')
plt.boxplot(np.array(outliers_values, dtype=object).transpose(), labels=np.array(outliers_labels, dtype=object))

plt.show()

# Correlation map
plt.figure(figsize=(16, 10))
print(data.corr())
heatmap = sns.heatmap(data.corr(), annot=True, cmap='BrBG', annot_kws={
                'fontsize': 6,
                'fontweight': 'bold',
                'fontfamily': 'serif'
            })
heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':10})
plt.show()
