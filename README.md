# Titanic Dataset — Exploratory Data Analysis (EDA)

## Dataset  
- **Source**: [Titanic Dataset - CSV file in this repository](train.csv)  
- **Rows**: 891 passengers
-  **Columns**: 12 features -  ` PassengerId`, `Survived`, `Pclass`, `Name`, `Sex`, `Age`, `SibSp`, `Parch`, `Ticket`, `Fare`, `Cabin`, `Embarked`.

##  Tools & Libraries  
- **Python**  
- **Pandas** — Data manipulation  
- **Matplotlib** & **Seaborn** — Data visualization  
- **NumPy** — Numerical operations  


## Code & Visuals  
### 1. Survival Counts
```python
plt.figure(figsize=(6,4))
sns.countplot(
    x='Survived',
    data=df_clean,
    hue='Survived',             
    palette={0: "red", 1: "green"},
    dodge=False,              
    legend=False
)
plt.xticks([0, 1], ['Not Survived', 'Survived'])
plt.title('Survival Counts')
plt.show()
```
![Survival Counts](https://github.com/jananiii18/Exploratory-Data-Analysis-EDA-/blob/e701629757b98fc708992a7ff3f1482eb0d02d19/SurvivalCounts.png)
**Observation:** Out of 891 passengers, **38%** survived and **62%** did not. This shows a strong class imbalance, which should be considered if building predictive models.

### 2. Age Distribution
```python
plt.figure(figsize=(8,4))
sns.histplot(
    data=df_clean,
    x='Age',
    bins=30,
    kde=True,
    color="#1f77b4",          
    edgecolor="black",        
    alpha=0.7                 
)
plt.title('Age Distribution of Passengers', fontsize=14, fontweight='bold')
plt.xlabel('Age', fontsize=12)
plt.ylabel('Number of Passengers', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.6) 
sns.despine() 
plt.show()
```
![Age Distribution](https://github.com/jananiii18/Exploratory-Data-Analysis-EDA-/blob/3ff513ff1ee361d8e683b82682c89ebe471b2ebe/AgeDistributionofPassengers.png)
**Observation:** Most passengers were between 20 and 40 years old. There is also a smaller group of children and teenagers, which could influence survival rates.

### 3. Survival by Sex
```python
plt.figure(figsize=(6,4))
sns.countplot(
    x='Sex',
    hue='Survived',
    data=df_clean,
    palette={0: "red", 1: "green"}, 
    edgecolor="black"
)
plt.title('Survival by Sex', fontsize=14, fontweight='bold')
plt.xlabel('Sex', fontsize=12)
plt.ylabel('Number of Passengers', fontsize=12)
plt.legend(title='Survived', labels=['No', 'Yes'])
plt.grid(axis='y', linestyle='--', alpha=0.6)
sns.despine()
plt.show()
```
![Survival by Sex](https://github.com/jananiii18/Exploratory-Data-Analysis-EDA-/blob/3ff513ff1ee361d8e683b82682c89ebe471b2ebe/SurvivalbySex.png)
**Observation:** Females had a survival rate of **74.2%**, compared to only **18.9%** for males. This supports the "women and children first" evacuation policy on the Titanic.

### 4. Survival by Passenger Class
```python
plt.figure(figsize=(6,4))
sns.countplot(
    x='Pclass',
    hue='Survived',
    data=df_clean,
    palette={0: "red", 1: "green"},  
    edgecolor="black"
)
plt.title('Survival by Passenger Class', fontsize=14, fontweight='bold')
plt.xlabel('Passenger Class', fontsize=12)
plt.ylabel('Number of Passengers', fontsize=12)
plt.legend(title='Survived', labels=['No', 'Yes'])
plt.grid(axis='y', linestyle='--', alpha=0.6)
sns.despine()
plt.show()
```
![Survival by class](https://github.com/jananiii18/Exploratory-Data-Analysis-EDA-/blob/f930b265a4b48561a141c7d6e07d71e59c2d2bed/SurvivalbyPassengerClass.png)
**Observation:** 1st Class passengers survived at **62.9%**, while 3rd Class passengers survived at only **24.2%**. Higher-class cabins likely provided faster access to lifeboats.

### 4. Correlation Matrix
```python
num_cols = ['Survived','Pclass','Age','SibSp','Parch','Fare','FamilySize','IsAlone','Sex_n']
num_cols = [c for c in num_cols if c in df_clean.columns]

plt.figure(figsize=(9,6))
corr_matrix = df_clean[num_cols].corr()

sns.heatmap(
    corr_matrix,
    annot=True,             
    fmt='.2f',              
    cmap='RdYlGn',           
    center=0,                
    linewidths=0.5,           
    annot_kws={"size": 10},  
    cbar_kws={'shrink': 0.8} 
)

plt.title('Correlation Matrix (Numeric Features)', fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.show()
```
![Correlation Matrix](https://github.com/jananiii18/Exploratory-Data-Analysis-EDA-/blob/3ff513ff1ee361d8e683b82682c89ebe471b2ebe/CorrelationMatrix.png)
**Observation:** Survival is positively correlated with being female (`Sex_n = 1`, correlation 0.54) and with fare (0.26). It is negatively correlated with passenger class (-0.34) and traveling alone (-0.20). These correlations align with earlier visual findings.

## Executive Summary

- **Gender Effect:** Females had a much higher survival rate (**74.2%**) compared to males (**18.9%**), likely due to "women and children first" evacuation protocols.
- **Class Effect:** 1st Class passengers survived at **62.9%**, while 3rd Class survival was only **24.2%**,showing that higher-class cabins had better lifeboat access.
- **Age Effect:** Children and younger passengers had higher survival chances, while middle-aged passengers had lower odds.
- **Fare Effect:** Survivors generally paid much higher fares on average **`~$48`** compared to **`~$22`** for non-survivors.This notable gap suggests a socio-economic survival advantage.
- **Travel Group Effect:** Passengers traveling with family had slightly better survival rates than those traveling alone, supported by the negative correlation of **-0.20** for the `IsAlone` feature.
- **Correlation Insights:** Survival is positively correlated with being female (`Sex_n = 1`, **0.54**) and with fare (**0.26**),and negatively correlated with passenger class (**-0.34**) and traveling alone (**-0.20**).

**Conclusion:** The analysis confirms that socio-economic status,gender,age,and travel group size were key factors influencing survival on the Titanic, with wealth and cabin class playing a major role.
