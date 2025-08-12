### Titanic Dataset — Exploratory Data Analysis (EDA)

## Dataset  
- **Source**: [Titanic Dataset - CSV file in this repository](train.csv)  
- **Rows**: 891 passengers
-  **Columns**: 12 features -  ` PassengerId`, `Survived`, `Pclass`, `Name`, `Sex`, `Age`, `SibSp`, `Parch`, `Ticket`, `Fare`, `Cabin`, `Embarked`.

##  Tools & Libraries  
- **Python**  
- **Pandas** — Data manipulation  
- **Matplotlib** & **Seaborn** — Data visualization  
- **NumPy** — Numerical operations  


### Code & Visuals  
## 1. Survival Counts
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
plt.title('Survival Counts', fontsize=14, fontweight='bold')
plt.xlabel('Survival Status', fontsize=12)
plt.ylabel('Number of Passengers', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.6)
sns.despine()
plt.show()
```
![Survival Counts](https://github.com/jananiii18/Exploratory-Data-Analysis-EDA-/blob/e701629757b98fc708992a7ff3f1482eb0d02d19/SurvivalCounts.png)
