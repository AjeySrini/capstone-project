
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv("2. Cars Data1.csv")
print(df.head())
print(df.dtypes)
print(df.shape)

nulls=pd.DataFrame({
    "Null Count":df.isnull().sum(),
    "Null %":df.isnull().mean()*100
})
print(nulls)

for c in ["MSRP","Invoice"]:
    if c in df.columns:
        df[c]=df[c].astype(str).str.replace("[$,]","",regex=True)
        df[c]=pd.to_numeric(df[c],errors="coerce")

for c in df.select_dtypes(include="object").columns:
    if df[c].nunique()/len(df)<0.5:
        df[c]=df[c].astype("category")

num=df.select_dtypes(include="number").columns
for c in num:
    if df[c].isnull().mean()<0.2:
        df[c]=df[c].fillna(df[c].median())

dups=df.duplicated().sum()
print("Duplicates:",dups)
df=df.drop_duplicates()

print(df.describe())
sk=df[num].skew().sort_values(key=abs,ascending=False)
print(sk)

for c in num[:2]:
    q1,q3=df[c].quantile([0.25,0.75]);iqr=q3-q1
    lb=q1-1.5*iqr;ub=q3+1.5*iqr
    print(c,((df[c]<lb)|(df[c]>ub)).sum())

plt.figure();df[num[0]].plot();plt.title("Line");plt.savefig("line_plot.png")
plt.figure();df.groupby(df.select_dtypes(exclude='number').columns[0])[num[0]].mean().plot.bar();plt.tight_layout();plt.savefig("bar_plot.png")
plt.figure();sns.histplot(df[sk.index[0]],bins=20);plt.savefig("histogram.png")
plt.figure();sns.scatterplot(x=df[num[0]],y=df[num[1]]);plt.savefig("scatter.png")
plt.figure();sns.boxplot(x=df.select_dtypes(exclude='number').columns[0],y=num[0],data=df);plt.xticks(rotation=45);plt.tight_layout();plt.savefig("boxplot.png")
plt.figure(figsize=(10,8));sns.heatmap(df[num].corr(),annot=True,cmap="coolwarm");plt.tight_layout();plt.savefig("heatmap.png")

pear=df[num].corr()
spear=df[num].corr(method="spearman")
diff=(spear-pear).abs().stack().reset_index()
diff.columns=["Var1","Var2","Diff"]
diff=diff[diff.Var1!=diff.Var2].sort_values("Diff",ascending=False)
print(diff.head(3))

cat=df.select_dtypes(exclude="number").columns[0]
grp=df.groupby(cat)[num[0]].agg(["mean","std","count"])
print(grp)

df.to_csv("cleaned_data.csv",index=False)
