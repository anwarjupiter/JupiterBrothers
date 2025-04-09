import pandas as pd

df = pd.read_csv("input/civil.csv")

"""How many ac are exists in the building ?"""
filtered_df = df.where(df['Building'] == 'Building 1').dropna().where(df['Equipment'] == 'AC').value_counts()
print(filtered_df)