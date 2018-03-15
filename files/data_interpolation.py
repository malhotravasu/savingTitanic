import pandas as pd
import numpy as np

def interpolate_data(df):
    medians = []
    for i in range(1,4):
        temp = df[df.Pclass == i]
        medians.append(temp.Age.median())
    new_age = []
    for i in range(len(df)):
        if pd.isna(df.iloc[i, 3]):
            new_age.append(medians[(df.iloc[i, 1]) - 1])
        else:
            new_age.append(df.iloc[i, 3])
    df['new_age'] = pd.Series(new_age)
    df.drop(['Age'], axis=1, inplace=True)
    return df
