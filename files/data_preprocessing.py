import pandas as pd
import numpy as np

df = pd.read_csv('./data/source/train.csv')

def clean_data(df):
    # Drop initial features
    features_to_drop = ['PassengerId', 'Ticket', 'Name']
    df.drop(features_to_drop, axis=1, inplace=True)
    # Cabin Manipulations
    cabin_func = lambda x: str(x[0]) if not pd.isnull(x) else np.NaN
    df['cabin_new'] = df['Cabin'].apply(cabin_func)
    df.drop(['Cabin'], axis=1, inplace=True)
    df = pd.concat([df ,pd.get_dummies(df['cabin_new'], prefix='Cabin')], axis=1)
    # Sex mapping
    Sex_map = lambda x: 1 if str(x) == 'female' else 0
    df['Sex'] = df['Sex'].apply(Sex_map)
    # Embarked Manipulations
    df = pd.concat([df ,pd.get_dummies(df['Embarked'], prefix='Embarked')], axis=1)
    # Fare Manipulations
    df['Fare_Round'] = df['Fare'].apply(np.around)
    df.drop(['cabin_new', 'Fare', 'Embarked'], axis=1, inplace=True)
    return df
