import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
def load_data(file_path):
    return pd.read_csv(file_path)
def preprocess_data(df):
    df=df.copy()
    y=df['TARGET']
    x=df.drop(columns=['TARGET'],axis=1)
    num_col= x.select_dtypes(include=['int64','float64']).columns
    cat_col= x.select_dtypes(include=['object']).columns
    num_transformer=Pipeline(steps=[
        ('imputer',SimpleImputer(strategy='mean')),
        ('scaler',StandardScaler())
    ])
    cat_transformer=Pipeline(steps=[
        ('imputer',SimpleImputer(strategy='most_frequent')),
        ('encoder',OneHotEncoder(handle_unknown='ignore'))
    ])
    preprocessor=ColumnTransformer(
        transformers=[
            ('num',num_transformer,num_col),
            ('cat',cat_transformer,cat_col)
        ]
    )
    x_transformed=preprocessor.fit_transform(x)
    return x_transformed, y, preprocessor
def split_data(x, y):
    return train_test_split(x,y,test_size=0.2,random_state=42)
