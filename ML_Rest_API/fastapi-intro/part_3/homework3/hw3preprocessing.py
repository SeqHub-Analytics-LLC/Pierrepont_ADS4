from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
def preprocesing():
    print("Hello World")
    df = pd.read_csv('breast-cancer.data', delimiter=',', 
                    names=['Class', 'age', 'menopause', 'tumor-size', 'inv-nodes', 
                            'node-caps', 'deg-malig', 'breast', 'breast-quad', 'irradiat'])

    print("Dataset shape:", df.shape)
    print("\nData types:\n", df.dtypes)
    print("\nMissing values:\n", df.isnull().sum()) 
    df.head()
    print("Missing values", df.isnull().sum())

    X = df.drop('Class', axis=1)
    y = df['Class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                        random_state=42, 
                                                        stratify=y)
    ordinal_encoded_columns=['age','tumor-size']
    age_categories = df['age'].unique()
    tumor_categories= df['tumor-size'].unique()
    encoder= OrdinalEncoder(categories= ordinal_encoded_columns)
    #df['']= encoder.
    print(tumor_categories)

if __name__ == "__main__":
    preprocesing()
