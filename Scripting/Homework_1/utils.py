from sklearn.preprocessing import OrdinalEncoder

def encode_age_category(data_df):
    labels = ['Sub-Junior', 'Junior', 'Open', 'Masters 1', 'Masters 2', 'Masters 3', 'Masters 4']
    ordinal_encoder = OrdinalEncoder(categories=[labels])
    data_df['AgeCategoryEncoded'] = ordinal_encoder.fit_transform(data_df[['AgeCategory']])
    return data_df