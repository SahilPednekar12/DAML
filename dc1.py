import pandas as pd


df = pd.read_csv("demo1.csv")

# 2. Drop missing values
drop_rows_any = df.dropna(axis=0, how='any')   
drop_rows_all = df.dropna(axis=0, how='all')    
drop_cols_any = df.dropna(axis=1, how='any')   
drop_cols_all = df.dropna(axis=1, how='all')   

drop_rows_any.to_csv("dropna_rows_any.csv", index=False)
drop_rows_all.to_csv("dropna_rows_all.csv", index=False)
drop_cols_any.to_csv("dropna_cols_any.csv", index=False)
drop_cols_all.to_csv("dropna_cols_all.csv", index=False)


# 3. Fill missing values
fill_default = df.fillna("Missing")                            
fill_ffill = df.fillna(method='ffill')                         
fill_bfill = df.fillna(method='bfill')                         
fill_mean = df.fillna(df.mean(numeric_only=True))               
fill_max = df.fillna(df.max(numeric_only=True))                 

fill_default.to_csv("fillna_default.csv", index=False)
fill_ffill.to_csv("fillna_ffill.csv", index=False)
fill_bfill.to_csv("fillna_bfill.csv", index=False)
fill_mean.to_csv("fillna_mean.csv", index=False)
fill_max.to_csv("fillna_max.csv", index=False)

# 4. Check missing values
print("Missing values in each column (any):\n", df.isna().any(axis=0))
print("\nAll values missing in column:\n", df.isna().all(axis=0))
print("\nMissing values in each row (any):\n", df.isna().any(axis=1))
print("\nAll values missing in row:\n", df.isna().all(axis=1))

# 5. Normalization / Scaling
numeric_cols = df.select_dtypes(include=['number'])

# Min-Max Scaling
minmax_scaled = numeric_cols.apply(lambda x: (x - x.min()) / (x.max() - x.min()))
minmax_scaled.to_csv("scaled_minmax.csv", index=False)

# Z-Score Normalization
zscore_scaled = numeric_cols.apply(lambda x: (x - x.mean()) / x.std())
zscore_scaled.to_csv("scaled_zscore.csv", index=False)

print("\nAll operations completed successfully!")
