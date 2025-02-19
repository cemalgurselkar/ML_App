import pandas as pd

df = pd.read_excel("Spinal-extention(ex5).xlsx")
df_sample = df.sample(10)
df_sample.to_excel("random_10_samples.xlsx", index=False)