import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

np.random.seed(42)
n_samples = 500

ids = np.arange(1000, 1000 + n_samples)
years = np.random.randint(1980, 2026, size = n_samples).astype(float)
rooms = np.random.randint(1, 6, size = n_samples)
sizes = np.random.randint(400, 3500, size = n_samples)
sizes_dirty = [f"{s} sqft" for s in sizes]
conditions = np.random.choice(["good", "medium", "bad"], size = n_samples, p=[0.4, 0.4, 0.2])
prices = (sizes * 120) + (rooms * 15000) + ((years - 1980) * 800) + np.random.randint(-10000, 10000, size=n_samples)
missing_years_idx = np.random.choice(n_samples, size=30, replace=False)
years[missing_years_idx] = np.nan
missing_sizes_idx = np.random.choice(n_samples, size=25, replace=False)
for idx in missing_sizes_idx:
    sizes_dirty[idx] = np.nan
df_dirty = pd.DataFrame({
    "id": ids,
    "Year_Built": years,
    "Size(sqft)": sizes_dirty,
    "Rooms": rooms,
    "Condition": conditions,
    "Price_USD": prices
})
df_dirty.to_csv("dirty_house_prices.csv", index=False)


df = pd.read_csv("dirty_house_prices.csv")
print(df.info())
print(df.head(3))


df = df.rename(columns={"Size(sqft)":"Size"})
df = df.rename(columns={"Price_USD": "Price"})
df["Size"] = df["Size"].str.replace("sqft", "")
df["Size"] = pd.to_numeric(df["Size"], errors="coerce")
df["Size"] = df["Size"].fillna(df["Size"].median())
df = df.drop(columns = "id")
df["Year_Built"] = df["Year_Built"].fillna(df["Year_Built"].mean())
df = pd.get_dummies(df, columns=["Condition"], drop_first=True)


X = df.drop(columns=["Price"])
y = df["Price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

pred = model.predict(X_test)
mae = mean_absolute_error(pred, y_test)
print(pred)
print(mae)