
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

data = pd.read_csv("ec2dataset.csv")
data.columns = data.columns.str.strip()

print("Dataset Information:")
data.info()

print("\nFirst 5 Rows:")
print(data.head())

cost_columns = [
    "On Demand",
    "Linux Reserved cost",
    "Linux Spot Minimum cost",
    "Windows On Demand cost",
    "Windows Reserved cost"
]

for column in cost_columns:
    data[column] = pd.to_numeric(
        data[column]
        .astype(str)
        .str.replace(r"[$,]", "", regex=True)
        .str.replace("hourly", "", regex=False)
        .str.strip(),
        errors="coerce"
    )

print("\nMissing Values in Cost Columns:")
print(data[cost_columns].isnull().sum())

cost_summary = data[cost_columns].describe()
print("\nCost Summary:")
print(cost_summary)

sns.set_theme(style="whitegrid")

plt.figure(figsize=(12, 6))
sns.boxplot(data=data[cost_columns], palette="Set2")
plt.title("Cost Comparison of Amazon EC2 Instances (Hourly)")
plt.ylabel("Cost (USD)")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

def detect_outliers(column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    return data[
        (data[column] < lower_bound) |
        (data[column] > upper_bound)
    ]

outliers_on_demand = detect_outliers("On Demand")

print("\nOn-Demand Cost Outliers:")
print(outliers_on_demand)

cost_comparison = (
    data[["Name", "On Demand", "Linux Reserved cost"]]
    .dropna()
    .sort_values("On Demand")
)

print("\nTop 10 Lowest-Cost Instances:")
print(cost_comparison.head(10))

def filter_instance_family(family):
    return data[
        data["Name"]
        .astype(str)
        .str.upper()
        .str.startswith(family.upper())
    ]

t2_instances = filter_instance_family("T2")
t3_instances = filter_instance_family("T3")

print("\nT2 Instance Costs Summary:")
print(t2_instances[cost_columns].describe())

print("\nT3 Instance Costs Summary:")
print(t3_instances[cost_columns].describe())

if not t2_instances.empty:
    plt.figure(figsize=(12, 6))
    sns.boxplot(
        data=t2_instances[cost_columns],
        palette="Blues",
        showmeans=True
    )
    plt.title("Cost Distribution for T2 Instances")
    plt.ylabel("Cost (USD)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()
else:
    print("No T2 instances found.")

if not t3_instances.empty:
    plt.figure(figsize=(12, 6))
    sns.boxplot(
        data=t3_instances[cost_columns],
        palette="Greens",
        showmeans=True
    )
    plt.title("Cost Distribution for T3 Instances")
    plt.ylabel("Cost (USD)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()
else:
    print("No T3 instances found.")

comparison = pd.concat([
    t2_instances[
        ["Name", "On Demand", "Linux Reserved cost"]
    ],
    t3_instances[
        ["Name", "On Demand", "Linux Reserved cost"]
    ]
])

comparison_sorted = (
    comparison.dropna()
    .sort_values("On Demand")
)

print("\n10 Lowest-Cost T2/T3 Instances:")
print(comparison_sorted.head(10))

data["Instance Memory"] = pd.to_numeric(
    data["Instance Memory"]
    .astype(str)
    .str.replace(",", "", regex=False)
    .str.extract(r"(\d+(?:\.\d+)?)", expand=False),
    errors="coerce"
)

data["vCPUs"] = pd.to_numeric(
    data["vCPUs"]
    .astype(str)
    .str.extract(r"(\d+)", expand=False),
    errors="coerce"
)

print("\nCleaned Memory and CPU Data:")
print(data[["Instance Memory", "vCPUs"]].head())

data_cleaned = data.dropna(
    subset=["On Demand", "Instance Memory", "vCPUs"]
)

print("\nMissing Values After Cleaning:")
print(
    data_cleaned[
        ["On Demand", "Instance Memory", "vCPUs"]
    ].isnull().sum()
)

X = data_cleaned[["Instance Memory", "vCPUs"]]
y = data_cleaned["On Demand"]

if len(data_cleaned) < 5:
    raise ValueError(
        "Not enough valid rows for training and testing."
    )

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print(f"\nTraining samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

model = LinearRegression()
model.fit(X_train, y_train)

print("\nModel Information:")
print(f"Intercept: {model.intercept_}")
print(f"Coefficients: {model.coef_}")

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5

print("\nModel Performance:")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")

plt.figure(figsize=(8, 6))

plt.scatter(
    y_test,
    y_pred,
    alpha=0.7,
    color="blue"
)

min_value = min(y_test.min(), y_pred.min())
max_value = max(y_test.max(), y_pred.max())

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    color="red",
    linestyle="--"
)

plt.title("Actual vs Predicted On-Demand Costs")
plt.xlabel("Actual On-Demand Cost (USD/hour)")
plt.ylabel("Predicted On-Demand Cost (USD/hour)")
plt.tight_layout()
plt.show()

new_instance = pd.DataFrame({
    "Instance Memory": [4],
    "vCPUs": [2]
})

predicted_cost = model.predict(new_instance)

print(
    f"\nPredicted On-Demand Cost for "
    f"4 GiB, 2 vCPUs: ${predicted_cost[0]:.4f}/hour"
)
