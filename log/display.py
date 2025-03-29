import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

file_path = "random_forest_no_chaining.csv"
df = pd.read_csv(file_path)

# Ensure expected columns exist
expected_columns = {"criterion", "max_depth", "min_samples_split", "valid_acc"}
if not expected_columns.issubset(df.columns):
    raise ValueError(f"CSV file must contain columns: {expected_columns}")

# Convert categorical columns to strings (for consistent grouping)
df["criterion"] = df["criterion"].astype(str)
df["min_samples_split"] = df["min_samples_split"].astype(str)

# Set up the plot
plt.figure(figsize=(10, 6))
sns.set(style="whitegrid")

# Plot accuracy for different criterion
# for criterion in df["criterion"].unique():
criterion = 'log_loss'
subset = df[df["criterion"] == criterion]
sns.lineplot(
    data=subset,
    x="max_depth",
    y="valid_acc",
    hue="min_samples_split",
    marker="o",
)

plt.xlabel("Max Depth")
plt.ylabel("Validation Accuracy")
plt.title(f"Random Forest Validation Accuracy vs. Max Depth (Criterion = {criterion})")
plt.legend(title="Min Samples Split")
plt.show()
