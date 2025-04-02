import pandas as pd

# Load the main dataset
data_df = pd.read_csv('cleaned_data_combined_modified.csv')

# Load the split indices
split_df = pd.read_csv('train_test_split.csv')

# Split into training, validation, and test sets
train_indices = split_df[split_df['type'] == 'train']['index']
val_indices = split_df[split_df['type'] == 'val']['index']
test_indices = split_df[split_df['type'] == 'test']['index']

# Create subsets using the indices
train_df = data_df.loc[data_df.index.isin(train_indices)]
val_df = data_df.loc[data_df.index.isin(val_indices)]
test_df = data_df.loc[data_df.index.isin(test_indices)]

# Verify the splits
print(f"Training set size: {len(train_df)}")
print(f"Validation set size: {len(val_df)}")
print(f"Test set size: {len(test_df)}")

# Save the splits to separate files
train_df.to_csv('food_train.csv', index=False)
val_df.to_csv('food_val.csv', index=False)
test_df.to_csv('food_test.csv', index=False)