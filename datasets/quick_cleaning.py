import pandas as pd

def split_csv(input_csv: str):
    # Load the CSV file
    df = pd.read_csv(input_csv)
    
    # Shuffle the data (excluding the header)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Separate the last column
    labels = df.iloc[:, -1]
    data_without_labels = df.iloc[:, :-1]
    
    # Write the last column to labels.csv
    labels.to_csv("labels.csv", index=False)
    
    # Write the modified data to data_without_labels.csv
    data_without_labels.to_csv("data_without_labels.csv", index=False)
    
    print("Files saved: labels.csv and data_without_labels.csv")

if __name__ == '__main__':
    split_csv('cleaned_data_combined_modified.csv')