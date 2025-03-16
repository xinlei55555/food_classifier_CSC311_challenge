import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def print_csv_info(file_path1, file_path2):
    # Read the CSV files
    df1 = pd.read_csv(file_path1)
    df2 = pd.read_csv(file_path2)
    
    # Print header and first five rows for the first CSV file
    print(f"Header and first five rows of {file_path1}:")
    print(df1.head().to_string())
    
    # Print header and first five rows for the second CSV file
    print(f"\nHeader and first five rows of {file_path2}:")
    print(df2.head().to_string())

if __name__ == '__main__':
    # Example usage
    file_path1 = '/mnt/DATA/0. ÉCOLE/University Time/2. YEAR 2 Documents by Class/2. Second Semester Courses/CSC311/Group Project 1/food_classifier/data/cleaned_data_combined_modified.csv'
    file_path2 = '/mnt/DATA/0. ÉCOLE/University Time/2. YEAR 2 Documents by Class/2. Second Semester Courses/CSC311/Group Project 1/food_classifier/data/cleaned_data_combined.csv'
    print_csv_info(file_path1, file_path2)