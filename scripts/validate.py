import pandas as pd
import os
import re
from tqdm import tqdm
import process
    
def check_unique(dataframe, column_name, column_label=None):
    """
    Check if all values in the specified column of the DataFrame are unique.

    :param dataframe: Pandas DataFrame
    :param column_name: Name of the column to check for uniqueness
    :return: A tuple containing the description, result status, and count of violations
    """
    if column_name not in dataframe.columns:
        raise ValueError(f"Column '{column_label}' not found in DataFrame")

    # Use the custom column label if provided, otherwise use the column name
    label = column_label if column_label else column_name

    description = f"{label} is unique"
    is_unique = dataframe[column_name].is_unique
    result = 'success' if is_unique else 'failed'
    
    # Count non-unique values (excluding the first occurrence)
    violation = dataframe[column_name].duplicated().sum() if not is_unique else 0

    return description, result, violation

def check_missing(dataframe, column_name, column_label=None):
    """
    Check if there are any null (missing) values in the specified column of the DataFrame more than 5%.

    :param dataframe: Pandas DataFrame
    :param column_name: Name of the column to check for null values
    :param column_label: Optional custom label for the column to use in print statements
    :return: A tuple with description, result, and number of violations
    """
    if column_name not in dataframe.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame")
    
    # Use the custom column label if provided, otherwise use the column name
    label = column_label if column_label else column_name

    percent_missing = dataframe[column_name].isnull().sum() / len(dataframe)
    description = f"Missing values in {label} are less than 5%"
    result = 'failed' if percent_missing > 0.05 else 'success'
    
    # Count missing values if the check failed
    violation = dataframe[column_name].isnull().sum() if result == 'failed' else 0

    return description, result, violation

# def check_range(dataframe, column_name, column_label=None, min_val=None, max_val=None, include_limit=False):
#     """
#     Check if non-null values in the specified column of the DataFrame meet the specified range criteria.
#     Min and Max values can be numbers or names of other columns in the DataFrame, applied row-wise.

#     :param dataframe: Pandas DataFrame
#     :param column_name: Name of the column to check
#     :param column_label: Optional label for reporting
#     :param min_val: Minimum value for the range or the name of the column to use as minimum
#     :param max_val: Maximum value for the range or the name of the column to use as maximum
#     :param include_limit: Boolean to include the limits in comparison
#     :return: A tuple containing the description, result status, and count of violations
#     """
#     if column_name not in dataframe.columns:
#         raise ValueError(f"Column '{column_label}' not found in DataFrame")

#     df_compare = dataframe.copy()

#     # Determine actual min and max values dynamically
#     if min_val in dataframe.columns:
#         df_compare = df_compare[[column_name, min_val]]
#     else:
#         min = min_val
        
#     if max_val in dataframe.columns:
#         df_compare[column_name, max_val].dropna()
#     else:
#         min = min_val
        
#     if min_val is not None and max_val is not None:
#         comparison_method = .....
#         description = f"{column_label} is between {min} and {max}, inclusive: {include_limit}"
#     elif min_val is not None:
#         comparison_method = .....
#         description = f"{column_label} is greater than {'or equal to ' if include_limit else ''}{min}"
#     elif max_val is not None:
#         comparison_method = .....
#         description = f"{column_label} is less than {'or equal to ' if include_limit else ''}{max}"
#     else:
#         raise ValueError("At least one of min or max must be provided")
    

#     # Apply the range check row-wise
#     results = ......
#     is_range = results.all()
#     result = 'success' if is_range else 'failed'
#     violation = (~results).sum()

#     return description, result, violation

def validate(df, validate_dict_path, df_name="DataFrame", print_report=False):
    
    # Import validate dict
    validate_dict = pd.read_excel(validate_dict_path, sheet_name="validate")
    
    validate_report = {
        "Description": [],
        "Type": [],
        "Total violations": [] 
    }
    
    # Iterate over each row in the validation dictionary
    for _, row in tqdm(validate_dict.iterrows(), total=validate_dict.shape[0], desc="Validating"):
        column = row['Column Name']
        label = row['Column Label']
        
        if row['Required'] == True:
            description, result, violation = check_missing(df, column, label)
            validate_report["Description"].append(description)
            validate_report["Type"].append(result)
            if result == 'failed':
                violation_percentage = (violation / len(df)) * 100  # Calculate the percentage
                validate_report["Total violations"].append(f"{violation} ({violation_percentage:.2f}%)")
            else:
                validate_report["Total violations"].append("")  # Blank for success
        
        if row.get('Unique') == True:
            description, result, violation = check_unique(df, column, label)
            validate_report["Description"].append(description)
            validate_report["Type"].append(result)
            if result == 'failed':
                violation_percentage = (violation / len(df)) * 100  # Calculate the percentage
                validate_report["Total violations"].append(f"{violation} ({violation_percentage:.2f}%)")
            else:
                validate_report["Total violations"].append("")  # Blank for success
        
        if not pd.isna(row['Min']) or not pd.isna(row['Max']):
            
            min = None if pd.isna(row['Min']) else row['Min']
            max = None if pd.isna(row['Max']) else row['Max']
            include_limit = input(f"Do you want to include the limit for {column} (True/False): ")
            description, result, violation = check_range(df, column, label, min, max, include_limit)
            validate_report["Description"].append(description)
            validate_report["Type"].append(result)
            if result == 'failed':
                non_null_count = len(df[column].dropna())
                if non_null_count > 0:
                    violation_percentage = (violation / non_null_count) * 100
                    validate_report["Total violations"].append(f"{violation} ({violation_percentage:.2f}%)")
                else:
                    validate_report["Total violations"].append("No data to validate")
            else:
                validate_report["Total violations"].append("")  # Blank for success

    # Create DataFrame from the report
    report_df = pd.DataFrame(validate_report)

    # Count the number of successes and failures
    success_count = (report_df['Type'] == 'success').sum()
    failed_count = (report_df['Type'] == 'failed').sum()

    # Print the summary
    print(f"\n---- Validation report for {df_name} ---------------------------\n")
    print(f"Total records: \t{len(df)}")
    print(f"Total columns: \t{len(df.columns)}")
    print("\n---------------------------------------------------------------------\n")
        
    print(f"Validation criteria: {len(report_df)}")
    print(f"Success: {success_count}")
    print(f"Failed: {failed_count}")
    print("\n---------------------------------------------------------------------\n")
    
    # Count and print the data types
    data_type_counts = df.dtypes.value_counts()
    print("Data Types in DataFrame:")
    for dtype, count in data_type_counts.items():
        print(f"\t{dtype}: \t{count}")

    if print_report:
        print("\n---------------------------------------------------------------------\n")
        print("List of failed validation")
        print(report_df[report_df['Type'] == 'failed'])
    
    return report_df

        
        
            
        
        