import pandas as pd

def split_at_second_occurrence(string, delimiter):
    parts = string.split(delimiter, 2)
    if len(parts) < 3:
        return string  # If there are less than 3 parts, return the original string
    return delimiter.join(parts[:2])

def normalize_and_compute_composite_score(csv_file_path, output_file_path, weights):
    # Load the CSV file
    data = pd.read_csv(csv_file_path)
    
    # Normalize each metric using min-max normalization
    data['L2RE_norm'] = (data['L2RE'] - data['L2RE'].min()) / (data['L2RE'].max() - data['L2RE'].min())
    data['MAE_norm'] = (data['MAE'] - data['MAE'].min()) / (data['MAE'].max() - data['MAE'].min())
    data['MSE_norm'] = (data['MSE'] - data['MSE'].min()) / (data['MSE'].max() - data['MSE'].min())
    data['max_APE_norm'] = (data['max_APE'] - data['max_APE'].min()) / (data['max_APE'].max() - data['max_APE'].min())

    # Calculate the composite score using the normalized metrics
    data['total_metrics'] = (
        weights['L2RE'] * data['L2RE_norm'] +
        weights['MAE'] * data['MAE_norm'] +
        weights['MSE'] * data['MSE_norm'] +
        weights['max_APE'] * data['max_APE_norm']
    )

    # Drop the intermediate normalized columns if not needed
    data.drop(columns=['L2RE_norm', 'MAE_norm', 'MSE_norm', 'max_APE_norm'], inplace=True)

    # Save the updated dataframe to a new CSV file
    data.to_csv(output_file_path, index=False)
    print(f"Normalized metrics and composite scores saved to {output_file_path}")

# Define weights
weights = {
    'L2RE': 0.3,
    'MAE': 0.3,
    'MSE': 0.2,
    'max_APE': 0.2
}

# File paths
csv_file_path = './src/utils/csv_results/wandb_runs_1.csv'
output_file_path = split_at_second_occurrence(csv_file_path, '.') + '_composite_score.csv'

# Run the normalization and composite score computation
normalize_and_compute_composite_score(csv_file_path, output_file_path, weights)