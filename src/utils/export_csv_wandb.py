import wandb
import pandas as pd

# Log in to your wandb account
wandb.login()

# Replace 'your_project' and 'your_entity' with your project and entity names
project = 'BioHeat_PINNs'
entity = 'adavit'

# Initialize the API
api = wandb.Api()

# Get the list of runs for the specified project
runs = api.runs(f"{entity}/{project}")

# Initialize an empty list to store run data
runs_data = []

# Iterate over all runs and collect their summary, config, and name
for run in runs:
    summary = run.summary._json_dict
    config = {k: v for k, v in run.config.items() if not k.startswith('_')}
    name = run.name
    state = run.state
    created_at = run.created_at
    user = run.entity
    tags = run.tags
    run_data = {
        'name': name,
        'state': state,
        'created_at': created_at,
        'user': user,
        'tags': tags,
        **config,
        **summary
    }
    runs_data.append(run_data)

# Convert the list of run data to a pandas DataFrame
df = pd.DataFrame(runs_data)

# Save the DataFrame to a CSV file
csv_filename = 'wandb_runs.csv'
df.to_csv(csv_filename, index=False)

print(f"Saved runs data to {csv_filename}")