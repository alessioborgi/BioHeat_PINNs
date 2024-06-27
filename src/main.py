import utils
import datetime
import os

# Set seed for reproducibility
utils.seed_all(31)

print(f"Current working directory: {os.getcwd()}")

prj = "BioHeat_PINNs"
n_test = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")  # Current timestamp in the format YYYYMMDD_HHMMSS

run = f"date_time_{n_test}"
print(f"Run identifier: {run}")

# Define the folder path with absolute path
base_dir = os.getcwd()
folder_path = os.path.join(base_dir, "tests", "models", run)
print(f"Attempting to create directory at: {folder_path}")

# Ensure parent directories exist and are writable
parent_dir = os.path.dirname(folder_path)
if not os.path.exists(parent_dir):
    print(f"Parent directory does not exist: {parent_dir}")
else:
    print(f"Parent directory exists: {parent_dir}")

try:
    os.makedirs(folder_path, exist_ok=True)
    print(f"Successfully created directory: {folder_path}")
except Exception as e:
    print(f"Error creating directory: {e}")

name, general_figures, model_dir, figures_dir = utils.set_name(prj, run)

# Create NBHO with some config.json
config = utils.read_config(run)
# config["output_injection_gain"] = 200
utils.write_config(config, run)

# Use a default filename instead of a timestamp-based one
utils.single_observer(prj, run, "0")