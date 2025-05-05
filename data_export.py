import pandas as pd
from datetime import datetime as dt
import json
import os


# Print DataFrame in Terminal and export to xlsx file
def print_data_frame(current_map, filename_str):
  # Prepare the data for DataFrame
  list_of_sensor_maps = [device_map['sensors'] for device_map in current_map]
  
  # Flatten each 48x48 sensor map to have a row with 48 elements
  flattened_data = [sensor_row for sensor_map in list_of_sensor_maps for sensor_row in sensor_map]

  # Create the DataFrame with 48 columns
  pd.set_option("display.max_rows", None)
  df = pd.DataFrame(flattened_data)
  print(df)

  # Save DataFrame to an Excel file
  timepoint = dt.now().isoformat()
  df.to_excel(f'{filename_str}_{timepoint}.xlsx', index=False)


# Creates a json file
def export_to_json(current_map, name):
  timepoint = dt.now().isoformat()
  path = "Refactored Exercises"
  isExist = os.path.exists(path)
  if not isExist:
    os.makedirs(path)
    print(f"Folder {path} was created")
  
  filename = f"{path}/_{name}_{timepoint}.json"

  json.dump(current_map, open(filename,'w'))
  print('The standing exercise was completed and the file was created ') 
