import json
from pathlib import Path
import time
from detect_available_mats import get_mats_information
from get_mat_matrix import get_mat_pressure_sum


connected_mats = get_mats_information()

walking_path = []

sequence_file = Path("_mat_sequence_configuration.json")


# Function to get pressure for each connected mat
def calculate_pressure():
  retrieved_pressure_values = []

  for mat in connected_mats:
    device_port = mat['port']
    mat_pressure_sum = get_mat_pressure_sum(mat)

    retrieved_pressure_values.append({
          "mat": mat,
          "detected_pressure": mat_pressure_sum
          })
      
    print(f"For mat connected in port {device_port}, the detected pressure is: {mat_pressure_sum}")
   
  standing_mat= max(retrieved_pressure_values, key=lambda x: x['detected_pressure'])
  mat_to_add_in_sequence = standing_mat['mat']

  if mat_to_add_in_sequence not in walking_path:
    walking_path.append(mat_to_add_in_sequence)

  return walking_path


# Repeats calculate_preassure for all connected mats
def create_walking_path():     
  for i in range(len(connected_mats)):
    print(f'\n Please step and stand on mat # {i + 1} in the next 5 seconds')
    time.sleep(5)
    calculate_pressure()
    print(f'\n Please step of the mat \n')
    time.sleep(2)
    
  # Create a json file containing mat(s) sequence
  json.dump(walking_path, open("_mat_sequence_configuration.json", 'w'))


# Reads the created file and returns the sequence of the connected mats
def get_treadmill_information():
  try:
    with open(sequence_file) as treadmill:
      if sequence_file.is_file():
        return json.load(treadmill)
  except FileNotFoundError:
    print("Walking path not found - Running configuration sequence")
    create_walking_path()

if __name__ == "__main__":
  create_walking_path()
