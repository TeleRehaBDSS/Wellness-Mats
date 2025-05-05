import json
from serial.tools import list_ports
from pathlib import Path
from connect_to_mat import grand_port_permissions
from configuration import config


mats_detected = Path("_detected_mats_configuration.json")
treadmill_file = Path("_mat_sequence_configuration.json")

vendor = config["VID_PID"]

walking_path = []

# Function to detect connected mats
def detect_connected_mats():
  enmu_ports = enumerate(list_ports.comports())

  detected_mats = []
  
  for n, (p, descriptor, hid) in enmu_ports:
    port = p
    usb_hub_location = hid.split(' ')[3].split('=')[1]
      
    mat_to_add = {"id": int(len(detected_mats) + 1),
                  "port": port,
                  "hub_port": usb_hub_location,
                }
    
    vendor_id = hid.split(' ')[1].split('=')[1]

    if vendor_id == vendor:
      detected_mats.append(mat_to_add)    
    
  # Create a json file containing mat(s) information
  json.dump(detected_mats, open("_detected_mats_configuration.json", 'w'))

  available_mats = get_mats_information()

  print(f"{len(available_mats)} mat(s) discovered.\nConfiguration file created.")

  for mat in detected_mats:
    grand_port_permissions(mat)

  return available_mats


# Reads the created file and returns the connected mats
def get_mats_information():
  try:
    with open(mats_detected) as detected_mats:
      if mats_detected.is_file():
        return json.load(detected_mats)
  except FileNotFoundError:
    print("Running mats detection.")
    detect_connected_mats()
    with open(mats_detected) as detected_mats:
      if mats_detected.is_file():
        return json.load(detected_mats)
