import time
import numpy as np
from configuration import config
from connect_to_mat import connect_to_port, request_new_map
from datetime import datetime as dt

# Configuration
rows = config['ROWS']
cols = config['COLS']

# Initialize values as empty array
values = np.zeros((rows, cols), dtype=int)

# Create an empty list to store mat sensor values
current_map = []


# Function to request and receive a new pressure map
def request_pressure_map(ser):
    if ser.in_waiting > 0:
        try:
            xbyte = ser.read().decode('utf-8')
        except Exception:
            print("Exception")
        if xbyte == 'N':
            active_points_receive_map(ser)
        else:
            ser.flush()


# Function to handle the incoming data points
def active_points_receive_map(ser):
  global values
  matrix = np.zeros((rows, cols), dtype=int)

  xbyte = ser.read().decode('utf-8')

  HighByte = ser.read()
  LowByte = ser.read()
  high = int.from_bytes(HighByte, 'big')
  low = int.from_bytes(LowByte, 'big')
  nPoints = ((high << 8) | low)

  xbyte = ser.read().decode('utf-8')
  xbyte = ser.read().decode('utf-8')
  x = 0
  y = 0
  n = 0
  while(n < nPoints):
    x = ser.read()
    y = ser.read()
    x = int.from_bytes(x, 'big')
    y = int.from_bytes(y, 'big')
    HighByte = ser.read()
    LowByte = ser.read()
    high = int.from_bytes(HighByte, 'big')
    low = int.from_bytes(LowByte, 'big')
    val = ((high << 8) | low)

    if val >= 33:
        matrix[y][x] = val
    else:
        # Define neighborhood size for proximity check
        neighborhood_size = 5
        half_neighborhood = neighborhood_size // 2

        # Determine neighborhood boundaries
        y_min = max(0, y - half_neighborhood)
        y_max = min(rows, y + half_neighborhood + 1)
        x_min = max(0, x - half_neighborhood)
        x_max = min(cols, x + half_neighborhood + 1)

        # Extract neighborhood
        neighbors = matrix[y_min:y_max, x_min:x_max]

        # Set the value to zero only if all neighbors are below the threshold
        if np.all(neighbors < 33):
            matrix[y][x] = 0
        else:
            matrix[y][x] = val
    n += 1
  values = np.fliplr(matrix)

# Function to calculate sum of sensor pressures
def get_mat_pressure_sum(mat):
    for i in range(2):
        if not i == 2:
            pass

        ser = connect_to_port(mat)
        request_new_map(ser)
        time.sleep(0.1)
        
        request_pressure_map(ser)
        
        collected_data = values.tolist()

        row_sum = [sum(row) for row in collected_data]
        mat_sum = sum(row_sum)
        
    return mat_sum


# Function to create a new device map for single mat
def create_new_device_map(mat, ser):
    device_port = mat['port']
    timepoint = dt.now().isoformat()
    
    request_pressure_map(ser)

    collected_data = values.tolist()
    
    new_map = {"dateTime": timepoint,
               "device_port": device_port,
               "sensors": collected_data}
    
    current_map.append(new_map)

    return new_map


# Function to create a new device map for walking path mats
def create_new_treadmill_map(ser):    
    request_pressure_map(ser)
    collected_data = values.tolist()
    
    return collected_data
