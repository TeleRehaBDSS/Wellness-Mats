import os
import serial
from configuration import config


# Configuration
baud = config['BAUDRATE']
time_out = config['TIMEOUT']
request = config['REQUEST_NEW_MAP']


# Function to connect to device serial port
def connect_to_port(device):
  ser = serial.Serial(
    port = device['port'],
    baudrate = baud,
    timeout = time_out
  )
  return ser


# Function to grand port permissions to the user 
# (required for request_new_map)
def grand_port_permissions(device):
  port = device['port']
  os.system(f'sudo chmod +006 {port}')
  print(f"Granted user permissions to device: {port}.")


# Function that requests a new map from the device port
def request_new_map(ser):
  data = request
  ser.write(data.encode())
