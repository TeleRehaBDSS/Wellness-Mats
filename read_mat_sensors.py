import multiprocessing
import sys
import time
from configuration import config
from connect_to_mat import connect_to_port, request_new_map
from create_walking_path import get_treadmill_information
import data_export
import numpy as np
from datetime import datetime as dt
from get_mat_matrix import create_new_device_map, create_new_treadmill_map
from data_walking_csv import export_to_jso
import csv
import json

current_treadmill = multiprocessing.Queue()

# Configuration
rows = config['ROWS']
cols = config['COLS']
exercise_runtime = config['STOPWATCH']

walking_path = get_treadmill_information()

detected_mats = len(walking_path)

current_map = []

values = np.zeros((rows, cols))

def get_sinlge_mat_pressure(mat, queue):
    ser = connect_to_port(mat)
    request_new_map(ser)
    time.sleep(0.01)

    device_map = create_new_device_map(mat, ser)
    sensors_values = device_map['sensors']

    current_map.append(device_map)

    # Push the 48x48 matrix into the queue
    queue.put(np.flipud(sensors_values))
    ser.close()

    return -1


# Function to collect single mat data
def get_mat_pressures(queue):
  mats = walking_path


  standing_mat = mats[0]

  stopwatch = time.time() + exercise_runtime
#   while True:
  while time.time() < stopwatch:
    #   start = dt.now()
      get_sinlge_mat_pressure(standing_mat, queue)
    #   end = dt.now()

    #   print('time for a frame = ', end - start)

  data_export.export_to_json(current_map, 'refactoring') 


# Function to collect treadmill widget visualization data
def process_visualization(visual_queue, data_queue):
    z_zero = values

    dataFlag = []
    lastData = []
    for i in range(detected_mats):
        dataFlag.append(False)
        lastData.append(None)
     
    while (True):
        treadmill_data = []

        try:
            tmp_data = data_queue.get_nowait()  # Non-blocking get
            m = tmp_data['mat']
            sensor_data = tmp_data['sensors']
              

            if (has_pressure(sensor_data) > 0):
                # print('m = ', m)

                for i in range(m):
                    if (dataFlag[i]):
                        treadmill_data.append(lastData[i])
                    else:
                        treadmill_data.append(z_zero)
                
                treadmill_data.append(tmp_data['sensors'])
                
                for i in range(m + 1, detected_mats):
                    if (dataFlag[i]):
                        treadmill_data.append(lastData[i])
                    else:
                        treadmill_data.append(z_zero)
                
                #treadmill_instance = np.fliplr(np.concatenate(treadmill_data, axis =0))
                treadmill_instance = np.concatenate(treadmill_data[::-1], axis =1)
                #rotated_instance = np.rot90(treadmill_instance, k=-1)
                visual_queue.put(np.flipud(treadmill_instance))

                
                # current_map.put(treadmill_instance)
                current_map.append(treadmill_instance)

                lastData[m] = sensor_data
                dataFlag[m] = True
            else:
                
                for i in range(m):
                    lastData[m] = treadmill_data.append(z_zero)

                dataFlag[m] = False
        except:
            k = 0
        


# Returns the sum of the sensor values collected by a single mat
def has_pressure(sensor_data):
    row_sum = [sum(row) for row in sensor_data]
    map_sum = sum(row_sum)

    return map_sum

def export_queue_to_csv(queue, csv_filename):
    with open(csv_filename, mode="w", newline="") as file:
        # Define CSV headers based on your new_map keys
        headers = ["timepoint", "sensors", "mat", "sample"]
        
        # Initialize the writer
        writer = csv.DictWriter(file, fieldnames=headers)
        writer.writeheader()  # Write the header

        # Fetch and write each row
        while not queue.empty():
            row = queue.get()
            # Convert 'sensors' data (array/matrix) to a JSON string
            row["sensors"] = json.dumps(row["sensors"])
            writer.writerow(row)

    print(f"Data successfully exported to {csv_filename}")

def data_collection_process(running_flag,data_queue,export_queue):
    # Start the data collection process
    data_process = []
    saved_data = []

    c = 0

    for mat in walking_path:
        #print('I am here.....')
        data_process.append('')
        data_process[c] = multiprocessing.Process(target=get_treadmill_mat_pressures, args=(c, mat, data_queue, export_queue, running_flag,saved_data,))
        data_process[c].start()
        c = c + 1




def get_treadmill_mat_pressures(counter, mat, queue, export_queue, running_flag,saved_data):
  stopwatch = time.time() + exercise_runtime
  sample = 0
  if running_flag: 
    while time.time() < stopwatch:
        timepoint = dt.now().isoformat()
        treadmill_data = []

        start = dt.now()
        current = get_treadmill_mat_pressure(mat)
        sample = sample + 1
        end = dt.now()
        #print(current)
        #print('Time to process a mat = ', counter, '-', end - start)

        np.set_printoptions(threshold=sys.maxsize)
        collected_data = current
        if running_flag == False:
          stopwatch = 0
          queue = []
        current_treadmill.put(current)
      #   print('mat = ', mat, 'sample = ', sample)
      
        # Push the matrix in the queque
        new_map = {"timepoint" : timepoint,
                  "sensors" : collected_data, 
                  "mat":counter, 
                  "sample": sample}
        saved_data.append(new_map)
        #collected_data.append(new_map) 
        #print(new_map)
        queue.put(new_map)
        export_queue.put(new_map)
        #print(new_map)
    #return 0;
    #export_to_jso(saved_data,f"Data_{counter}")
    
    

  
            


def get_treadmill_mat_pressure(mat):    
    ser = connect_to_port(mat)
    request_new_map(ser)
    time.sleep(0.01)
    device_map = create_new_treadmill_map(ser)
        
    return device_map
