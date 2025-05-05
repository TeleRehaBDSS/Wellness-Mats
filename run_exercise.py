import multiprocessing
import sys
import time
import os
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = "/home/uoi/.local/lib/python3.8/site-packages/PyQt5/Qt/plugins"
os.environ["QT_QPA_PLATFORM"] = "xcb"
from PyQt5 import QtWidgets
from frontend_widget import PressureMapAppSingle, PressureMapAppTreadmill, data_queue, visual_queue
from read_mat_sensors import data_collection_process, get_mat_pressures, process_visualization



# Start standing / Balance exercise
def single_mat_exercise():
  # Start the data collection pro
  # cess
  data_process = multiprocessing.Process(target=get_mat_pressures, args=(data_queue,))
  data_process.start()

  # Create the PyQt application
  app = QtWidgets.QApplication(sys.argv)
  ex = PressureMapAppSingle(data_queue)
  sys.exit(app.exec_())
      
  # Wait for the data collection process to finish (if needed)
  data_process.join()


# Start walking exercise
def walking_exercise():
  #data_collection_process(True)
  # Start the data collection process
  dc = multiprocessing.Process(target=data_collection_process, args=(True, ))
  dc.start()
  vp = multiprocessing.Process(target=process_visualization, args=(visual_queue, data_queue,))
  vp.start()
  
  # Create the PyQt application
  app = QtWidgets.QApplication(sys.argv)
  ex = PressureMapAppTreadmill(visual_queue)
  
  sys.exit(app.exec_())
   

if __name__ == '__main__':
    time.sleep(4)
    #single_mat_exercise()
    walking_exercise()
