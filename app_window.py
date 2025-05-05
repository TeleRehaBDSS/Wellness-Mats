from frontend_widget import PressureMapAppSingle, PressureMapAppTreadmill
import sys
import os
import numpy as np
from configuration import config, exercise_type_map, metrics_type_map
import multiprocessing
from multiprocessing import Value
from read_mat_sensors import get_mat_pressures, process_visualization, data_collection_process
from PyQt5.QtGui import QPixmap
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = "/home/uoi/.local/lib/python3.8/site-packages/PyQt5/Qt/plugins"
os.environ["QT_QPA_PLATFORM"] = "xcb"
from PyQt5 import QtWidgets, QtCore
import json
import csv
import time

def export_collected_data_from_queue(self):
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    exercise_name_clean = self.exercise_box.currentText().replace(' ', '_')
    filename = f"{self.patient_id}_{exercise_name_clean}_{timestamp}.csv"

    rows = []

    while not self.export_queue.empty():
        try:
            row = self.export_queue.get_nowait()

            # 👇 Debug print each item before saving
            print("[Export DEBUG] Got from queue:", type(row), row if isinstance(row, dict) else "matrix shape: " + str(row.shape))

            # 🧠 If treadmill-style dict → append directly
            if isinstance(row, dict) and "sensors" in row:
                rows.append(row)

            # 🧠 If raw numpy matrix from single mat → wrap it
            elif isinstance(row, np.ndarray):
                wrapped = {
                    "timepoint": time.time(),
                    "sensors": row.tolist(),
                    "mat": 0,
                    "sample": len(rows) + 1
                }
                rows.append(wrapped)

        except Exception as e:
            print(f"[Export ERROR] Failed to read from queue: {e}")
            break

    # Save to CSV if rows collected
    if rows:
        with open(filename, mode='w', newline='') as file:
            fieldnames = rows[0].keys()
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                row['sensors'] = json.dumps(row['sensors'])  # Serialize
                writer.writerow(row)

        QtWidgets.QMessageBox.information(self, "Data Saved", f"Data saved successfully as:\n\n{filename}")
    else:
        QtWidgets.QMessageBox.warning(self, "No Data", "No data was collected to save.")



# Configuration
exercise_types = config["type"]
exercise_map = exercise_type_map 
metrics_map = metrics_type_map #Dummies for now

class ComboBoxNoWheel(QtWidgets.QComboBox): #Class for combobox (without wheel)
    def __init__(self, parent=None):
        super().__init__(parent)

    # Override the wheelEvent to disable scrolling
    def wheelEvent(self, event):
        event.ignore()  # Ignore wheel events to prevent scrolling

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setWindowTitle("TeleRehaB DSS")
        self.setGeometry(100, 100, 1000, 600)

        # Create a title bar with the logo and window title
        self.create_custom_title_bar()

        # Allow the window to be maximized and resizable
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.setMinimumSize(800, 600)

        # Apply stylesheet
        self.setStyleSheet("""
            QWidget {
                font-family: "Segoe UI", sans-serif;
                font-size: 14px;
                background-color: #2b2b2b;
                color: #ffffff;
            }

            QComboBox {
                border: 2px solid #444;
                border-radius: 10px;
                padding: 8px;
                background-color: #3d3d3d;
                color: #ffffff;
                
            }

            QPushButton {
                background-color: #4a4a4a;
                color: white;
                border-radius: 10px;
                padding: 10px 15px;
                
            }

            QPushButton:hover {
                background-color: #575757;
            }

            QPushButton:pressed {
                background-color: #3d3d3d;
                
            }

            QTextEdit {
                border: 2px solid #555;
                border-radius: 10px;
                padding: 12px;
                background-color: #3a3a3a;
                color: #ffffff;
               
            }

            QLabel {
                color: #cfcfcf;
                font-weight: bold;
            }

            QMainWindow {
                background-color: #2b2b2b;
            }

            #chronometer, #exit_button {
                font-size: 32px;
                color: #ffffff;
                border: 2px solid #444;
                border-radius: 15px;
                padding: 10px;
                background-color: #3a3a3a;
                
            }
                           
            QWidget#heatmap_container {
                background-color: #2c2c2c;
                border: 3px solid #444;
                border-radius: 15px;
                padding: 20px;
                
            }
        """)

        # Central widget for the layout
        self.central_widget = QtWidgets.QWidget(self)
        self.setCentralWidget(self.central_widget)

        # Main layout: vertical layout to stack the main content and the chronometer
        self.main_layout = QtWidgets.QVBoxLayout(self.central_widget)

        # Content layout: horizontal layout to arrange exercise info and heatmap side by side
        self.content_layout = QtWidgets.QHBoxLayout()
        self.main_layout.addLayout(self.content_layout)

        # Exercise type selection
        self.type_box = ComboBoxNoWheel(self)
        self.type_box.addItems(exercise_types)  # Add exercise types from config
        self.type_box.currentIndexChanged.connect(self.on_type_change)

        # Exercise selection
        self.exercise_box = ComboBoxNoWheel(self)

        self.start_button = QtWidgets.QPushButton('Start Exercise')
        self.start_button.clicked.connect(self.start_exercise)

        self.stop_button = QtWidgets.QPushButton('Stop Exercise')
        self.stop_button.clicked.connect(self.stop_exercise)

        self.exit_button = QtWidgets.QPushButton('Exit')
        self.exit_button.setObjectName("exit_button")
        self.exit_button.clicked.connect(self.show_exit_dialog)

        self.exercise_info = QtWidgets.QTextEdit(self)
        self.exercise_info.setReadOnly(True)

        # Fixed container for heatmap
        self.heatmap_container = QtWidgets.QWidget(self)
        self.heatmap_container.setObjectName("heatmap_container")
        self.heatmap_layout = QtWidgets.QVBoxLayout(self.heatmap_container)

        # Add a placeholder before the heatmap is displayed
        self.heatmap_placeholder = QtWidgets.QLabel(self)
        self.heatmap_placeholder.setText("Heatmap will appear here")
        self.heatmap_placeholder.setAlignment(QtCore.Qt.AlignCenter)
        self.heatmap_placeholder.setStyleSheet("border: 1px solid gray; color: gray;")
        self.heatmap_layout.addWidget(self.heatmap_placeholder)

        # Placeholder has the same size policy as the heatmap
        self.heatmap_placeholder.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.heatmap_layout.addWidget(self.heatmap_placeholder)

        # Vertical layout to stack the exercise info components
        self.info_layout = QtWidgets.QVBoxLayout()
        # 🆕 Patient ID field here
        self.info_layout.addWidget(QtWidgets.QLabel('Patient ID'))
        self.patient_id_box = QtWidgets.QLineEdit(self)
        self.info_layout.addWidget(self.patient_id_box)
        self.info_layout.addWidget(QtWidgets.QLabel('Select Test'))
        self.info_layout.addWidget(self.type_box)
        self.info_layout.addWidget(QtWidgets.QLabel('Select Exercise:'))
        self.info_layout.addWidget(self.exercise_box)
        self.info_layout.addWidget(self.start_button)
        self.info_layout.addWidget(self.stop_button)
        self.info_layout.addWidget(QtWidgets.QLabel('Exercise Information:'))
        self.info_layout.addWidget(self.exercise_info)
        self.info_layout.addWidget(self.exit_button)

        # Chronometer
        self.chronometer = QtWidgets.QLabel('00:00:00', self)
        self.chronometer.setAlignment(QtCore.Qt.AlignCenter)
        self.chronometer.setObjectName("chronometer")
        # self.main_layout.addWidget(self.chronometer)

        # Timer for chronometer
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_chronometer)
        self.elapsed_time = 0

        # Heatmap and Chronometer layout
        self.right_side_layout = QtWidgets.QVBoxLayout()
        self.right_side_layout.addWidget(self.heatmap_container)
        self.right_side_layout.addWidget(self.chronometer)

        # Add info layout and heatmap container to the content layout
        self.content_layout.addLayout(self.info_layout)
        self.content_layout.addLayout(self.right_side_layout)

        # Stopwatch limit from config
        self.stopwatch_limit = config.get("STOPWATCH", 60)  # Default to 60 seconds if not in config

        # Track the process running state
        self.data_process = None
        self.visual_process = None

        self.patient_id = ""
        # Set default values for type and exercise selection
        self.set_default_values()


    def export_collected_data_from_queue(self):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        exercise_name_clean = self.exercise_box.currentText().replace(' ', '_')
        filename = f"{self.patient_id}_{exercise_name_clean}_{timestamp}.csv"

        rows = []

        while not self.export_queue.empty():
            try:
                row = self.export_queue.get_nowait()
                if isinstance(row, dict):
                    rows.append(row)
            except:
                break

        if rows:
            with open(filename, mode='w', newline='') as file:
                fieldnames = rows[0].keys()
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                writer.writeheader()
                for row in rows:
                    row['sensors'] = json.dumps(row['sensors'])  # Ensure matrix is serialized
                    writer.writerow(row)

            QtWidgets.QMessageBox.information(self, "Data Saved", f"Data saved successfully as:\n\n{filename}")
        else:
            QtWidgets.QMessageBox.warning(self, "No Data", "No data was collected to save.")

    def create_exit_message_box(self):
        # Create the message box for confirmation
        msg_box = QtWidgets.QMessageBox()
        msg_box.setWindowTitle('Confirm Exit')
        msg_box.setText('<font color="white">Are you sure you want to quit?</font>')
        msg_box.setStandardButtons(QtWidgets.QMessageBox.No | QtWidgets.QMessageBox.Yes)
        msg_box.setIcon(QtWidgets.QMessageBox.Question)

        # Set the style
        msg_box.setStyleSheet("""
        QMessageBox {
            background-color: #2b2b2b;
            color: white;
            font-family: "Segoe UI", sans-serif;
            font-size: 14px;
        }

        QPushButton {
            background-color: #4a4a4a;
            color: white;
            border-radius: 10px;
            padding: 10px 15px;
        }

        QPushButton:hover {
            background-color: #575757;
        }

        QPushButton:pressed {
            background-color: #3d3d3d;
        }
    """)
    
        return msg_box

    def show_exit_dialog(self):

        msg_box = self.create_exit_message_box() 
        # Execute and get the user choice
        choice = msg_box.exec_()

        # Handle the response
        if choice == QtWidgets.QMessageBox.Yes:
            self.terminate_application()
        else:
            msg_box.close()
            
    
    def create_custom_title_bar(self):
        # Create a widget for the custom title bar
        title_bar_widget = QtWidgets.QWidget(self)
        title_bar_layout = QtWidgets.QHBoxLayout(title_bar_widget)

        # Load the logo
        logo_label = QtWidgets.QLabel(self)
        pixmap = QPixmap("/home/uoi/Documents/GitHub/Telerehab_UOI/WP3/wellness_mat_pp/Telerehab.png")  # Provide the correct path to your logo
        logo_label.setPixmap(pixmap.scaled(60, 60, QtCore.Qt.KeepAspectRatio)) 

        # Create the title label
        title_label = QtWidgets.QLabel("TeleRehaB DSS", self)
        title_label.setStyleSheet("font-weight: bold; font-size: 16px;")

        # Add the logo and title label to the layout
        title_bar_layout.addWidget(logo_label)
        title_bar_layout.addWidget(title_label)

        # Add some stretch to align the title to the left
        title_bar_layout.addStretch()

        # Set the custom title bar layout as the title bar widget
        self.setMenuWidget(title_bar_widget)

    def set_default_values(self):
        # Set the default index for type_box
        default_type_index = 0  # Set the index you want as default (e.g., 0 for the first type)
        self.type_box.setCurrentIndex(default_type_index)

        # Get the corresponding exercises for the default type
        default_type = self.type_box.itemText(default_type_index)
        corresponding_exercises = exercise_map.get(default_type, [])

        # Set the exercise_box items based on the default type
        self.exercise_box.clear()
        self.exercise_box.addItems(corresponding_exercises)

        # Set the default index for exercise_box
        default_exercise_index = 0  # Set the index you want as default (e.g., 0 for the first exercise)
        self.exercise_box.setCurrentIndex(default_exercise_index)

    def on_type_change(self):
        selected_type = self.type_box.currentText()
        corresponding_exercises = exercise_map.get(selected_type, [])
        self.exercise_box.clear()  # Clear current options
        self.exercise_box.addItems(corresponding_exercises)  # Add new options based on selected type
    

    def start_exercise(self):
        self.patient_id = self.patient_id_box.text().strip()
        if not self.patient_id:
            QtWidgets.QMessageBox.warning(self, "Missing Patient ID", "Please enter a Patient ID before starting an exercise.")
            return
        self.running = Value('b', True)
        self.exercise_info.clear()
        self.type_box.setEnabled(False)
        self.exercise_box.setEnabled(False)
        exercise_name = self.exercise_box.currentText()
        exercise_type = self.type_box.currentText()

        # Create new queues fresh every time you start a new exercise
        self.data_queue = multiprocessing.Queue()
        self.visual_queue = multiprocessing.Queue()
        self.export_queue = multiprocessing.Queue()

        # Clear the heatmap container if it already has widgets
        for i in reversed(range(self.heatmap_layout.count())): 
            widget = self.heatmap_layout.itemAt(i).widget()
            if widget is not None:
                widget.deleteLater()

        # Use the correct widget and data queue for the selected exercise
        single_mat_exercises = [
            "Sit to stand", "Rise to toes", "Stand on one leg",
            "Compensatory stepping correction- FORWARD",
            "Compensatory stepping correction- BACKWARD",
            "Compensatory stepping correction- LATERAL",
            "Stance, Eyes open", "Stance, Eyes closed"
        ]

        if self.exercise_box.currentText() in single_mat_exercises:
            # Single Mat
            self.heatmap_widget = PressureMapAppSingle(self.data_queue)
            self.heatmap_layout.addWidget(self.heatmap_widget)

            self.data_process = multiprocessing.Process(target=get_mat_pressures, args=(self.data_queue,))
            self.data_process.start()
            self.visual_process = None
            self.export_queue = self.data_queue
        else:
            # Treadmill
            self.heatmap_widget = PressureMapAppTreadmill(self.visual_queue)
            self.heatmap_layout.addWidget(self.heatmap_widget)

            self.data_process = multiprocessing.Process(target=data_collection_process, args=(self.running, self.data_queue,self.export_queue))
            self.visual_process = multiprocessing.Process(target=process_visualization, args=(self.visual_queue, self.data_queue,))
            self.data_process.start()
            self.visual_process.start()

        self.heatmap_layout.addWidget(self.heatmap_widget)

        # Start chronometer
        self.elapsed_time = 0
        self.timer.start(1000)  # Timer updates every second

    def stop_exercise(self):
        self.running = Value('b', False)
        exercise_type = self.type_box.currentText()
        exercise = self.exercise_box.currentText()
        self.type_box.setEnabled(True)
        self.exercise_box.setEnabled(True)

        if self.data_process is not None and self.data_process.is_alive():
            self.data_process.terminate()
            self.data_process.join()
            self.data_process = None

        if self.visual_process is not None and self.visual_process.is_alive():
            self.visual_process.terminate()
            self.visual_process.join()
            self.visual_process = None

        # Clear queues
        self.data_queue = None
        self.visual_queue = None

        # Clear heatmap
        for i in reversed(range(self.heatmap_layout.count())):
            widget = self.heatmap_layout.itemAt(i).widget()
            if widget is not None:
                widget.deleteLater()

        # Add placeholder
        self.heatmap_placeholder = QtWidgets.QLabel(self)
        self.heatmap_placeholder.setText("Heatmap will appear here")
        self.heatmap_placeholder.setAlignment(QtCore.Qt.AlignCenter)
        self.heatmap_placeholder.setStyleSheet("border: 1px solid gray; color: gray;")
        self.heatmap_placeholder.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.heatmap_layout.addWidget(self.heatmap_placeholder)

        # Stop chronometer
        self.timer.stop()
        self.chronometer.setText('00:00:00')
        self.elapsed_time = 0

        metrics = metrics_map[exercise_type]
        formatted_metrics = self.format_metrics(metrics)
        self.exercise_info.setText("Exercise : %s\n %s" % (exercise, formatted_metrics))
        if hasattr(self, 'export_queue') and self.export_queue is not None:
            print("[DEBUG] Export queue size before export:", self.export_queue.qsize())
            self.export_collected_data_from_queue()
        # 🆕 Show success popup
        QtWidgets.QMessageBox.information(self, "Exercise Completed", "Data upload successful and metrics generated!")


    def format_metrics(self, metrics):
        formatted = ""
        for key, value in metrics.items():
            formatted += "%s: %s\n" % (key, value)
      # Convert each item into a string
        return formatted
    
    def closeEvent(self, event):
        msg_box = self.create_exit_message_box()
        choice = msg_box.exec_()

        if choice == QtWidgets.QMessageBox.Yes:
            event.accept()  # Accept the event and close the app
        else:
            event.ignore()  # Ignore the event and keep the app running

    def terminate_application(self):
        self.stop_exercise
        sys.exit(0)

    def update_chronometer(self):
        self.elapsed_time += 1
        time_str = QtCore.QTime(0, 0).addSecs(self.elapsed_time).toString('hh:mm:ss')
        self.chronometer.setText(time_str)

        # Check if the elapsed time exceeds the stopwatch limit
        if self.elapsed_time >= self.stopwatch_limit:
            self.stop_exercise()  # Stop the exercise if the stopwatch limit is reached

def main():
    app = QtWidgets.QApplication(sys.argv)
    main_window = MainWindow()
    # Show  the application window maximized
    main_window.showMaximized()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
