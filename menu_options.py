import os
import sys
import time
from create_walking_path import get_treadmill_information
from run_exercise import single_mat_exercise, walking_exercise


def clear_screen():
  time.sleep(1)
  os.system('cls' if os.name == 'nt' else 'clear')

def menu():
  print("Please type in a number to select the respective menu option")
  print("\n")
  print("[0] Terminate the application")
  print("[1] Create connected mats sequence")
  print("[2] Start Standing exercise")
  print("[3] Start Walking exercise")
  print("\n")

def get_user_input():
  clear_screen()
  try:
    menu()

    selection = int(input("Enter your selection: "))
  except ValueError:
    print("Please input a valid option")
  
  user_selection(selection)

def user_selection(selection):
  
  if selection == 0:
    print("Application Terminated")
    sys.exit()
  
  elif selection == 1:
    get_treadmill_information()

  elif selection == 2:
    single_mat_exercise()
    
  elif selection == 3:
    walking_exercise()

  else:
    print("Please input a valid option")

  print("\n")
