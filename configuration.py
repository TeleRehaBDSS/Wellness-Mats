import random

config = {
  "STOPWATCH": 40,
  "VID_PID": "16C0:0483",
  "REQUEST_NEW_MAP": "R",
  "ROWS": 48,
  "COLS": 48,
  "BAUDRATE": 115200,
  "TIMEOUT": 0.1,
  "type": ["FGA","MiniBEST"]
}

exercise_type_map = {
    "FGA": ["Gait Level Surface", "Change in gait speed","Gait with horizontal head turns","Gait with vertical head turns","Gait and pivot turn","Step over obstacle","Gait with narrow base of support","Gai with eyes closed","Ambulating backwards"],
    "MiniBEST": ["Sit to stand","Rise to toes","Stand on one leg","Compensatory stepping correction- FORWARD","Compensatory stepping correction- BACKWARD","Compensatory stepping correction- LATERAL","Stance, Eyes open","Stance, Eyes closed","Change in gait speed","Walk with head turns - HORIZONTAL","Walk with pivot turns","Step over obstacles","Timed up & go with duals task"]
}

metrics_type_map ={
    "MiniBEST": {
            'Number of movements': random.randint(3, 7),
            'Pace movements per second': round(random.uniform(0.1, 0.3), 3),
            'Mean movements range degrees': round(random.uniform(40 , 75),3),
            'Std movements range degrees' : round(random.uniform(40 , 75),3),
            'Mean duration seconds' : round(random.uniform(0 , 1),3),
            'Std duration seconds ' : round(random.uniform(0 , 1),3)
        },
    "FGA": {
            'Number of steps': 9,
            'Mean speed (m/s)': 0.97 ,
            'Std speed ': 0.02
        }
    }
