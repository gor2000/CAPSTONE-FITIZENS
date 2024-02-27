import numpy as np
import importlib
import sys

def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

def get_exercise_function(exercise_name):
    """Dynamically imports and returns the exercise detection function."""
    try:
        exercise_module = importlib.import_module(f"exercises.{exercise_name}")
        return exercise_module.detect_exercise
    except (ImportError, AttributeError) as e:
        print(f"Error: Could not find a detection function for {exercise_name}.")
        sys.exit(1)