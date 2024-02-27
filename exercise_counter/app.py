import argparse
import sys

from thresholds import Thresholds


def main(exercise_name):
    # Map exercise names to their thresholds and module names
    _thresholds = Thresholds()  # Initialize the ExerciseThresholds class

    try:
        exercise_thresholds = _thresholds.get_threshold(exercise_name)
        if exercise_thresholds and exercise_name == 'squat':
            pass

    except ValueError:
        print(f"{exercise_name} is not a valid exercise")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run exercise detection.")
    parser.add_argument("--exercise", type=str, help="The type of exercise to detect (e.g., 'squat', 'pull_up')")

    args = parser.parse_args()

    if args.exercise:
        main(args.exercise)
    else:
        print("Please specify an exercise with --exercise")
        sys.exit(1)
