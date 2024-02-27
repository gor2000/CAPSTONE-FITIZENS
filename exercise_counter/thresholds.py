class Thresholds:
    def __init__(self):
        self.squat_thresholds = {
            "standing_angle": 160,
            "squat_angle": 100
        }

    def get_threshold(self, exercise_name):
        if exercise_name == 'squat':
            return self.squat_thresholds
        else:
            raise ValueError(f'No threshold defined for exercise {exercise_name}')