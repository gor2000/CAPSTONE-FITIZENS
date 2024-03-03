import cv2
from ultralytics.utils.plotting import Annotator
from ultralytics.utils.checks import check_imshow

class Annotator_altered(Annotator):
    def __init__(self, im0, *args, **kwargs):  # Annotator(im0, line_width=2)
        super().__init__(im0, *args, **kwargs)  # Call the constructor of the base class (if needed)
        # print("Subclass constructor called.")
        # print(self.im0)

    def draw_specific_points(self, keypoints, indices=[2, 5, 7], shape=(640, 640), radius=2):
        """
        Draw specific keypoints for gym steps counting.

        Args:
            keypoints (list): list of keypoints data to be plotted (required)
            indices (list): keypoints ids list to be plotted
            shape (tuple): imgsz for model inference
            radius (int): Keypoint radius value
        """
        for i, k in enumerate(keypoints):

            # if i in indices:
            x_coord, y_coord = k[0], k[1]
            if x_coord % shape[1] != 0 and y_coord % shape[0] != 0:
                if len(k) == 3:
                    conf = k[2]
                    if conf < 0.5:  ## confidence value (if smaller skipp)
                        continue
                cv2.circle(self.im, (int(x_coord), int(y_coord)), radius, (0, 255, 0), -1, lineType=cv2.LINE_AA)
        return self.im

    def plot_angle_and_count_and_stage(self, angle_text, count_text, stage_text, center_kpt, line_thickness=2,
                                       n_people=1, video_source_is_video=True):
        """
        Plot the pose angle, count value and step stage.

        Args:
            angle_text (str): angle value for workout monitoring (required)
            count_text (str): counts value for workout monitoring (required)
            stage_text (str): stage decision for workout monitoring (required)
            center_kpt (int): centroid pose index for workout monitoring (required)
            line_thickness (int): thickness for text display
        """
        angle_text, count_text, stage_text = (f" {angle_text:.0f}", f"Count : {count_text}", f" {stage_text}")

        if n_people > 1:
            extra_scale = n_people + 1
        else:
            extra_scale = n_people

        if not video_source_is_video:
            extra_scale = n_people + 4

        font_scale = (5 + (line_thickness / 10.0)) / extra_scale  ##befroe it was 10
        draw_angle = False

        # Draw angle
        (angle_text_width, angle_text_height), _ = cv2.getTextSize(angle_text, 0, font_scale, line_thickness)
        # angle_text_position = (int(center_kpt[0]), int(center_kpt[1])) #tuple(np.multiply(elbow_r, [640, 480]).astype(int)) (maybe use this to adjust to output frame size)
        # print(center_kpt, self.im.shape[0],self.im.shape[1])
        angle_text_position = (int(center_kpt[0]), int(center_kpt[
                                                           1]))  # tuple(np.multiply(elbow_r, [640, 480]).astype(int)) (maybe use this to adjust to output frame size)

        angle_background_position = (angle_text_position[0], angle_text_position[1] - angle_text_height - 5)
        angle_background_size = (angle_text_width + 2 * 5, angle_text_height + 2 * 5 + (line_thickness * 2))

        if draw_angle:
            cv2.rectangle(
                self.im,
                angle_background_position,
                (
                    angle_background_position[0] + angle_background_size[0],
                    angle_background_position[1] + angle_background_size[1],
                ),
                (255, 255, 255),
                -1,
            )
            cv2.putText(self.im, angle_text, angle_text_position, 0, font_scale, (0, 0, 0), line_thickness)

        # Draw Counts
        (count_text_width, count_text_height), _ = cv2.getTextSize(count_text, 0, font_scale, line_thickness)
        count_text_position = (angle_text_position[0], angle_text_position[1] + angle_text_height + 10)

        count_background_position = (
            angle_background_position[0],
            angle_background_position[1] + angle_background_size[1] + 5,
        )  ## before 10,10
        count_background_size = (count_text_width + 3, count_text_height + 2 + (line_thickness * 1))

        cv2.rectangle(
            self.im,
            count_background_position,
            (
                count_background_position[0] + count_background_size[0],
                count_background_position[1] + count_background_size[1],
            ),
            (255, 255, 255),
            -1,
        )
        cv2.putText(self.im, count_text, count_text_position, 0, font_scale, (0, 0, 0), line_thickness)

        # Draw Stage
        (stage_text_width, stage_text_height), _ = cv2.getTextSize(stage_text, 0, font_scale, line_thickness)
        stage_text_position = (int(center_kpt[0]), int(center_kpt[1]) + angle_text_height + count_text_height + 40)
        stage_background_position = (stage_text_position[0], stage_text_position[1] - stage_text_height - 5)

        ##befroe 10,10
        stage_background_size = (stage_text_width + 3, stage_text_height + 5)

        cv2.rectangle(
            self.im,
            stage_background_position,
            (
                stage_background_position[0] + stage_background_size[0],
                stage_background_position[1] + stage_background_size[1],
            ),
            (255, 255, 255),
            -1,
        )
        cv2.putText(self.im, stage_text, stage_text_position, 0, font_scale, (0, 0, 0), line_thickness)





class ExcerciseCounter:
    """ class to manage multi-person excercise counting."""

    def __init__(self):
        """Initializes the class with default vals."""

        # Image and line thickness
        self.im0 = None
        self.tf = None
        self.show_skeleton = False

        # Keypoints and count information
        self.keypoints = None
        self.poseup_angle = None
        self.posedown_angle = None
        self.threshold = 0.001
        self.upperbody_angle = None
        self.n_people = None

        # Store stage, count and angle information
        self.angle_r = None  # added
        self.angle_l = None  # added
        self.angle_upperbody_l = None
        self.angle_upperbody_r = None
        self.count = None
        self.stage = None
        self.pose_type = None

        # Visual Information
        self.view_img = False
        self.annotator = None
        self.video_source_is_video = None

        # Check if environment support imshow
        self.env_check = check_imshow(warn=True)

    def set_args(
            self,
            line_thickness=0.1,
            view_img=False,
            pose_type=None,
            video_source_is_video=False,
            show_skeleton=False,
    ):
        """
        Configures the line_thickness, save image and view image parameters.

        Args:
            kpts_to_check (list): 3 keypoints for counting
            line_thickness (int): Line thickness for bounding boxes.
            view_img (bool): display the im0
            pose_up_angle (float): Angle to set pose position up
            pose_down_angle (float): Angle to set pose position down
            pose_type (str): "pushup", "pullup" or "abworkout"
        """

        self.tf = line_thickness
        self.view_img = view_img
        self.pose_type = pose_type
        self.video_source_is_video = video_source_is_video
        self.show_skeleton = show_skeleton

    def start_counting(self, im0, results, frame_count):
        """
        Function used to count exercise.

        Args:
            im0 (ndarray): Current frame from the video stream.
            results (list): Pose estimation data
            frame_count (int): store current frame count
        """
        self.im0 = im0

        ## if this is the first frame then initialize the objects (counter,angles,stages) with their respective lengths
        if frame_count == 1:
            self.count = [0] * len(results[0])
            self.angle_r = [0] * len(results[0])  # added
            self.angle_l = [0] * len(results[0])  # added
            self.angle_upperbody_l = [0] * len(results[0])  # added
            self.angle_upperbody_r = [0] * len(results[0])  # added
            self.stage = ["-" for _ in results[0]]  ## for each element (name is irrelevant thats why _ )
        self.keypoints = results[0].keypoints.data
        self.annotator = Annotator_altered(im0, line_width=1)  ### to write on image (image and width of the writing)

        num_keypoints = len(results[0])  ## amount of people working out
        self.n_people = num_keypoints
        # print(f"# people identified: {num_keypoints}")

        # Resize self.angle, self.count, and self.stage if the number of keypoints has changed
        if len(self.angle_r) != num_keypoints:
            self.angle_r = [0] * num_keypoints
            self.angle_l = [0] * num_keypoints
            self.angle_upperbody_l = [0] * num_keypoints
            self.angle_upperbody_r = [0] * num_keypoints
            self.count = [0] * num_keypoints
            self.stage = ["-" for _ in range(num_keypoints)]

        if self.pose_type == "pushup":
            self.push_up_excercise()

        if self.pose_type == "pullup":
            self.pull_up_excercise()

        if self.pose_type == "crunch":
            self.crunch_excercise()

        if self.pose_type == "benchpress":
            print("test")

        if self.pose_type == "triceps_curl":
            print("test")

        if self.pose_type == "biceps_curl":
            print("test")

        if self.pose_type == "squad":
            None

        return self.im0

    def map_keypoint_to_body_part(self, k):

        body_part = ['nose', 'eye_l', 'eye_r', 'ear_l', 'ear_r', 'shoulder_l', 'shoulder_r',
                     'elbow_l', 'elbow_r', 'wrist_l', 'wrist_r', 'hip_l', 'hip_r', 'knee_l', 'knee_r', 'ankle_l',
                     'ankle_r']
        keypoints_index = {var: i for i, var in enumerate(body_part)}
        keypoints_values_map = {var: k[i] for i, var in enumerate(body_part)}

        return keypoints_values_map, keypoints_index

    def pull_up_excercise(self):

        ### Set thresholds
        self.poseup_angle = 70,
        self.posedown_angle = 140

        angle_keypoints_l = ["shoulder_l", "elbow_l", "wrist_l"]
        angle_keypoints_r = ["shoulder_r", "elbow_r", "wrist_r"]

        for ind, k in enumerate(reversed(self.keypoints)):

            body_parts, body_parts_index = self.map_keypoint_to_body_part(k)

            # self.im0 = self.annotator.draw_specific_points(k, shape=(640*0.5, 640*0.5), radius=1)
            # self.im0 = self.annotator.draw_specific_points(k, shape=(640*0.5, 640*0.5), radius=1)

            self.kpts_to_check_l = [body_parts_index.get(item, None) for item in angle_keypoints_l]
            self.kpts_to_check_r = [body_parts_index.get(item, None) for item in angle_keypoints_r]

            ### Measure angles
            self.angle_l[ind] = self.annotator.estimate_pose_angle(
                body_parts[angle_keypoints_l[0]].cpu(),  ## erster Keypoint
                body_parts[angle_keypoints_l[1]].cpu(),  ## zweiter keypoint bsp. elbow
                body_parts[angle_keypoints_l[2]].cpu(),  ## 3ter keypoints bspw. shoulder
            )

            self.angle_r[ind] = self.annotator.estimate_pose_angle(
                body_parts[angle_keypoints_r[0]].cpu(),  ## erster Keypoint
                body_parts[angle_keypoints_r[1]].cpu(),  ## zweiter keypoint bsp. elbow
                body_parts[angle_keypoints_r[2]].cpu(),  ## 3ter keypoints bspw. shoulder
            )

            ## Define conditions for states and counts
            # print(f"ind: {ind} ,stage: {self.stage[ind]}")
            condition_down_1 = self.angle_l[ind] > self.posedown_angle and self.angle_r[ind] > self.posedown_angle
            condition_down_2 = body_parts['wrist_l'][1].cpu() < body_parts['shoulder_l'][1].cpu() and \
                               body_parts['wrist_r'][1].cpu() < body_parts['shoulder_r'][1].cpu()

            condition_up_1 = self.angle_l[ind] < self.poseup_angle and self.angle_r[ind] < self.poseup_angle
            condition_up_2 = body_parts['nose'][1].cpu() < body_parts['wrist_l'][1].cpu() and body_parts['nose'][
                1].cpu() < body_parts['wrist_r'][1].cpu()
            condition_up_3 = self.stage[ind] == "down"

            ### Test conditions
            if condition_down_1 and condition_down_2:
                self.stage[ind] = "down"

            if condition_up_1 and condition_up_2 and condition_up_3:
                self.stage[ind] = "up"
                self.count[ind] += 1

            self.annotator.plot_angle_and_count_and_stage(
                angle_text=self.angle_l[ind],
                count_text=self.count[ind],
                stage_text=self.stage[ind],
                center_kpt=k[0],
                line_thickness=self.tf,
                n_people=self.n_people,
                video_source_is_video=self.video_source_is_video,
            )

            if self.show_skeleton:
                self.annotator.kpts(k, shape=(640, 640), radius=1, kpt_line=True)

    def push_up_excercise(self):

        ### Set thresholds
        self.poseup_angle = 140,
        self.posedown_angle = 100
        self.upperbody_angle = 120

        angle_keypoints_l = ["shoulder_l", "elbow_l", "wrist_l"]
        angle_keypoints_r = ["shoulder_r", "elbow_r", "wrist_r"]
        angle_keypoints_upper_body_r = ["ankle_r", "hip_r", "nose"]
        angle_keypoints_upper_body_l = ["ankle_l", "hip_l", "nose"]

        for ind, k in enumerate(reversed(self.keypoints)):

            body_parts, body_parts_index = self.map_keypoint_to_body_part(k)

            ### Measure angles
            self.angle_l[ind] = self.annotator.estimate_pose_angle(
                body_parts[angle_keypoints_l[0]].cpu(),  ## erster Keypoint
                body_parts[angle_keypoints_l[1]].cpu(),  ## zweiter keypoint bsp. elbow
                body_parts[angle_keypoints_l[2]].cpu(),  ## 3ter keypoints bspw. shoulder
            )

            self.angle_r[ind] = self.annotator.estimate_pose_angle(
                body_parts[angle_keypoints_r[0]].cpu(),  ## erster Keypoint
                body_parts[angle_keypoints_r[1]].cpu(),  ## zweiter keypoint bsp. elbow
                body_parts[angle_keypoints_r[2]].cpu(),  ## 3ter keypoints bspw. shoulder
            )

            self.angle_upperbody_r[ind] = self.annotator.estimate_pose_angle(
                body_parts[angle_keypoints_upper_body_r[0]].cpu(),  ## erster Keypoint
                body_parts[angle_keypoints_upper_body_r[1]].cpu(),  ## zweiter keypoint bsp. elbow
                body_parts[angle_keypoints_upper_body_r[2]].cpu(),  ## 3ter keypoints bspw. shoulder
            )

            self.angle_upperbody_l[ind] = self.annotator.estimate_pose_angle(
                body_parts[angle_keypoints_upper_body_l[0]].cpu(),  ## erster Keypoint
                body_parts[angle_keypoints_upper_body_l[1]].cpu(),  ## zweiter keypoint bsp. elbow
                body_parts[angle_keypoints_upper_body_l[2]].cpu(),  ## 3ter keypoints bspw. shoulder
            )

            ## Define conditions for states and counts
            condition_up_1 = self.angle_l[ind] > self.poseup_angle or self.angle_r[ind] > self.poseup_angle
            condition_up_2 = self.angle_upperbody_l[ind] > self.upperbody_angle or self.angle_upperbody_r[
                ind] > self.upperbody_angle

            condition_down_1 = self.angle_l[ind] < self.posedown_angle or self.angle_r[ind] < self.posedown_angle
            condition_down_2 = self.angle_upperbody_l[ind] > self.upperbody_angle or self.angle_upperbody_r[
                ind] > self.upperbody_angle
            condition_down_3 = self.stage[ind] == "up"

            print(f"{self.angle_l[ind]},{self.angle_r[ind]},{self.stage[ind]}")
            ### Test conditions
            if condition_up_1 and condition_up_2:
                self.stage[ind] = "up"
                # if ind ==1:
                # print("down")
            if condition_down_1 and condition_down_2 and condition_down_3:
                self.stage[ind] = "down"
                self.count[ind] += 1

            # print(self.video_source_is_video)
            self.annotator.plot_angle_and_count_and_stage(
                angle_text=self.angle_l[ind],
                count_text=self.count[ind],
                stage_text=self.stage[ind],
                center_kpt=k[0],
                line_thickness=self.tf,
                n_people=self.n_people,
                video_source_is_video=self.video_source_is_video,
            )

            if self.show_skeleton:
                self.annotator.kpts(k, shape=(640, 640), radius=1, kpt_line=True)

    def to_implement_excercise(self):

        ### Set thresholds
        self.poseup_angle = 140,
        self.posedown_angle = 60
        self.upperbody_angle = 120

        angle_keypoints_l = ["shoulder_l", "elbow_l", "wrist_l"]
        angle_keypoints_r = ["shoulder_r", "elbow_r", "wrist_r"]
        angle_keypoints_upper_body_r = ["ankle_r", "hip_r", "nose"]
        angle_keypoints_upper_body_l = ["ankle_l", "hip_l", "nose"]

        for ind, k in enumerate(reversed(self.keypoints)):

            body_parts, body_parts_index = self.map_keypoint_to_body_part(k)

            ### Measure angles
            self.angle_l[ind] = self.annotator.estimate_pose_angle(
                body_parts[angle_keypoints_l[0]].cpu(),  ## erster Keypoint
                body_parts[angle_keypoints_l[1]].cpu(),  ## zweiter keypoint bsp. elbow
                body_parts[angle_keypoints_l[2]].cpu(),  ## 3ter keypoints bspw. shoulder
            )

            self.angle_r[ind] = self.annotator.estimate_pose_angle(
                body_parts[angle_keypoints_r[0]].cpu(),  ## erster Keypoint
                body_parts[angle_keypoints_r[1]].cpu(),  ## zweiter keypoint bsp. elbow
                body_parts[angle_keypoints_r[2]].cpu(),  ## 3ter keypoints bspw. shoulder
            )

            self.angle_upperbody_r[ind] = self.annotator.estimate_pose_angle(
                body_parts[angle_keypoints_upper_body_r[0]].cpu(),  ## erster Keypoint
                body_parts[angle_keypoints_upper_body_r[1]].cpu(),  ## zweiter keypoint bsp. elbow
                body_parts[angle_keypoints_upper_body_r[2]].cpu(),  ## 3ter keypoints bspw. shoulder
            )

            self.angle_upperbody_l[ind] = self.annotator.estimate_pose_angle(
                body_parts[angle_keypoints_upper_body_l[0]].cpu(),  ## erster Keypoint
                body_parts[angle_keypoints_upper_body_l[1]].cpu(),  ## zweiter keypoint bsp. elbow
                body_parts[angle_keypoints_upper_body_l[2]].cpu(),  ## 3ter keypoints bspw. shoulder
            )

            ## Define conditions for states and counts
            condition_up_1 = self.angle_l[ind] > self.poseup_angle or self.angle_r[ind] > self.poseup_angle
            condition_up_2 = self.angle_upperbody_l[ind] > self.upperbody_angle or self.angle_upperbody_r[
                ind] > self.upperbody_angle

            condition_down_1 = self.angle_l[ind] < self.posedown_angle or self.angle_r[ind] < self.posedown_angle
            condition_down_2 = self.angle_upperbody_l[ind] > self.upperbody_angle or self.angle_upperbody_r[
                ind] > self.upperbody_angle
            condition_down_3 = self.stage[ind] == "up"

            ### Test conditions
            if condition_up_1 and condition_up_2:
                self.stage[ind] = "up"

            if condition_down_1 and condition_down_2 and condition_down_3:
                self.stage[ind] = "down"
                self.count[ind] += 1

            self.annotator.plot_angle_and_count_and_stage(
                angle_text=self.angle_l[ind],
                count_text=self.count[ind],
                stage_text=self.stage[ind],
                center_kpt=k[0],
                line_thickness=self.tf,
            )
            if self.show_skeleton:
                self.annotator.kpts(k, shape=(640, 640), radius=1, kpt_line=True)

    def crunch_excercise(self):

        ### Set thresholds
        self.poseup_angle = 110,
        self.posedown_angle = 130

        angle_keypoints_l = ["knee_l", "hip_l", "shoulder_l"]
        angle_keypoints_r = ["knee_r", "hip_r", "shoulder_r"]

        for ind, k in enumerate(reversed(self.keypoints)):

            body_parts, body_parts_index = self.map_keypoint_to_body_part(k)

            ### Measure angles
            self.angle_l[ind] = self.annotator.estimate_pose_angle(
                body_parts[angle_keypoints_l[0]].cpu(),  ## erster Keypoint
                body_parts[angle_keypoints_l[1]].cpu(),  ## zweiter keypoint bsp. elbow
                body_parts[angle_keypoints_l[2]].cpu(),  ## 3ter keypoints bspw. shoulder
            )

            self.angle_r[ind] = self.annotator.estimate_pose_angle(
                body_parts[angle_keypoints_r[0]].cpu(),  ## erster Keypoint
                body_parts[angle_keypoints_r[1]].cpu(),  ## zweiter keypoint bsp. elbow
                body_parts[angle_keypoints_r[2]].cpu(),  ## 3ter keypoints bspw. shoulder
            )

            ## Define conditions for states and counts
            condition_up_1 = self.angle_l[ind] < self.poseup_angle or self.angle_r[ind] < self.poseup_angle
            condition_up_2 = self.stage[ind] == "down"

            condition_down_1 = self.angle_l[ind] > self.posedown_angle or self.angle_r[ind] > self.posedown_angle

            # print( self.angle_l[ind], self.angle_r[ind])

            ### Test conditions
            # print(f"{self.angle_l[ind]},{self.angle_r[ind]},{self.stage[ind]}")

            if condition_up_1 and condition_up_2:
                self.stage[ind] = "up"
                self.count[ind] += 1
                print(f"count post: {self.count[ind]}")

            if condition_down_1:
                self.stage[ind] = "down"

            self.annotator.plot_angle_and_count_and_stage(
                angle_text=self.angle_l[ind],
                count_text=self.count[ind],
                stage_text=self.stage[ind],
                center_kpt=k[0],
                line_thickness=self.tf,
                n_people=self.n_people,
                video_source_is_video=self.video_source_is_video,
            )
            if self.show_skeleton:
                self.annotator.kpts(k, shape=(640, 640), radius=1, kpt_line=True)
