# BacklineAnalyzer.py
import cv2
import numpy as np
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
import tensorflow as tf
from tflite_runtime.interpreter import Interpreter
from AudioCuePlayer import AudioCuePlayer
import os

# MoveNet keypoint dictionary (from MoveNet documentation)
KEYPOINT_DICT = {
    'nose': 0,
    'left_eye': 1,
    'right_eye': 2,
    'left_ear': 3,
    'right_ear': 4,
    'left_shoulder': 5,
    'right_shoulder': 6,
    'left_elbow': 7,
    'right_elbow': 8,
    'left_wrist': 9,
    'right_wrist': 10,
    'left_hip': 11,
    'right_hip': 12,
    'left_knee': 13,
    'right_knee': 14,
    'left_ankle': 15,
    'right_ankle': 16
}

def calculate_curvature(points):
    """
    Calculate discrete curvature for a set of points using the angle between consecutive segments.
    
    Args:
        points (numpy.ndarray): Array of shape (n, 2) with x, y coordinates.
    
    Returns:
        list: Curvature values in radians for each interior point.
    """
    if len(points) < 3:
        return []

    desired_window = 21
    poly_order = 2
    n = len(points)
    max_window = n if n % 2 == 1 else n - 1
    window_size = min(desired_window, max_window)
    
    if window_size <= poly_order:
        window_size = max_window

    x_smooth = savgol_filter(points[:, 0], window_size, poly_order)
    y_smooth = savgol_filter(points[:, 1], window_size, poly_order)
    points = np.column_stack((x_smooth, y_smooth))
    
    curvatures = []
    for i in range(1, len(points) - 1):
        p1 = points[i - 1]
        p2 = points[i]
        p3 = points[i + 1]
        v1 = p2 - p1
        v2 = p3 - p2
        angle = np.arctan2(v2[1], v2[0]) - np.arctan2(v1[1], v1[0])
        angle = np.abs((angle + np.pi) % (2 * np.pi) - np.pi)
        curvatures.append(angle)
    
    return curvatures

class Landmark:
    """Simple class to mimic MediaPipe's landmark structure."""
    def __init__(self, x, y, score):
        self.x = x
        self.y = y
        self.score = score

class BacklineAnalyzer:
    def __init__(self, model_path, audio_files=None, thresholds=None, version=2, cooldown=10):
        """
        Initialize the BacklineAnalyzer with a MoveNet model.
        
        Args:
            model_path (str): Path to the MoveNet TensorFlow Lite model file.
            audio_files (dict): Audio files for cues.
            thresholds (dict): Thresholds for posture metrics.
            version (int): Version of the backline extraction algorithm (default is 2).
            cooldown (float): Cooldown for audio cues in seconds.
        """
        # Load MoveNet model
        self.interpreter = Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.input_size = 192  # MoveNet Lightning input size
        self.version = version
        self.upper_curvature_history = []
        self.lower_curvature_history = []
        if audio_files and thresholds:
            self.audio_player = AudioCuePlayer(audio_files, thresholds, cooldown)
        else:
            self.audio_player = None

    def process_frame(self, frame):
        """Orchestrate frame processing by calling modular methods."""
        result_img = frame.copy()
        contour_points = self._get_contour_points(frame)
        if contour_points is None:
            return result_img

        landmarks = self._get_pose_landmarks(frame)
        if landmarks is None:
            return result_img

        back_points = self._trace_backline(contour_points, landmarks, frame)
        if back_points is None:
            return result_img

        smoothed_back_points = self._interpolate_and_smooth(back_points)
        upper_back, lower_back = self._split_backline(smoothed_back_points)
        avg_upper_curvature, avg_lower_curvature = self._calculate_and_update_curvature(upper_back, lower_back)
        
        # Calculate shoulder-hip alignment ratio and points
        shoulder_hip_ratio, shoulder_point, hip_point = self._calculate_shoulder_hip_ratio(landmarks, frame)
        
        # Calculate forward/backward lean
        lean_angle, lean_shoulder_point, lean_hip_point = self._calculate_lean(landmarks, frame)
        
        # Play audio cues if audio_player is initialized
        if self.audio_player:
            self.audio_player.play_cue('upper_back', avg_upper_curvature)
            self.audio_player.play_cue('shoulder_hip', shoulder_hip_ratio)
            self.audio_player.play_cue('lower_back', avg_lower_curvature)
            
            if lean_angle > 0:
                self.audio_player.play_cue('forward_lean', abs(lean_angle))
            elif lean_angle < 0:
                self.audio_player.play_cue('backward_lean', abs(lean_angle))
        
        # Visualize results
        self._visualize(result_img, smoothed_back_points, upper_back, lower_back, 
                        avg_upper_curvature, avg_lower_curvature, shoulder_hip_ratio, 
                        shoulder_point, hip_point, lean_angle)

        return result_img

    def _get_contour_points(self, frame):
        """Extract silhouette contour points using green screen masking."""
        lower_green = np.array([35, 20, 80])
        upper_green = np.array([85, 255, 255])
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_green, upper_green)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            print("No silhouette detected.")
            return None
        silhouette = max(contours, key=cv2.contourArea)
        contour_points = silhouette.squeeze()
        if not isinstance(contour_points, np.ndarray) or len(contour_points.shape) != 2 or contour_points.shape[1] != 2 or len(contour_points) < 3:
            print("Invalid contour: not enough points or incorrect shape.")
            return None
        return contour_points

    def _get_pose_landmarks(self, frame):
        """Detect pose landmarks using MoveNet."""
        # Prepare input image for MoveNet
        input_image = tf.expand_dims(frame, axis=0)
        input_image = tf.image.resize_with_pad(input_image, self.input_size, self.input_size)
        input_image = tf.cast(input_image, dtype=tf.uint8)

        # Run inference
        self.interpreter.set_tensor(self.input_details[0]['index'], input_image.numpy())
        self.interpreter.invoke()
        keypoints_with_scores = self.interpreter.get_tensor(self.output_details[0]['index'])

        # Convert MoveNet output to a list of Landmark objects
        landmarks = []
        for i in range(17):
            y = keypoints_with_scores[0, 0, i, 0]  # Normalized y
            x = keypoints_with_scores[0, 0, i, 1]  # Normalized x
            score = keypoints_with_scores[0, 0, i, 2]
            landmarks.append(Landmark(x, y, score))
        return landmarks

    def _trace_backline(self, contour_points, landmarks, frame):
        """Trace the backline from neck to hip using contour and landmarks."""
        left_shoulder = [landmarks[KEYPOINT_DICT['left_shoulder']].x * frame.shape[1],
                         landmarks[KEYPOINT_DICT['left_shoulder']].y * frame.shape[0]]
        right_shoulder = [landmarks[KEYPOINT_DICT['right_shoulder']].x * frame.shape[1],
                          landmarks[KEYPOINT_DICT['right_shoulder']].y * frame.shape[0]]
        left_hip = [landmarks[KEYPOINT_DICT['left_hip']].x * frame.shape[1],
                    landmarks[KEYPOINT_DICT['left_hip']].y * frame.shape[0]]
        right_hip = [landmarks[KEYPOINT_DICT['right_hip']].x * frame.shape[1],
                     landmarks[KEYPOINT_DICT['right_hip']].y * frame.shape[0]]
        hip_mid = [(left_hip[0] + right_hip[0]) / 2, (left_hip[1] + right_hip[1]) / 2]
        neck = [(left_shoulder[0] + right_shoulder[0]) / 2, (left_shoulder[1] + right_shoulder[1]) / 2]

        neck_y = neck[1]
        indices = np.where(np.abs(contour_points[:, 1] - neck_y) < 5)[0]
        if len(indices) == 0:
            print("No contour points found near neck y-level.")
            return None
        start_idx = indices[np.argmin(contour_points[indices, 0])]

        back_points = []
        current_idx = start_idx
        visited = set()
        while current_idx not in visited:
            visited.add(current_idx)
            point = contour_points[current_idx]
            if point[1] > hip_mid[1]:
                break
            back_points.append(point)
            current_idx = (current_idx + 1) % len(contour_points)
        
        back_points = np.array(back_points)
        if len(back_points) < 3:
            print("Not enough points to process backline.")
            return None
        return back_points

    def _interpolate_and_smooth(self, back_points):
        """Interpolate and smooth backline points."""
        num_points = 50
        distances = np.cumsum(np.sqrt(np.sum(np.diff(back_points, axis=0)**2, axis=1)))
        distances = np.insert(distances, 0, 0)
        total_distance = distances[-1]
        if total_distance == 0:
            print("Total distance is zero, cannot interpolate.")
            return back_points
        interp_distances = np.linspace(0, total_distance, num_points)
        interp_x = interp1d(distances, back_points[:, 0], kind='linear')(interp_distances)
        interp_y = interp1d(distances, back_points[:, 1], kind='linear')(interp_distances)
        back_points_interp = np.column_stack((interp_x, interp_y))

        window_size = 11
        poly_order = 2
        if len(back_points_interp) >= window_size:
            x_smooth = savgol_filter(back_points_interp[:, 0], window_size, poly_order)
            y_smooth = savgol_filter(back_points_interp[:, 1], window_size, poly_order)
            return np.column_stack((x_smooth, y_smooth))
        return back_points_interp

    def _split_backline(self, smoothed_back_points):
        """Split backline into upper and lower segments."""
        upper_ratio = 0.4
        num_upper = int(len(smoothed_back_points) * upper_ratio)
        return smoothed_back_points[:num_upper], smoothed_back_points[num_upper:]

    def _calculate_and_update_curvature(self, upper_back, lower_back):
        """Calculate curvature and update history."""
        upper_curvatures = calculate_curvature(upper_back)
        lower_curvatures = calculate_curvature(lower_back)
        avg_upper = np.mean(upper_curvatures) if upper_curvatures else 0
        avg_lower = np.mean(lower_curvatures) if lower_curvatures else 0
        
        self.upper_curvature_history.append(avg_upper)
        self.lower_curvature_history.append(avg_lower)
        if len(self.upper_curvature_history) > 5:
            self.upper_curvature_history.pop(0)
        if len(self.lower_curvature_history) > 5:
            self.lower_curvature_history.pop(0)
        
        return avg_upper, avg_lower

    def _calculate_shoulder_hip_ratio(self, landmarks, frame):
        """Calculate the alignment ratio between right shoulder and right hip."""
        right_shoulder = landmarks[KEYPOINT_DICT['right_shoulder']]
        right_hip = landmarks[KEYPOINT_DICT['right_hip']]
        
        shoulder_x = right_shoulder.x * frame.shape[1]
        shoulder_y = right_shoulder.y * frame.shape[0]
        hip_x = right_hip.x * frame.shape[1]
        hip_y = right_hip.y * frame.shape[0]
        
        x_diff = abs(shoulder_x - hip_x)
        y_diff = abs(shoulder_y - hip_y)
        
        if y_diff == 0:
            alignment_ratio = 0
        else:
            alignment_ratio = x_diff / y_diff
        
        return alignment_ratio, (shoulder_x, shoulder_y), (hip_x, hip_y)
        
    def _calculate_lean(self, landmarks, frame):
        """Calculate the forward/backward lean using right shoulder and right hip."""
        right_shoulder = landmarks[KEYPOINT_DICT['right_shoulder']]
        right_hip = landmarks[KEYPOINT_DICT['right_hip']]
        
        shoulder_x = right_shoulder.x * frame.shape[1]
        shoulder_y = right_shoulder.y * frame.shape[0]
        hip_x = right_hip.x * frame.shape[1]
        hip_y = right_hip.y * frame.shape[0]
        
        hip_to_shoulder_vector = [shoulder_x - hip_x, shoulder_y - hip_y]
        vertical_vector = [0, -1]
        
        hip_to_shoulder_norm = np.sqrt(hip_to_shoulder_vector[0]**2 + hip_to_shoulder_vector[1]**2)
        if hip_to_shoulder_norm == 0:
            return 0, (shoulder_x, shoulder_y), (hip_x, hip_y)
            
        hip_to_shoulder_unit = [hip_to_shoulder_vector[0]/hip_to_shoulder_norm, 
                                hip_to_shoulder_vector[1]/hip_to_shoulder_norm]
        
        dot_product = hip_to_shoulder_unit[0] * vertical_vector[0] + hip_to_shoulder_unit[1] * vertical_vector[1]
        dot_product = max(-1.0, min(1.0, dot_product))
        
        angle_rad = np.arccos(dot_product)
        angle_deg = angle_rad * (180.0 / np.pi)
        
        if shoulder_x > hip_x:
            angle_deg = angle_deg  # Positive = forward lean
        else:
            angle_deg = -angle_deg  # Negative = backward lean
            
        return angle_deg, (shoulder_x, shoulder_y), (hip_x, hip_y)

    def _visualize(self, result_img, smoothed_points, upper_back, lower_back, 
                   avg_upper, avg_lower, shoulder_hip_ratio, 
                   shoulder_point, hip_point, lean_angle=0):
        """Draw backline, curvature info, alignment ratios, and landmark points on the frame."""
        for x, y in smoothed_points:
            cv2.circle(result_img, (int(x), int(y)), 3, (255, 255, 0), -1)

        if len(upper_back) >= 2:
            for i in range(len(upper_back) - 1):
                cv2.line(result_img,
                         (int(upper_back[i, 0]), int(upper_back[i, 1])),
                         (int(upper_back[i + 1, 0]), int(upper_back[i + 1, 1])),
                         (255, 0, 0),  # Blue for upper back
                         thickness=3)
        if len(lower_back) >= 2:
            for i in range(len(lower_back) - 1):
                cv2.line(result_img,
                         (int(lower_back[i, 0]), int(lower_back[i, 1])),
                         (int(lower_back[i + 1, 0]), int(lower_back[i + 1, 1])),
                         (255, 0, 255),  # Magenta for lower back
                         thickness=3)

        moving_avg_upper = np.mean(self.upper_curvature_history) if self.upper_curvature_history else avg_upper
        moving_avg_lower = np.mean(self.lower_curvature_history) if self.lower_curvature_history else avg_lower

        cv2.putText(result_img, f"Upper Back Curvature: {moving_avg_upper:.4f} rad",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(result_img, f"Lower Back Curvature: {moving_avg_lower:.4f} rad",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
        cv2.putText(result_img, f"Shoulder-Hip Alignment: {shoulder_hip_ratio:.4f}",
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        lean_text = "Forward" if lean_angle > 0 else "Backward" if lean_angle < 0 else "Neutral"
        lean_color = (0, 165, 255) if lean_angle > 0 else (255, 255, 0) if lean_angle < 0 else (255, 255, 255)
        cv2.putText(result_img, f"{lean_text} Lean: {abs(lean_angle):.1f} degrees",
                    (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, lean_color, 2)

        cv2.circle(result_img, (int(shoulder_point[0]), int(shoulder_point[1])), 5, (0, 0, 255), -1)
        cv2.circle(result_img, (int(hip_point[0]), int(hip_point[1])), 5, (0, 255, 255), -1)
        
        cv2.line(result_img, 
                 (int(hip_point[0]), int(hip_point[1])), 
                 (int(hip_point[0]), int(hip_point[1] - 200)),
                 (255, 255, 255),
                 thickness=2,
                 lineType=cv2.LINE_AA)
        
        cv2.line(result_img,
                 (int(hip_point[0]), int(hip_point[1])),
                 (int(shoulder_point[0]), int(shoulder_point[1])),
                 lean_color,
                 thickness=2,
                 lineType=cv2.LINE_AA)
