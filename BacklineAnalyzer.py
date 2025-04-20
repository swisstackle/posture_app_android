# BacklineAnalyzer.py
import cv2
import numpy as np
import tensorflow as tf
from tflite_runtime.interpreter import Interpreter
from AudioCuePlayer import AudioCuePlayer
import os
import time

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

    n = len(points)
    window_size = min(21, n)
    if window_size % 2 == 0:  # Ensure window size is odd
        window_size -= 1

    # Create moving average window
    window = np.ones(window_size) / window_size
    
    # Apply moving average smoothing to x and y coordinates
    x_smooth = np.convolve(points[:, 0], window, mode='valid')
    y_smooth = np.convolve(points[:, 1], window, mode='valid')
    
    # Pad the results to maintain original length
    pad_size = (len(points) - len(x_smooth)) // 2
    x_smooth = np.pad(x_smooth, (pad_size, len(points) - len(x_smooth) - pad_size), 'edge')
    y_smooth = np.pad(y_smooth, (pad_size, len(points) - len(y_smooth) - pad_size), 'edge')
    
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
    def __init__(self, model_path, audio_files=None, thresholds=None, version=2, cooldown=10, success_audio=None):
        """
        Initialize the BacklineAnalyzer with a MoveNet model.
        
        Args:
            model_path (str): Path to the MoveNet TensorFlow Lite model file.
            audio_files (dict): Audio files for cues.
            thresholds (dict): Thresholds for posture metrics.
            version (int): Version of the backline extraction algorithm (default is 2).
            cooldown (float): Cooldown for audio cues in seconds.
            success_audio (dict, optional): Success audio files.
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
            self.audio_player = AudioCuePlayer(audio_files, thresholds, cooldown, 
                                          focus_timeout=15, success_frames=5,
                                          success_audio=success_audio)
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
        
        shoulder_hip_ratio, shoulder_point, hip_point = self._calculate_shoulder_hip_ratio(landmarks, frame)
        lean_angle, lean_shoulder_point, lean_hip_point = self._calculate_lean(landmarks, frame)
        
        # Collect all posture metrics in one place
        metrics = {
            'upper_back': avg_upper_curvature,
            'lower_back': avg_lower_curvature,
            'shoulder_hip': shoulder_hip_ratio,
            'forward_lean': max(0, lean_angle)  # Only positive values for forward lean
        }
        
        # Handle focus mode if audio player exists
        focus_info = None
        if self.audio_player:
            # Evaluate which issue to focus on
            focus_issue = self.audio_player.evaluate_posture_metrics(metrics)
            
            if focus_issue:
                # If we have a focus issue
                if self.audio_player.current_focus is None:
                    # Start focusing on this issue
                    self.audio_player.start_focus(focus_issue, metrics[focus_issue])
                else:
                    # Update the current focus
                    self.audio_player.update_focus(self.audio_player.current_focus, 
                                                metrics[self.audio_player.current_focus])
                
                # Save focus information for visualization
                focus_info = {
                    'issue': self.audio_player.current_focus,
                    'metric': metrics[self.audio_player.current_focus] if self.audio_player.current_focus in metrics else 0,
                    'threshold': self.audio_player.thresholds[self.audio_player.current_focus] 
                            if self.audio_player.current_focus in self.audio_player.thresholds else 0,
                    'success_count': self.audio_player.consecutive_good_frames,
                    'success_target': self.audio_player.success_frames,
                    'duration': time.time() - self.audio_player.focus_start_time if self.audio_player.current_focus else 0
                }
        
        self._visualize(result_img, smoothed_back_points, upper_back, lower_back, 
                       avg_upper_curvature, avg_lower_curvature, shoulder_hip_ratio, 
                       shoulder_point, hip_point, lean_angle, focus_info)

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
        """
        Trace the backline starting from the highest point of the greenscreen contour
        and moving downward to the hip level.
        """
        # Get hip midpoint to know where to stop tracing
        left_hip = [landmarks[KEYPOINT_DICT['left_hip']].x * frame.shape[1],
                   landmarks[KEYPOINT_DICT['left_hip']].y * frame.shape[0]]
        right_hip = [landmarks[KEYPOINT_DICT['right_hip']].x * frame.shape[1],
                    landmarks[KEYPOINT_DICT['right_hip']].y * frame.shape[0]]
        hip_mid = [(left_hip[0] + right_hip[0]) / 2, (left_hip[1] + right_hip[1]) / 2]
        
        # Get right shoulder x-coordinate to help filter out front-facing points
        right_shoulder_x = landmarks[KEYPOINT_DICT['right_shoulder']].x * frame.shape[1]
        
        # Find the top points of the contour (lowest y-coordinates)
        # Only consider points to the left of the right shoulder (back side of profile)
        back_side_points = contour_points[contour_points[:, 0] < right_shoulder_x]
        if len(back_side_points) == 0:
            print("No contour points found on back side of body profile.")
            return None
        
        # Find the top 10% of points by y-coordinate
        y_sorted_indices = np.argsort(back_side_points[:, 1])
        top_indices = y_sorted_indices[:max(int(len(y_sorted_indices) * 0.1), 5)]  # Take at least 5 points
        
        # From these top points, find the leftmost one
        start_idx = np.argmin(back_side_points[top_indices, 0])
        start_point = back_side_points[top_indices[start_idx]]
        
        # Now find this point in the original contour points array
        start_point_idx = np.where((contour_points[:, 0] == start_point[0]) & 
                                  (contour_points[:, 1] == start_point[1]))[0]
        
        if len(start_point_idx) == 0:
            print("Could not find start point in contour.")
            return None
        
        start_idx = start_point_idx[0]
        
        # Trace downward from the start point
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
        """Interpolate and smooth backline points using moving average."""
        num_points = 50
        # Calculate cumulative distances along the back_points
        diffs = np.diff(back_points, axis=0)
        segment_lengths = np.sqrt(np.sum(diffs**2, axis=1))
        distances = np.insert(np.cumsum(segment_lengths), 0, 0)
        total_distance = distances[-1]

        if total_distance == 0:
            print("Total distance is zero, cannot interpolate.")
            return back_points

        # Create new evenly spaced distances for interpolation
        interp_distances = np.linspace(0, total_distance, num_points)

        # Interpolate the x and y coordinates using np.interp (replacing interp1d)
        interp_x = np.interp(interp_distances, distances, back_points[:, 0])
        interp_y = np.interp(interp_distances, distances, back_points[:, 1])
        back_points_interp = np.column_stack((interp_x, interp_y))

        # Apply moving average smoothing (replacing savgol_filter)
        window_size = 11
        if len(back_points_interp) >= window_size:
            window = np.ones(window_size) / window_size
            x_smoothed_valid = np.convolve(back_points_interp[:, 0], window, mode='valid')
            y_smoothed_valid = np.convolve(back_points_interp[:, 1], window, mode='valid')
            
            # Pad to preserve original length
            pad_size = (len(back_points_interp) - len(x_smoothed_valid)) // 2
            x_smooth = np.pad(x_smoothed_valid, (pad_size, len(back_points_interp) - len(x_smoothed_valid) - pad_size), 'edge')
            y_smooth = np.pad(y_smoothed_valid, (pad_size, len(back_points_interp) - len(y_smoothed_valid) - pad_size), 'edge')
            
            return np.column_stack((x_smooth, y_smooth))

        return back_points_interp

    def _split_backline(self, smoothed_back_points):
        """Split backline into upper and lower segments."""
        upper_ratio = 0.5  # 50/50 split
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
        alignment_ratio = x_diff / y_diff if y_diff != 0 else 0
        
        return alignment_ratio, (shoulder_x, shoulder_y), (hip_x, hip_y)
        
    def _calculate_lean(self, landmarks, frame):
        """Calculate the forward lean using right shoulder and right hip."""
        right_shoulder = landmarks[KEYPOINT_DICT['right_shoulder']]
        right_hip = landmarks[KEYPOINT_DICT['right_hip']]
        
        shoulder_x = right_shoulder.x * frame.shape[1]
        shoulder_y = right_shoulder.y * frame.shape[0]
        hip_x = right_hip.x * frame.shape[1]
        hip_y = right_hip.y * frame.shape[0]
        
        hip_to_shoulder_vector = [shoulder_x - hip_x, shoulder_y - hip_y]
        vertical_vector = [0, -1]
        norm = np.sqrt(hip_to_shoulder_vector[0]**2 + hip_to_shoulder_vector[1]**2)
        if norm == 0:
            return 0, (shoulder_x, shoulder_y), (hip_x, hip_y)
        unit_vector = [hip_to_shoulder_vector[0] / norm, hip_to_shoulder_vector[1] / norm]
        dot_product = unit_vector[0] * vertical_vector[0] + unit_vector[1] * vertical_vector[1]
        dot_product = max(-1.0, min(1.0, dot_product))
        angle_rad = np.arccos(dot_product)
        angle_deg = angle_rad * (180.0 / np.pi)
        if shoulder_x > hip_x:
            lean_angle = angle_deg
        else:
            lean_angle = 0  # Ignore backward lean by setting it to 0
        return lean_angle, (shoulder_x, shoulder_y), (hip_x, hip_y)

    def _visualize(self, result_img, smoothed_points, upper_back, lower_back, 
                  avg_upper, avg_lower, shoulder_hip_ratio, 
                  shoulder_point, hip_point, lean_angle=0, focus_info=None):
        """Draw backline, curvature info, alignment ratios, and landmark points on the frame."""
        # Display backline points
        for x, y in smoothed_points:
            cv2.circle(result_img, (int(x), int(y)), 3, (255, 255, 0), -1)

        # Draw upper back line
        if len(upper_back) >= 2:
            for i in range(len(upper_back) - 1):
                cv2.line(result_img,
                         (int(upper_back[i, 0]), int(upper_back[i, 1])),
                         (int(upper_back[i + 1, 0]), int(upper_back[i + 1, 1])),
                         (255, 0, 0), 3)
        # Draw lower back line
        if len(lower_back) >= 2:
            for i in range(len(lower_back) - 1):
                cv2.line(result_img,
                         (int(lower_back[i, 0]), int(lower_back[i, 1])),
                         (int(lower_back[i + 1, 0]), int(lower_back[i + 1, 1])),
                         (255, 0, 255), 3)

        # Calculate moving averages for smoother metrics
        moving_avg_upper = np.mean(self.upper_curvature_history) if self.upper_curvature_history else avg_upper
        moving_avg_lower = np.mean(self.lower_curvature_history) if self.lower_curvature_history else avg_lower

        # Standard metrics display
        cv2.putText(result_img, f"Upper Back Curvature: {moving_avg_upper:.4f} rad",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(result_img, f"Lower Back Curvature: {moving_avg_lower:.4f} rad",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        cv2.putText(result_img, f"Shoulder-Hip Alignment: {shoulder_hip_ratio:.4f}",
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        lean_text = "Forward" if lean_angle > 0 else "Neutral"
        lean_color = (0, 165, 255) if lean_angle > 0 else (255, 255, 255)
        cv2.putText(result_img, f"{lean_text} Lean: {abs(lean_angle if lean_angle > 0 else 0):.1f} degrees",
                    (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, lean_color, 2)

        # Draw focus mode information if available
        if focus_info is not None and 'issue' in focus_info and focus_info['issue'] is not None:
            # Create a semi-transparent overlay for focus mode
            overlay = result_img.copy()
            
            # Draw a highlight box for the focus area
            issue = focus_info['issue']
            if issue == 'upper_back':
                pts = upper_back.astype(np.int32)
                cv2.polylines(overlay, [pts], False, (0, 0, 255), 5)
                focus_area_text = "Focus: Upper Back"
                focus_point = (int(upper_back[len(upper_back)//2, 0]), int(upper_back[len(upper_back)//2, 1]))
            elif issue == 'lower_back':
                pts = lower_back.astype(np.int32)
                cv2.polylines(overlay, [pts], False, (255, 0, 255), 5)
                focus_area_text = "Focus: Lower Back"
                focus_point = (int(lower_back[len(lower_back)//2, 0]), int(lower_back[len(lower_back)//2, 1]))
            elif issue == 'shoulder_hip':
                cv2.line(overlay, 
                         (int(shoulder_point[0]), int(shoulder_point[1])),
                         (int(hip_point[0]), int(hip_point[1])),
                         (0, 255, 0), 5)
                focus_area_text = "Focus: Shoulder-Hip Alignment"
                focus_point = (int((shoulder_point[0] + hip_point[0])/2), 
                               int((shoulder_point[1] + hip_point[1])/2))
            elif issue == 'forward_lean':
                cv2.line(overlay,
                         (int(hip_point[0]), int(hip_point[1])),
                         (int(shoulder_point[0]), int(shoulder_point[1])),
                         lean_color, 5)
                focus_area_text = "Focus: Forward Lean"
                focus_point = (int((shoulder_point[0] + hip_point[0])/2), 
                               int((shoulder_point[1] + hip_point[1])/2))
            else:
                focus_area_text = f"Focus: {issue}"
                focus_point = (result_img.shape[1]//2, result_img.shape[0]//2)
            
            # Add focus area text near the focus point
            cv2.putText(overlay, focus_area_text,
                        (focus_point[0] - 100, focus_point[1] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Blend the overlay with the original image
            alpha = 0.7  # Transparency factor
            cv2.addWeighted(overlay, alpha, result_img, 1 - alpha, 0, result_img)
            
            # Draw progress bar for maintaining good posture
            progress_width = 200
            progress_height = 20
            progress_x = result_img.shape[1] - progress_width - 10
            progress_y = 30
            
            # Draw progress bar background
            cv2.rectangle(result_img, (progress_x, progress_y), 
                         (progress_x + progress_width, progress_y + progress_height),
                         (100, 100, 100), -1)
            
            # Draw progress
            if 'success_target' in focus_info and focus_info['success_target'] > 0:
                success_count = focus_info.get('success_count', 0)
                progress = min(1.0, success_count / focus_info['success_target'])
                cv2.rectangle(result_img, (progress_x, progress_y), 
                             (int(progress_x + progress_width * progress), progress_y + progress_height),
                             (0, 255, 0), -1)
            
            # Draw progress text
            success_count = focus_info.get('success_count', 0)
            success_target = focus_info.get('success_target', 1)
            cv2.putText(result_img, f"Progress: {success_count}/{success_target}",
                       (progress_x, progress_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Add focus duration
            duration = focus_info.get('duration', 0.0)
            cv2.putText(result_img, f"Focus Time: {duration:.1f}s",
                       (progress_x, progress_y + progress_height + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Add actionable instruction based on the focus issue
            instruction_y = result_img.shape[0] - 50
            if issue == 'upper_back':
                instruction = "Straighten your upper back"
            elif issue == 'lower_back':
                instruction = "Flatten your lower back"
            elif issue == 'shoulder_hip':
                instruction = "Align shoulders over hips"
            elif issue == 'forward_lean':
                instruction = "Reduce forward lean"
            else:
                instruction = "Correct your posture"
                
            cv2.putText(result_img, instruction,
                       (result_img.shape[1]//2 - 150, instruction_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        
        # Always draw landmark points
        cv2.circle(result_img, (int(shoulder_point[0]), int(shoulder_point[1])), 5, (0, 0, 255), -1)
        cv2.circle(result_img, (int(hip_point[0]), int(hip_point[1])), 5, (0, 255, 255), -1)
        
        cv2.line(result_img, 
                 (int(hip_point[0]), int(hip_point[1])), 
                 (int(hip_point[0]), int(hip_point[1] - 200)),
                 (255, 255, 255), 2, cv2.LINE_AA)
        
        cv2.line(result_img,
                 (int(hip_point[0]), int(hip_point[1])),
                 (int(shoulder_point[0]), int(shoulder_point[1])),
                 lean_color, 2, cv2.LINE_AA)
