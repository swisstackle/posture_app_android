import time
import threading
from kivy.core.audio import SoundLoader

class AudioCuePlayer:
    def __init__(self, audio_files, thresholds, cooldown=10, global_cooldown=5, 
                 focus_timeout=15, success_frames=5, success_audio=None):
        """
        Initialize the AudioCuePlayer with audio files and thresholds.

        Args:
            audio_files (dict): Dictionary mapping posture issues to audio file paths.
            thresholds (dict): Dictionary mapping posture issues to threshold values.
            cooldown (float): Minimum time (in seconds) between playing the same audio cue.
            global_cooldown (float): Minimum time (in seconds) between any audio cues.
            focus_timeout (float): Time (in seconds) before abandoning a focus issue if no progress.
            success_frames (int): Number of consecutive frames below threshold to consider an issue resolved.
            success_audio (dict, optional): Dictionary mapping posture issues to success audio file paths.
        """
        self.audio_files = {issue: SoundLoader.load(file_path) for issue, file_path in audio_files.items()}
        self.thresholds = thresholds
        self.cooldown = cooldown
        self.global_cooldown = global_cooldown
        self.focus_timeout = focus_timeout
        self.success_frames = success_frames
        self.last_played = {issue: 0 for issue in audio_files}  # Per-issue cooldown tracking
        self.last_global_play = 0  # Timestamp of the last time any audio was played
        self.lock = threading.Lock()  # Thread-safe lock for audio playback
        
        # Focus mode state variables
        self.current_focus = None  # The issue currently in focus (or None if no focus)
        self.focus_start_time = 0  # When the current focus began
        self.focus_value_history = []  # Track values of current focus issue
        self.consecutive_good_frames = 0  # Count of frames below threshold
        self.positive_reinforcement_given = False  # Track if we've given positive feedback
        
        # Use provided success audio or set a default with general feedback only
        if success_audio:
            self.success_audio = {issue: SoundLoader.load(file_path) for issue, file_path in success_audio.items()}
        else:
            self.success_audio = {'general': SoundLoader.load('audio/good_job.mp3')}
        
        # Flag to disable audio cues (useful during calibration)
        self.quiet_mode = False
    
    def evaluate_posture_metrics(self, metrics):
        """
        Evaluate all posture metrics and determine which issue to focus on.
        All issues are treated as equally important.
        
        Args:
            metrics (dict): Dictionary with current values for all posture metrics.
        
        Returns:
            str: The issue to focus on, or None if all metrics are good.
        """
        # If we're already focusing on an issue, stick with it
        if self.current_focus is not None:
            return self.current_focus
        
        # Find issues that exceed their thresholds
        issues = []
        for issue, value in metrics.items():
            if issue in self.thresholds and value > self.thresholds[issue]:
                # Calculate severity as percentage over threshold
                severity = (value - self.thresholds[issue]) / self.thresholds[issue]
                issues.append((issue, value, severity))
        
        # If no issues, return None
        if not issues:
            return None
            
        # Sort by severity (how much they exceed their threshold by percentage)
        issues.sort(key=lambda x: x[2], reverse=True)
        
        return issues[0][0]
    
    def start_focus(self, issue, initial_value):
        """
        Start focusing on a specific posture issue.
        
        Args:
            issue (str): The posture issue to focus on.
            initial_value (float): The initial value of the issue metric.
        """
        self.current_focus = issue
        self.focus_start_time = time.time()
        self.focus_value_history = [initial_value]
        self.consecutive_good_frames = 0
        self.positive_reinforcement_given = False
        
        # Give initial guidance (if not in quiet mode)
        if not self.quiet_mode and issue in self.audio_files:
            self.audio_files[issue].play()
            self.last_played[issue] = time.time()
            self.last_global_play = time.time()
    
    def update_focus(self, issue, value):
        """
        Update the focus mode with the latest value for the current focus issue.
        
        Args:
            issue (str): The posture issue being tracked.
            value (float): The current value of the issue metric.
            
        Returns:
            bool: True if focus should continue, False if resolved or abandoned.
        """
        # Make sure this is our focus issue
        if issue != self.current_focus:
            return True
            
        # Add to history
        self.focus_value_history.append(value)
        
        # Check if value is now below threshold (good posture)
        if value <= self.thresholds[issue]:
            self.consecutive_good_frames += 1
            
            # Check if we've maintained good posture long enough to consider it resolved
            if self.consecutive_good_frames >= self.success_frames:
                # Issue resolved! Give positive feedback if we haven't already
                if not self.positive_reinforcement_given and not self.quiet_mode:
                    success_key = issue if issue in self.success_audio else 'general'
                    if success_key in self.success_audio and self.success_audio[success_key]:
                        self.success_audio[success_key].play()
                    self.positive_reinforcement_given = True
                
                # After giving positive feedback, exit focus mode
                if self.consecutive_good_frames >= self.success_frames + 10:  # Give extra frames after success
                    self.exit_focus()
                    return False
        else:
            # Reset consecutive good frames counter
            self.consecutive_good_frames = 0
            
            # Check if we should repeat guidance (based on cooldown)
            if not self.quiet_mode and issue in self.audio_files:
                current_time = time.time()
                if current_time - self.last_played[issue] >= self.cooldown:
                    self.audio_files[issue].play()
                    self.last_played[issue] = current_time
                    self.last_global_play = current_time
            
        # Check if we've timed out on this focus issue
        if time.time() - self.focus_start_time > self.focus_timeout:
            # If no improvement after timeout, abandon this focus
            self.exit_focus()
            return False
            
        return True
    
    def exit_focus(self):
        """Exit the current focus mode."""
        self.current_focus = None
        self.focus_value_history = []
        self.consecutive_good_frames = 0
    
    def play_cue(self, issue, current_value):
        """
        Play the audio cue for a specific posture issue if conditions are met.
        This legacy method is maintained for compatibility.
        
        Args:
            issue (str): The posture issue (e.g., 'upper_back').
            current_value (float): The current value of the posture metric.
        """
        # Don't play cues if in quiet mode
        if self.quiet_mode:
            return
            
        if issue not in self.audio_files or issue not in self.thresholds:
            return

        if current_value <= self.thresholds[issue]:
            return  # Value doesn't exceed threshold, no cue needed

        with self.lock:
            current_time = time.time()
            if current_time - self.last_global_play < self.global_cooldown:
                return
            if current_time - self.last_played[issue] < self.cooldown:
                return
            
            # Only play if we're not in focus mode, or this is our focus issue
            if self.current_focus is None or self.current_focus == issue:
                self.audio_files[issue].play()
                self.last_played[issue] = current_time
                self.last_global_play = current_time
