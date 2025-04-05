# AudioCuePlayer.py
import time
from kivy.core.audio import SoundLoader
import threading

class AudioCuePlayer:
    def __init__(self, audio_files, thresholds, cooldown=10, global_cooldown=5):
        """
        Initialize the AudioCuePlayer with audio files and thresholds.

        Args:
            audio_files (dict): Dictionary mapping posture issues to audio file paths.
            thresholds (dict): Dictionary mapping posture issues to threshold values.
            cooldown (float): Minimum time (in seconds) between playing the same audio cue.
            global_cooldown (float): Minimum time (in seconds) between any audio cues.
        """
        self.audio_files = {issue: SoundLoader.load(file) for issue, file in audio_files.items()}
        self.thresholds = thresholds
        self.cooldown = cooldown
        self.global_cooldown = global_cooldown
        self.last_played = {issue: 0 for issue in audio_files}
        self.last_global_play = 0
        self.lock = threading.Lock()

    def play_cue(self, issue, current_value):
        """
        Play the audio cue for a specific posture issue if conditions are met.

        Args:
            issue (str): The posture issue (e.g., 'upper_back').
            current_value (float): The current value of the posture metric.
        """
        if issue not in self.audio_files or issue not in self.thresholds:
            return

        if current_value <= self.thresholds[issue]:
            return

        with self.lock:
            current_time = time.time()

            if current_time - self.last_global_play < self.global_cooldown:
                return

            if current_time - self.last_played[issue] < self.cooldown:
                return

            self.audio_files[issue].play()
            self.last_played[issue] = current_time
            self.last_global_play = current_time
