# main.py
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.uix.camera import Camera
from kivy.uix.label import Label
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.core.window import Window
import cv2
import numpy as np
import os
import time
from BacklineAnalyzer import BacklineAnalyzer
from Calibrator import Calibrator

class PostureApp(App):
    def build(self):
        """Initialize the Kivy app layout and components."""
        # Main layout
        self.layout = BoxLayout(orientation='vertical')
        
        # Status label
        self.status_label = Label(text="PostureApp Ready", size_hint=(1, 0.1))
        self.layout.add_widget(self.status_label)
        
        # Camera widget
        self.camera = Camera(resolution=(640, 480), play=True)
        self.layout.add_widget(self.camera)
        
        # Image widget to display processed frames
        self.img = Image(size_hint=(1, 0.7))
        self.layout.add_widget(self.img)
        
        # Button layout
        button_layout = BoxLayout(orientation='horizontal', size_hint=(1, 0.1))
        
        # Calibration button
        self.calibrate_button = Button(text="Calibrate", on_press=self.start_calibration)
        button_layout.add_widget(self.calibrate_button)
        
        # Quit button
        quit_button = Button(text="Quit", on_press=self.quit_app)
        button_layout.add_widget(quit_button)
        
        self.layout.add_widget(button_layout)
        
        # Audio files and thresholds
        audio_files = {
            'upper_back': 'audio/straighten_upper_back.mp3',
            'shoulder_hip': 'audio/align_shoulder_hip.mp3',
            'lower_back': 'audio/straighten_lower_back.mp3',
            'forward_lean': 'audio/reduce_forward_lean.mp3'
        }
        
        # Success audio files
        success_audio = {
            'upper_back': 'audio/good_job.mp3',
            'shoulder_hip': 'audio/good_job.mp3',
            'lower_back': 'audio/good_job.mp3',
            'forward_lean': 'audio/good_job.mp3',
            'general': 'audio/good_job.mp3'
        }
        
        # Default thresholds (will be overridden if calibration is performed or loaded)
        default_thresholds = {
            'upper_back': 0.03,
            'shoulder_hip': 0.1,
            'lower_back': 0.03,
            'forward_lean': 25
        }
        
        # Initialize BacklineAnalyzer with MoveNet model
        model_path = os.path.join(os.path.dirname(__file__), 'model.tflite')
        self.analyzer = BacklineAnalyzer(model_path, audio_files=audio_files, 
                                         thresholds=default_thresholds, version=2, 
                                         cooldown=5, success_audio=success_audio)
        
        # Initialize the calibrator
        self.calibrator = Calibrator(self.analyzer, calibration_duration=10)
        
        # Try to load existing calibration if file exists
        calibration_file = os.path.join(os.path.dirname(__file__), 'calibration.json')
        if os.path.exists(calibration_file):
            if self.calibrator.load_thresholds(calibration_file):
                self.status_label.text = "Loaded calibrated thresholds"
        
        # Application state variables
        self.calibration_mode = False
        self.countdown_mode = False
        self.countdown_start_time = 0
        self.countdown_duration = 5  # seconds
        self.calibration_complete = False
        
        # Schedule frame processing
        Clock.schedule_interval(self.update, 1.0 / 15.0)  # 15 FPS for mobile performance
        
        return self.layout

    def update(self, dt):
        """Process camera frames and update the display."""
        # Get frame from camera texture
        texture = self.camera.texture
        if texture:
            frame = self.texture_to_frame(texture)
            
            # Apply transformations - keep frame in correct orientation for processing
            frame = cv2.resize(frame, (640, 480))
            
            # Process frame based on current mode
            if self.countdown_mode:
                # Calculate remaining countdown time
                elapsed = time.time() - self.countdown_start_time
                remaining = self.countdown_duration - elapsed
                
                if remaining <= 0:
                    # Countdown finished, start calibration
                    self.countdown_mode = False
                    self.calibration_mode = True
                    self.calibrator.start_calibration()
                    self.status_label.text = "Calibration in progress..."
                else:
                    # Draw countdown overlay
                    overlay = frame.copy()
                    cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
                    
                    # Add text to overlay
                    cv2.putText(overlay, "GET INTO POSITION", 
                              (frame.shape[1]//2 - 150, frame.shape[0]//2 - 50), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                    cv2.putText(overlay, f"Calibration starts in: {int(remaining)}s", 
                              (frame.shape[1]//2 - 150, frame.shape[0]//2 + 50), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
                    
                    # Show the processed frame with overlay
                    alpha = 0.7
                    cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0, frame)
                    
            elif self.calibration_mode:
                # Process frame through calibrator
                frame, is_complete = self.calibrator.process_frame(frame)
                
                if is_complete:
                    self.calibration_mode = False
                    self.calibration_complete = True
                    
                    # Save calibration results
                    calibration_file = os.path.join(os.path.dirname(__file__), 'calibration.json')
                    self.calibrator.save_thresholds(calibration_file)
                    self.status_label.text = "Calibration complete!"
            else:
                # Normal analysis mode
                frame = self.analyzer.process_frame(frame)
                
                # If calibration was just completed, show a message
                if self.calibration_complete:
                    cv2.putText(frame, "Calibration Complete!", 
                              (frame.shape[1]//2 - 150, 50), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    # Only show this message for a short time
                    if time.time() - self.calibrator.start_time > self.calibrator.calibration_duration + 3:
                        self.calibration_complete = False
            
            # Convert to texture and update image widget
            texture = self.frame_to_texture(frame)
            self.img.texture = texture

    def texture_to_frame(self, texture):
        """Convert Kivy texture to OpenCV frame."""
        frame = np.frombuffer(texture.pixels, dtype='uint8')
        frame = frame.reshape(texture.height, texture.width, 4)  # RGBA
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)  # Convert to BGR
        return frame

    def frame_to_texture(self, frame):
        """Convert OpenCV frame to Kivy texture."""
        buf = cv2.flip(frame, 0).tobytes()
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        return texture
        
    def start_calibration(self, instance):
        """Start the calibration process."""
        self.countdown_mode = True
        self.countdown_start_time = time.time()
        self.status_label.text = "Get into position for calibration..."
        
    def quit_app(self, instance):
        """Exit the application."""
        App.get_running_app().stop()

if __name__ == '__main__':
    PostureApp().run()
