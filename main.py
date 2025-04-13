# main.py
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.camera import Camera
from kivy.clock import Clock
from kivy.graphics.texture import Texture
import cv2
import numpy as np
import os
from BacklineAnalyzer import BacklineAnalyzer

class PostureApp(App):
    def build(self):
        """Initialize the Kivy app layout and components."""
        self.layout = BoxLayout(orientation='vertical')
        
        # Camera widget
        self.camera = Camera(resolution=(640, 480), play=True)
        self.layout.add_widget(self.camera)
        
        # Image widget to display processed frames
        self.img = Image()
        self.layout.add_widget(self.img)
        
        # Audio files and thresholds (same as original)
        audio_files = {
            'upper_back': 'audio/straighten_upper_back.mp3',
            'shoulder_hip': 'audio/align_shoulder_hip.mp3',
            'lower_back': 'audio/straighten_lower_back.mp3',
            'forward_lean': 'audio/reduce_forward_lean.mp3',
            'backward_lean': 'audio/reduce_backward_lean.mp3'
        }
        thresholds = {
            'upper_back': 0.03,
            'shoulder_hip': 0.1,
            'lower_back': 0.03,
            'forward_lean': 25,
            'backward_lean': 25
        }
        
        # Initialize BacklineAnalyzer with MoveNet model
        model_path = os.path.join(os.path.dirname(__file__), 'model.tflite')
        self.analyzer = BacklineAnalyzer(model_path, audio_files=audio_files, thresholds=thresholds, version=2, cooldown=10)
        
        # Schedule frame processing
        Clock.schedule_interval(self.update, 1.0 / 15.0)  # 15 FPS for mobile performance
        return self.layout

    def update(self, dt):
        """Process camera frames and update the display."""
        # Get frame from camera texture
        texture = self.camera.texture
        if texture:
            frame = np.frombuffer(texture.pixels, dtype='uint8')
            frame = frame.reshape(texture.height, texture.width, 4)  # RGBA
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)  # Convert to BGR
            
            # Apply original transformations
            frame = cv2.resize(frame, (1280, 720))
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            
            # Process frame
            processed_frame = self.analyzer.process_frame(frame)
            
            # Convert to texture and update image widget
            texture = self.frame_to_texture(processed_frame)
            self.img.texture = texture

    def frame_to_texture(self, frame):
        """Convert frame to Kivy texture."""
        buf = cv2.flip(frame, 0).tobytes()
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        return texture

if __name__ == '__main__':
    PostureApp().run()
