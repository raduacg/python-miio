import sys
import threading
import numpy as np
import warnings
import time
import queue
import json
import cv2
import win32gui
import psutil
import os

from sklearn.cluster import KMeans
from sklearn.exceptions import ConvergenceWarning
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QPushButton,
    QWidget, QLabel, QComboBox, QSlider, QColorDialog, QMessageBox,
    QListWidget, QInputDialog, QTabWidget, QListWidgetItem, QCheckBox
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor
from PIL import ImageGrab
from miio.miot_device import MiotDevice
from miio.exceptions import DeviceException
from pynput import mouse, keyboard

from PyQt5 import QtGui, QtCore  # Import for dark mode

import logging

# Configure logging
logging.basicConfig(
    filename='command_log.txt',
    level=logging.INFO,
    format='%(asctime)s:%(levelname)s:%(message)s'
)


# Set CPU affinity to the last two cores
process = psutil.Process(os.getpid())
num_cores = os.cpu_count()
last_core_index = num_cores - 1  # Zero-based index
process.cpu_affinity([last_core_index - 1, last_core_index])


class OverlayWidget(QWidget):
    def __init__(self, rect):
        super().__init__()
        self.rect = rect
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setGeometry(rect)

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setPen(QtGui.QPen(Qt.red, 3))
        painter.drawRect(0, 0, self.rect.width(), self.rect.height())


class DeviceControlWidget(QWidget):
    def __init__(self, device_info, app):
        super().__init__()
        self.app = app
        self.device_info = device_info
        self.device = device_info["device"]
        self.name = device_info["name"]
        self.region = device_info.get("region", {
            "margin_left": 0,
            "margin_right": 0,
            "margin_top": 0,
            "margin_bottom": 0
        })

        # Create the layout and controls for the device
        self.layout = QVBoxLayout()

        # Device name
        self.layout.addWidget(QLabel(f"Device: {self.name}"))

        # On/Off Buttons
        self.add_button("Turn On", lambda: self.send_command(2, 1, True))
        self.add_button("Turn Off", lambda: self.send_command(2, 1, False))

        # Mode Selection
        mode_label = QLabel("Mode:")
        self.mode_combo = QComboBox()
        self.mode_combo.addItems([str(i) for i in range(0, 9)])  # Modes 0 to 8
        self.mode_combo.currentIndexChanged.connect(self.change_mode)
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(mode_label)
        mode_layout.addWidget(self.mode_combo)
        self.layout.addLayout(mode_layout)

        # Brightness Control
        brightness_label = QLabel("Brightness:")
        self.brightness_slider = QSlider(Qt.Horizontal)
        self.brightness_slider.setMinimum(1)
        self.brightness_slider.setMaximum(100)
        self.brightness_slider.setValue(100)  # Default value
        self.brightness_slider.valueChanged.connect(self.change_brightness)
        brightness_layout = QHBoxLayout()
        brightness_layout.addWidget(brightness_label)
        brightness_layout.addWidget(self.brightness_slider)
        self.layout.addLayout(brightness_layout)

        # Color Selection
        self.color_button = QPushButton("Select Color")
        self.color_button.clicked.connect(self.select_color)
        self.layout.addWidget(self.color_button)

        # Ambient Color Sync Button
        self.ambient_button = QPushButton("Start Ambient Sync")
        self.ambient_button.setCheckable(True)
        self.ambient_button.clicked.connect(self.toggle_ambient_mode)
        self.layout.addWidget(self.ambient_button)

        # Margins Sliders
        self.layout.addWidget(QLabel("Margin Left (%)"))
        self.margin_left_slider = QSlider(Qt.Horizontal)
        self.margin_left_slider.setMinimum(0)
        self.margin_left_slider.setMaximum(100)
        self.margin_left_slider.setValue(self.region.get("margin_left", 0))
        self.margin_left_slider.valueChanged.connect(self.update_margins)
        self.layout.addWidget(self.margin_left_slider)

        self.layout.addWidget(QLabel("Margin Right (%)"))
        self.margin_right_slider = QSlider(Qt.Horizontal)
        self.margin_right_slider.setMinimum(0)
        self.margin_right_slider.setMaximum(100)
        self.margin_right_slider.setValue(self.region.get("margin_right", 0))
        self.margin_right_slider.valueChanged.connect(self.update_margins)
        self.layout.addWidget(self.margin_right_slider)

        self.layout.addWidget(QLabel("Margin Top (%)"))
        self.margin_top_slider = QSlider(Qt.Horizontal)
        self.margin_top_slider.setMinimum(0)
        self.margin_top_slider.setMaximum(100)
        self.margin_top_slider.setValue(self.region.get("margin_top", 0))
        self.margin_top_slider.valueChanged.connect(self.update_margins)
        self.layout.addWidget(self.margin_top_slider)

        self.layout.addWidget(QLabel("Margin Bottom (%)"))
        self.margin_bottom_slider = QSlider(Qt.Horizontal)
        self.margin_bottom_slider.setMinimum(0)
        self.margin_bottom_slider.setMaximum(100)
        self.margin_bottom_slider.setValue(self.region.get("margin_bottom", 0))
        self.margin_bottom_slider.valueChanged.connect(self.update_margins)
        self.layout.addWidget(self.margin_bottom_slider)

        # Flash Region Button
        self.flash_region_button = QPushButton("Flash Selected Region")
        self.flash_region_button.clicked.connect(self.flash_selected_region)
        self.layout.addWidget(self.flash_region_button)

        # Coordinates Display
        self.coordinates_label = QLabel("Selected Region Margins:")
        self.layout.addWidget(self.coordinates_label)
        self.update_coordinates_label()

        # Select Region Button (Optional)
        # self.select_region_button = QPushButton("Select Region with Mouse")
        # self.select_region_button.clicked.connect(self.select_region_with_mouse_clicks)
        # self.layout.addWidget(self.select_region_button)

        # Action Buttons
        self.add_button("Brightness Up", lambda: self.send_action(3, 1))
        self.add_button("Brightness Down", lambda: self.send_action(3, 2))
        self.add_button("Toggle TV", lambda: self.send_action(3, 6))
        self.add_button("Toggle Rhythm", lambda: self.send_action(3, 12))

        # Set Red/Blue Button
        self.add_button("Set Red/Blue", self.set_red_blue)
        
        # DIY ID Buttons
        self.add_button("DIY ID 1", lambda: self.send_command(3, 12, 1))
        self.add_button("DIY ID 2", lambda: self.send_command(3, 12, 2))
        self.add_button("DIY ID 3", lambda: self.send_command(3, 12, 3))

        self.setLayout(self.layout)

        # Ambient Mode Variables
        self.ambient_thread = None
        self.stop_ambient = threading.Event()
        self.command_queue = queue.Queue()
        self.previous_color = np.array([0, 0, 0], dtype=np.float32)
        self.current_brightness = 100
        self.target_brightness = 100
        self.saturation_factor = 1.2  # Increase saturation by 20%
        self.brightness_factor = 1.3  # Increase brightness by 30%

    def add_button(self, label, command):
        button = QPushButton(label)
        button.clicked.connect(command)
        self.layout.addWidget(button)

    def send_command(self, siid, piid, value):
        try:
            self.device.set_property_by(siid, piid, value)
            #logging.info(f"{self.name}: send_command(siid={siid}, piid={piid}, value={value})")
        except DeviceException as e:
            logging.error(f"{self.name}: Failed to send_command(siid={siid}, piid={piid}, value={value}) - {e}")
    
    def send_action(self, siid, aiid, params=None):
        try:
            if params is None:
                params = []
            self.device.call_action_by(siid, aiid, params)
            #logging.info(f"{self.name}: send_action(siid={siid}, aiid={aiid}, params={params})")
        except DeviceException as e:
            logging.error(f"{self.name}: Failed to send_action(siid={siid}, aiid={aiid}, params={params}) - {e}")


    def change_mode(self):
        mode_value = int(self.mode_combo.currentText())
        self.send_command(2, 2, mode_value)  # siid=2, piid=2

    def change_brightness(self):
        brightness_value = self.brightness_slider.value()
        self.send_command(2, 3, brightness_value)  # siid=2, piid=3

    def select_color(self):
        color = QColorDialog.getColor()
        if color.isValid():
            # Get RGB values
            r = color.red()
            g = color.green()
            b = color.blue()
            # Convert to integer
            color_value = (r << 16) + (g << 8) + b
            self.send_command(2, 4, int(color_value))  # siid=2, piid=4

    def toggle_ambient_mode(self):
        if self.ambient_button.isChecked():
            # Start ambient mode
            self.stop_ambient.clear()
            # Start the ambient loop in a new thread
            self.ambient_thread = threading.Thread(target=self.ambient_loop, name=f"AmbientThread-{self.name}")
            self.ambient_thread.start()
        else:
            # Stop ambient mode
            self.stop_ambient_mode()

    def stop_ambient_mode(self):
        self.stop_ambient.set()
        if self.ambient_thread:
            self.ambient_thread.join()
        self.ambient_button.setChecked(False)

    def update_margins(self):
        self.region["margin_left"] = self.margin_left_slider.value()
        self.region["margin_right"] = self.margin_right_slider.value()
        self.region["margin_top"] = self.margin_top_slider.value()
        self.region["margin_bottom"] = self.margin_bottom_slider.value()
        self.update_coordinates_label()
        self.flash_selected_region()
        self.app.save_device_configs()

    def update_coordinates_label(self):
        self.coordinates_label.setText(f"Margins (%):\n"
                                       f"Left: {self.region['margin_left']}%, "
                                       f"Right: {self.region['margin_right']}%, "
                                       f"Top: {self.region['margin_top']}%, "
                                       f"Bottom: {self.region['margin_bottom']}%")

    def flash_selected_region(self):
        # Calculate the region based on margins and screen size
        screens = self.app.screens
        # For simplicity, using primary screen
        screen = screens[0]
        rect = screen.geometry()
        screen_width = rect.width()
        screen_height = rect.height()

        # Apply margins
        left = int(self.region["margin_left"] / 100 * screen_width)
        right = int(screen_width - self.region["margin_right"] / 100 * screen_width)
        top = int(self.region["margin_top"] / 100 * screen_height)
        bottom = int(screen_height - self.region["margin_bottom"] / 100 * screen_height)

        # Create a transparent window to display the rectangle
        overlay_rect = QtCore.QRect(left, top, right - left, bottom - top)
        self.overlay = OverlayWidget(overlay_rect)
        self.overlay.show()

        # Close the overlay after 2 seconds
        QtCore.QTimer.singleShot(2000, self.overlay.close)

    def ambient_loop(self):
        sample_interval = 0.18  # 180 milliseconds
        color_transition_interval = 0.36  # 360 milliseconds
        sample_buffer = []
        last_color_update_time = time.time()
        brightness_history = []  # To store brightness values for smoothing
        max_brightness_samples = 4  # Number of samples to consider for moving average
    
        while not self.stop_ambient.is_set():
            start_time = time.time()
            try:
                if self.app.alt_key_pressed:
                    # ALT key is pressed, do not capture or send any commands
                    time.sleep(sample_interval)
                    continue
                elif self.app.paused:
                    # Clear the command queue
                    with self.command_queue.mutex:
                        self.command_queue.queue.clear()
                    # Handle paused state
                    # Use black color to fade to black
                    blended_color = np.array([0, 0, 0], dtype=np.uint8)
                    # Add to command queue with current timestamp
                    self.command_queue.put((blended_color, 0, time.time()))
                    time.sleep(sample_interval)
                    continue
    
                # Capture the screen
                img = ImageGrab.grab(all_screens=True)
                img_np = np.array(img)
    
                # Calculate the region based on margins and screen size
                screens = self.app.screens
                # For simplicity, using primary screen
                screen = screens[0]
                rect = screen.geometry()
                screen_width = rect.width()
                screen_height = rect.height()
    
                # Apply margins
                left = int(self.region["margin_left"] / 100 * screen_width)
                right = int(screen_width - self.region["margin_right"] / 100 * screen_width)
                top = int(self.region["margin_top"] / 100 * screen_height)
                bottom = int(screen_height - self.region["margin_bottom"] / 100 * screen_height)
    
                # Crop the image to the selected region
                cropped_img = img_np[top:bottom, left:right]
    
                # Resize image to speed up processing
                img_small = cv2.resize(cropped_img, (160, 80))  # Reduced size for faster processing
                height, width, _ = img_small.shape
    
                # Create weights
                x_weights = np.ones(width)
                y_weights = np.ones(height)
    
                x_center_start = int(width * (1/3))
                x_center_end = int(width * (2/3))
    
                y_center_start = int(height * (1/3))
                y_center_end = int(height * (2/3))
    
                x_weights[x_center_start:x_center_end] *= 1.2  # Increase weight by 20%
                y_weights[y_center_start:y_center_end] *= 1.2  # Increase weight by 20%
    
                weights = np.outer(y_weights, x_weights).flatten()
    
                # Flatten pixels
                pixels = img_small.reshape(-1, 3)
    
                # Calculate brightness for each pixel
                brightness = pixels.mean(axis=1)
    
                # Calculate overall brightness
                weighted_brightness = np.sum(brightness * weights) / np.sum(weights)
    
                # Append to brightness history
                brightness_history.append(weighted_brightness)
                if len(brightness_history) > max_brightness_samples:
                    brightness_history.pop(0)
    
                # Calculate moving average of brightness
                avg_brightness = np.mean(brightness_history)
    
                # Scale brightness to 0-100 range
                scaled_brightness = np.clip((avg_brightness / 255) * 100, 10, 100)  # Minimum brightness of 10
    
                # Determine the dominant color
                if len(pixels) == 0:
                    # No pixels, use previous color
                    dominant_color = self.previous_color.astype(np.uint8)
                else:
                    # Compute weighted average color
                    weighted_sum = np.sum(pixels * weights[:, np.newaxis], axis=0)
                    total_weight = np.sum(weights)
                    avg_color = weighted_sum / total_weight
                    dominant_color = avg_color.astype(np.uint8)
    
                    # Increase saturation and brightness
                    dominant_color = self.increase_saturation(dominant_color)
    
                    # Enhance specific colors
                    dominant_color = self.enhance_specific_colors(dominant_color)
    
                # Smooth color transition
                blended_color = self.blend_colors(self.previous_color, dominant_color, alpha=0.32)
                self.previous_color = blended_color
    
                # Ensure color values are valid
                blended_color = np.clip(blended_color, 0, 255).astype(np.uint8)
    
                # Add to sample buffer
                sample_buffer.append((blended_color, scaled_brightness))
    
                # Keep only the last three samples
                if len(sample_buffer) > 3:
                    sample_buffer.pop(0)
    
                # Send color updates every 360ms
                if time.time() - last_color_update_time >= color_transition_interval:
                    # Average the samples
                    avg_color = np.mean([s[0] for s in sample_buffer], axis=0).astype(np.uint8)
                    avg_brightness = int(np.mean([s[1] for s in sample_buffer]))
                    # Reset sample buffer and update last_color_update_time
                    sample_buffer = []
                    last_color_update_time = time.time()
                    # Add to command queue with current timestamp
                    self.command_queue.put((avg_color, avg_brightness, time.time()))
    
            except Exception as e:
                print(f"Error in ambient_loop for {self.name}: {e}")
                pass  # Suppress exceptions to improve performance
    
            # Wait for the next sample
            elapsed = time.time() - start_time
            sleep_time = max(0, sample_interval - elapsed)
            self.stop_ambient.wait(sleep_time)

    def initiate_fade_transition(self, to_brightness):
        if to_brightness < self.current_brightness:
            # Transition to black in stages: current_brightness -> 40% -> 0%
            self.brightness_steps = [self.current_brightness, 40, 0]
            self.fade_step_durations = [0.8, 0.8]  # Durations in seconds
        else:
            # Transition from black to color: 0% -> 40% -> 100%
            self.brightness_steps = [0, 40, 100]
            self.fade_step_durations = [0.8, 0.8]
        self.fade_transition_start_time = time.time()
        self.current_brightness_index = 0
        self.next_brightness_step_time = self.fade_transition_start_time + self.fade_step_durations[0]

    def handle_fade_transition(self):
        if hasattr(self, 'fade_transition_start_time') and self.fade_transition_start_time is not None:
            current_time = time.time()
            if current_time >= self.next_brightness_step_time:
                # Move to next brightness step
                self.current_brightness_index += 1
                if self.current_brightness_index >= len(self.brightness_steps):
                    # Transition complete
                    self.current_brightness = self.brightness_steps[-1]
                    self.fade_transition_start_time = None
                else:
                    self.current_brightness = self.brightness_steps[self.current_brightness_index]
                    # Schedule next step
                    self.next_brightness_step_time += self.fade_step_durations[self.current_brightness_index - 1]

    def increase_saturation(self, color_rgb):
        color_array = np.uint8([[color_rgb]])  # Convert to 1x1 image
        hsv = cv2.cvtColor(color_array, cv2.COLOR_RGB2HSV)
        h, s, v = hsv[0][0]

        # Adjust saturation
        s = np.clip(int(s * self.saturation_factor), 0, 255)

        # Adjust brightness (Value channel)
        v = np.clip(int(v * self.brightness_factor), 0, 255)

        hsv[0][0][1] = s
        hsv[0][0][2] = v
        color_array = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        return color_array[0][0]

    def enhance_specific_colors(self, color_rgb):
        color_array = np.uint8([[color_rgb]])  # Convert to 1x1 image
        hsv = cv2.cvtColor(color_array, cv2.COLOR_RGB2HSV)
        h, s, v = hsv[0][0]

        # Define HSV ranges for specific colors

        # Trees (Green)
        lower_green = np.array([40, 40, 40])
        upper_green = np.array([90, 255, 255])

        # Campfire (Orange/Red)
        lower_orange = np.array([10, 100, 100])
        upper_orange = np.array([25, 255, 255])

        # Neon lights: High saturation and brightness
        lower_neon = np.array([0, 150, 150])
        upper_neon = np.array([179, 255, 255])  # Full hue range

        # Check if the color falls within any of the ranges
        if self.is_color_in_range(h, lower_green, upper_green) or \
                self.is_color_in_range(h, lower_orange, upper_orange) or \
                self.is_color_in_range(h, lower_neon, upper_neon):
            # Enhance the color by increasing brightness and saturation
            s = np.clip(int(s * self.saturation_factor * 1.6), 0, 255)
            v = np.clip(int(v * self.brightness_factor * 1.4), 0, 255)
            hsv[0][0][1] = s
            hsv[0][0][2] = v
            color_array = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
            return color_array[0][0]
        else:
            return color_rgb

    def is_color_in_range(self, hue, lower_hsv, upper_hsv):
        # Handle the case when hue wraps around
        if lower_hsv[0] <= upper_hsv[0]:
            # Normal range
            in_range = lower_hsv[0] <= hue <= upper_hsv[0]
        else:
            # Hue wraps around
            in_range = hue >= lower_hsv[0] or hue <= upper_hsv[0]
        return in_range

    def blend_colors(self, color1, color2, alpha):
        """Blend two colors with a given alpha (0.0 to 1.0)"""
        return (1 - alpha) * color1 + alpha * color2

    def set_red_blue(self):
        # Example value; the actual format depends on the device's API
        # Here, we assume that 'diy-color' accepts a string defining the colors per segment
        red = "FF0000"
        blue = "0000FF"
        # Assuming the strip has 20 segments, 10 red, 10 blue
        value = red * 10 + blue * 10
        self.send_command(3, 9, value)

    def command_sender_loop(self):
        last_avg_color = None
        last_avg_brightness = None
        command_timeout = 2.0  # Commands older than 2 seconds are discarded
        max_color_distance = np.linalg.norm([255, 255, 255])  # Maximum possible color distance
    
        while True:
            try:
                # Wait for the next command
                avg_color, brightness, command_time = self.command_queue.get()
                current_time = time.time()
                # Discard old commands
                if current_time - command_time > command_timeout:
                    continue
                # If ALT key is pressed, do not send any commands
                if self.app.alt_key_pressed:
                    continue
                # Compute color similarity
                if last_avg_color is not None:
                    color_diff = np.linalg.norm(avg_color - last_avg_color)
                    similarity = 1 - (color_diff / max_color_distance)
                else:
                    similarity = 0  # Ensure first command is sent
    
                brightness_diff = abs(brightness - (last_avg_brightness or 0))
    
                # Decide whether to send the command
                if (hasattr(self, 'fade_transition_start_time') and self.fade_transition_start_time is not None) \
                        or similarity < 0.95 or brightness_diff > 5:
                    # Convert avg_color to integer value
                    r, g, b = avg_color
                    color_value = (int(r) << 16) + (int(g) << 8) + int(b)
    
                    # Send the color command
                    self.send_command(2, 4, int(color_value))
    
                    # Send the brightness command
                    self.send_command(2, 3, int(brightness))
    
                    last_avg_color = avg_color
                    last_avg_brightness = brightness
                else:
                    # Skip sending the command
                    logging.info(f"{self.name}: Skipped sending command due to high similarity ({similarity * 100:.2f}%)")
            except Exception as e:
                logging.error(f"Error in command_sender_loop for {self.name}: {e}")

    def start_command_sender_thread(self):
        # Start command sender thread
        self.command_sender_thread = threading.Thread(target=self.command_sender_loop, name=f"CommandSenderThread-{self.name}")
        self.command_sender_thread.daemon = True
        self.command_sender_thread.start()


class DeviceControlApp(QMainWindow):
    def __init__(self):
        super().__init__()

        # Set dark mode
        app.setStyle('Fusion')
        dark_palette = QtGui.QPalette()
        dark_palette.setColor(QtGui.QPalette.Window, QtGui.QColor(53, 53, 53))
        dark_palette.setColor(QtGui.QPalette.WindowText, QtCore.Qt.white)
        dark_palette.setColor(QtGui.QPalette.Base, QtGui.QColor(25, 25, 25))
        dark_palette.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor(53, 53, 53))
        dark_palette.setColor(QtGui.QPalette.ToolTipBase, QtCore.Qt.white)
        dark_palette.setColor(QtGui.QPalette.ToolTipText, QtCore.Qt.white)
        dark_palette.setColor(QtGui.QPalette.Text, QtCore.Qt.white)
        dark_palette.setColor(QtGui.QPalette.Button, QtGui.QColor(53, 53, 53))
        dark_palette.setColor(QtGui.QPalette.ButtonText, QtCore.Qt.white)
        dark_palette.setColor(QtGui.QPalette.BrightText, QtCore.Qt.red)
        dark_palette.setColor(QtGui.QPalette.Link, QtGui.QColor(42, 130, 218))
        dark_palette.setColor(QtGui.QPalette.Highlight, QtGui.QColor(42, 130, 218))
        dark_palette.setColor(QtGui.QPalette.HighlightedText, QtCore.Qt.black)
        app.setPalette(dark_palette)

        self.setWindowTitle("Smart Device Controller")
        self.setGeometry(100, 100, 600, 800)

        # Main layout
        self.layout = QVBoxLayout()

        # Load configurations
        self.load_device_configs()
        self.initialize_devices()

        # Initialize screens
        self.screens = app.screens()

        # Tab widget for devices
        self.tabs = QTabWidget()
        self.layout.addWidget(self.tabs)

        for device_info in self.devices:
            device_widget = DeviceControlWidget(device_info, self)
            self.tabs.addTab(device_widget, device_widget.name)
            # Start command sender thread for the device
            device_widget.start_command_sender_thread()

        # Focus Keywords Management
        focus_label = QLabel("Focus Keywords:")
        self.layout.addWidget(focus_label)

        self.focus_keywords_list_widget = QListWidget()
        self.load_focus_keywords()
        self.focus_keywords_list_widget.addItems(self.focus_keywords)
        self.layout.addWidget(self.focus_keywords_list_widget)

        self.add_keyword_button = QPushButton("Add Keyword")
        self.add_keyword_button.clicked.connect(self.add_focus_keyword)
        self.layout.addWidget(self.add_keyword_button)

        self.remove_keyword_button = QPushButton("Remove Selected Keyword")
        self.remove_keyword_button.clicked.connect(self.remove_focus_keyword)
        self.layout.addWidget(self.remove_keyword_button)

        # Re-scan Display Size Button
        self.rescan_button = QPushButton("Re-scan Display Size")
        self.rescan_button.clicked.connect(self.rescan_display_size)
        self.layout.addWidget(self.rescan_button)

        # Dual Monitor Checkbox
        self.dual_monitor_checkbox = QCheckBox("Enable Dual Monitor Support")
        self.dual_monitor_checkbox.setChecked(False)
        self.layout.addWidget(self.dual_monitor_checkbox)

        # Central widget
        container = QWidget()
        container.setLayout(self.layout)
        self.setCentralWidget(container)

        # Previous color for smooth transitions
        self.previous_color = np.array([0, 0, 0], dtype=np.float32)

        # Focus Monitoring
        self.paused = False
        self.paused_reasons = set()  # To track multiple pause reasons
        self.previous_target_brightness = 100

        # Start ALT key listener
        self.start_alt_key_listener()

        # Start focus monitoring thread
        self.focus_thread = threading.Thread(target=self.focus_monitor_loop, name="FocusMonitorThread")
        self.focus_thread.daemon = True
        self.focus_thread.start()

    def rescan_display_size(self):
        # Update screens
        self.screens = app.screens()
        QMessageBox.information(self, "Display Rescan", "Display size and screens have been updated.")

    def load_device_configs(self):
        # Load device configurations from devices.json
        if os.path.exists("devices.json"):
            with open("devices.json", "r") as f:
                data = json.load(f)
                self.devices_config = data.get("devices", [])
        else:
            # No config file found
            QMessageBox.warning(self, "Device Configurations", "No devices.json configuration file found.")
            self.devices_config = []

    def initialize_devices(self):
        self.devices = []
        for device_info in self.devices_config:
            try:
                device = MiotDevice(ip=device_info["ip"], token=device_info["token"])
                self.devices.append({
                    "device": device,
                    "name": device_info.get("name", "Unnamed Device"),
                    "region": device_info.get("region", {
                        "margin_left": 0,
                        "margin_right": 0,
                        "margin_top": 0,
                        "margin_bottom": 0
                    })
                })
            except DeviceException as e:
                print(f"Failed to initialize device {device_info.get('name', 'Unnamed Device')}: {e}")

    def save_device_configs(self):
        # Save device configurations to devices.json
        devices_to_save = []
        for i in range(len(self.devices)):
            device_info = self.devices[i]
            # Get updated region from the widget
            widget = self.tabs.widget(i)
            device_info["region"] = widget.region
            devices_to_save.append({
                "name": device_info["name"],
                "ip": device_info["device"].ip,
                "token": device_info["device"].token,
                "region": device_info["region"]
            })
        data = {'devices': devices_to_save}
        with open("devices.json", "w") as f:
            json.dump(data, f)

    def load_focus_keywords(self):
        if os.path.exists("focus_keywords.json"):
            with open("focus_keywords.json", "r") as f:
                data = json.load(f)
                self.focus_keywords = data.get("focus_keywords", [])
        else:
            self.focus_keywords = ["discord", "slack", "notepad", "reddit", "code", "chatgpt", "9gag", "irewind"]

    def add_focus_keyword(self):
        text, ok = QInputDialog.getText(self, 'Add Focus Keyword', 'Enter keyword:')
        if ok and text:
            self.focus_keywords.append(text.lower())
            self.focus_keywords_list_widget.addItem(text.lower())
            self.save_focus_keywords()

    def remove_focus_keyword(self):
        selected_items = self.focus_keywords_list_widget.selectedItems()
        if not selected_items:
            return
        for item in selected_items:
            keyword = item.text()
            self.focus_keywords.remove(keyword)
            self.focus_keywords_list_widget.takeItem(self.focus_keywords_list_widget.row(item))
        self.save_focus_keywords()

    def save_focus_keywords(self):
        data = {'focus_keywords': self.focus_keywords}
        with open("focus_keywords.json", "w") as f:
            json.dump(data, f)

    def focus_monitor_loop(self):
        check_interval = 1.0  # Check every 1 second
        focus_delay = 3.0     # Reduced delay for quicker response
        last_focus_change_time = None

        while True:
            try:
                active_window_title = win32gui.GetWindowText(win32gui.GetForegroundWindow()).lower()
                # Check if any keyword is in the window title
                is_paused_due_to_focus = any(keyword in active_window_title for keyword in self.focus_keywords)

                if is_paused_due_to_focus != ('focus_keyword' in self.paused_reasons):
                    if last_focus_change_time is None:
                        last_focus_change_time = time.time()
                    elif time.time() - last_focus_change_time >= focus_delay:
                        if is_paused_due_to_focus:
                            self.paused_reasons.add('focus_keyword')
                            print(f"Paused capturing because focused window is '{active_window_title}'")
                        else:
                            self.paused_reasons.discard('focus_keyword')
                            print(f"Resumed capturing because focused window is '{active_window_title}'")
                        self.update_paused_state()
                        last_focus_change_time = None
                else:
                    last_focus_change_time = None

                time.sleep(check_interval)
            except Exception as e:
                print(f"Error in focus_monitor_loop: {e}")
                time.sleep(check_interval)

    def update_paused_state(self):
        was_paused = self.paused
        self.paused = bool(self.paused_reasons)
        if self.paused != was_paused:
            for i in range(self.tabs.count()):
                widget = self.tabs.widget(i)
                if self.paused:
                    widget.initiate_fade_transition(to_brightness=0)
                else:
                    widget.initiate_fade_transition(to_brightness=100)

    def start_alt_key_listener(self):
        self.alt_key_pressed = False

        def on_press(key):
            try:
                if key == keyboard.Key.alt:
                    if not self.alt_key_pressed:
                        self.alt_key_pressed = True
                        print("ALT key pressed. Pausing all capturing and commands.")
            except AttributeError:
                pass

        def on_release(key):
            try:
                if key == keyboard.Key.alt:
                    if self.alt_key_pressed:
                        self.alt_key_pressed = False
                        print("ALT key released. Resuming capturing and commands.")
            except AttributeError:
                pass

        self.alt_listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        self.alt_listener.start()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            for i in range(self.tabs.count()):
                widget = self.tabs.widget(i)
                widget.stop_ambient_mode()
        else:
            super().keyPressEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DeviceControlApp()
    window.show()
    sys.exit(app.exec_())
