import sys
import threading
import numpy as np
import warnings
import time
import queue
import json
import cv2
import win32gui  # For getting active window title

from sklearn.cluster import KMeans
from sklearn.exceptions import ConvergenceWarning
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QPushButton,
    QWidget, QLabel, QComboBox, QSlider, QColorDialog, QMessageBox
)
from PyQt5.QtCore import Qt, QRect, QPoint
from PyQt5.QtGui import QColor
from PIL import ImageGrab
from miio.miot_device import MiotDevice
from miio.exceptions import DeviceException
from pynput import mouse
import os

from PyQt5 import QtGui, QtCore  # Import for dark mode

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
        self.setGeometry(100, 100, 400, 600)

        # Device details
        self.device_ip = "192.168.1.212"
        self.device_token = "4ee353cd722f4e791dfb9481526d411a"
        self.device = MiotDevice(ip=self.device_ip, token=self.device_token)

        # Main layout
        self.layout = QVBoxLayout()

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
        self.brightness_slider.setValue(50)  # Default value
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

        # Select Region Button
        self.select_region_button = QPushButton("Select Region")
        self.select_region_button.clicked.connect(self.select_region_with_mouse_clicks)
        self.layout.addWidget(self.select_region_button)

        # Coordinates Display
        self.coordinates_label = QLabel("Selected Region Coordinates:")
        self.layout.addWidget(self.coordinates_label)

        # Action Buttons
        self.add_button("Brightness Up", lambda: self.send_action(3, 1))
        self.add_button("Brightness Down", lambda: self.send_action(3, 2))
        self.add_button("Toggle TV", lambda: self.send_action(3, 6))
        self.add_button("Toggle Rhythm", lambda: self.send_action(3, 12))

        # Set Red/Blue Button
        self.add_button("Set Red/Blue", self.set_red_blue)

        # Central widget
        container = QWidget()
        container.setLayout(self.layout)
        self.setCentralWidget(container)

        # Region Selection
        self.selected_region = None
        self.load_settings()
        self.ambient_thread = None
        self.stop_ambient = threading.Event()
        self.command_queue = queue.Queue()

        # Start command sender thread
        self.command_sender_thread = threading.Thread(target=self.command_sender_loop, name="CommandSenderThread")
        self.command_sender_thread.daemon = True
        self.command_sender_thread.start()

        # Previous color for smooth transitions
        self.previous_color = np.array([0, 0, 0], dtype=np.float32)

        # Initialize brightness variables
        self.current_brightness = 100
        self.target_brightness = 100
        self.last_brightness_change_time = time.time()

        # Brightness steps
        self.brightness_steps = [0, 20, 40, 60, 80, 100]
        self.brightness_transition_duration = 2.0  # 2 seconds for full transition
        self.brightness_step_duration = self.brightness_transition_duration / (len(self.brightness_steps) - 1)

        # Saturation factor
        self.saturation_factor = 1.3  # Increase saturation by 30%

        # Focus Monitoring
        self.paused = False
        self.focus_keywords = ["discord", "slack", "notepad", "reddit", "code"]
        self.previous_target_brightness = self.target_brightness

        # Start focus monitoring thread
        self.focus_thread = threading.Thread(target=self.focus_monitor_loop, name="FocusMonitorThread")
        self.focus_thread.daemon = True
        self.focus_thread.start()

    def add_button(self, label, command):
        button = QPushButton(label)
        button.clicked.connect(command)
        self.layout.addWidget(button)

    def send_command(self, siid, piid, value):
        try:
            self.device.set_property_by(siid, piid, value)
        except DeviceException:
            pass  # Suppress errors to improve performance

    def send_action(self, siid, aiid, params=None):
        try:
            if params is None:
                params = []
            self.device.call_action_by(siid, aiid, params)
        except DeviceException:
            pass  # Suppress errors to improve performance

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
            if self.selected_region:
                # Start ambient mode
                self.stop_ambient.clear()
                # Start the ambient loop in a new thread
                self.ambient_thread = threading.Thread(target=self.ambient_loop, name="AmbientThread")
                self.ambient_thread.start()
            else:
                # No region selected, prompt user
                self.select_region_with_mouse_clicks()
                if self.selected_region:
                    self.stop_ambient.clear()
                    self.ambient_thread = threading.Thread(target=self.ambient_loop, name="AmbientThread")
                    self.ambient_thread.start()
                else:
                    self.ambient_button.setChecked(False)
        else:
            # Stop ambient mode
            self.stop_ambient_mode()

    def stop_ambient_mode(self):
        self.stop_ambient.set()
        if self.ambient_thread:
            self.ambient_thread.join()
        self.ambient_button.setChecked(False)

    def select_region_with_mouse_clicks(self):
        self.mouse_click_positions = []
        self.hide()  # Hide the main window while selecting

        def on_click(x, y, button, pressed):
            if pressed:
                self.mouse_click_positions.append((x, y))
                if len(self.mouse_click_positions) >= 4:
                    # Stop listener after four clicks
                    return False

        QMessageBox.information(self, "Select Region", "Please click four points to define the quadrilateral region.")
        with mouse.Listener(on_click=on_click) as listener:
            listener.join()

        self.show()  # Show the main window again

        if len(self.mouse_click_positions) >= 4:
            # Use the four clicks to define the quadrilateral
            self.selected_region = self.mouse_click_positions[:4]
            self.coordinates_label.setText(f"Selected Region Coordinates:\n"
                                           f"Point 1: {self.selected_region[0]}\n"
                                           f"Point 2: {self.selected_region[1]}\n"
                                           f"Point 3: {self.selected_region[2]}\n"
                                           f"Point 4: {self.selected_region[3]}")
            self.save_settings()
        else:
            self.selected_region = None

    def save_settings(self):
        # Save the selected region to a JSON file
        data = {'selected_region': [list(point) for point in self.selected_region]}
        with open("settings.json", "w") as f:
            json.dump(data, f)

    def load_settings(self):
        # Load the selected region from a file if it exists
        if os.path.exists("settings.json"):
            with open("settings.json", "r") as f:
                data = json.load(f)
                self.selected_region = [tuple(point) for point in data.get('selected_region', [])]
                if self.selected_region:
                    self.coordinates_label.setText(f"Selected Region Coordinates:\n"
                                                   f"Point 1: {self.selected_region[0]}\n"
                                                   f"Point 2: {self.selected_region[1]}\n"
                                                   f"Point 3: {self.selected_region[2]}\n"
                                                   f"Point 4: {self.selected_region[3]}")

    def ambient_loop(self):
        sample_interval = 0.016  # 16 milliseconds (60Hz)
        while not self.stop_ambient.is_set():
            start_time = time.time()
            try:
                if self.paused:
                    # If paused, use black color and handle brightness transition
                    dominant_color = np.array([0, 0, 0], dtype=np.uint8)
                    if self.target_brightness != 0:
                        self.initiate_brightness_transition(0)
                    self.handle_brightness_transition()
                    # Add to command queue
                    self.command_queue.put((dominant_color, self.current_brightness))
                    time.sleep(sample_interval)
                    continue

                # Capture the screen
                img = ImageGrab.grab()
                img_np = np.array(img)

                # Get the four selected points
                pts_src = np.array(self.selected_region, dtype=np.float32)

                # Order the points
                pts = self.order_points(pts_src)

                # Compute the width and height
                maxWidth, maxHeight = self.compute_max_width_height(pts)

                # Define destination points
                dst = np.array([
                    [0, 0],
                    [maxWidth - 1, 0],
                    [maxWidth - 1, maxHeight - 1],
                    [0, maxHeight - 1]], dtype="float32")

                # Compute the perspective transform
                M = cv2.getPerspectiveTransform(pts, dst)

                # Warp the image
                warped = cv2.warpPerspective(img_np, M, (maxWidth, maxHeight))

                # Resize image to speed up processing
                img_small = cv2.resize(warped, (200, 200))

                # Convert image to numpy array
                pixels = img_small.reshape(-1, 3)

                # Calculate overall brightness
                overall_brightness = np.mean(pixels)

                # Adjust target brightness based on overall brightness
                if overall_brightness < 24:
                    new_target_brightness = 0
                    dominant_color = np.array([0, 0, 0], dtype=np.uint8)
                else:
                    new_target_brightness = 100
                    # Filter out near-white pixels
                    threshold = 240
                    pixels = pixels[~np.all(pixels >= threshold, axis=1)]

                    # Additional filtering: Treat near-white pixels as black in dark scenes
                    if overall_brightness < 60:
                        pixels = pixels[~np.any(pixels >= 200, axis=1)]

                    if len(pixels) == 0:
                        # If all pixels are filtered out, use black
                        dominant_color = np.array([0, 0, 0], dtype=np.uint8)
                        new_target_brightness = 0
                    else:
                        # Use K-Means clustering to find the dominant color
                        unique_colors = np.unique(pixels, axis=0)
                        num_unique_colors = len(unique_colors)
                        num_clusters = min(num_unique_colors, 6)
                        if num_clusters == 0:
                            continue
                        kmeans = KMeans(n_clusters=num_clusters, n_init=1)
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", category=ConvergenceWarning)
                            kmeans.fit(pixels)
                        colors = kmeans.cluster_centers_
                        labels = kmeans.labels_
                        counts = np.bincount(labels)
                        dominant_color = colors[counts.argmax()].astype(np.uint8)

                        # Increase saturation
                        dominant_color = self.increase_saturation(dominant_color)

                        # Enhance specific colors
                        dominant_color = self.enhance_specific_colors(dominant_color)

                # Smooth transition between colors
                blended_color = self.blend_colors(self.previous_color, dominant_color, alpha=0.32)
                self.previous_color = blended_color

                # Ensure color values are valid
                blended_color = np.clip(blended_color, 0, 255).astype(np.uint8)

                # Handle brightness transition
                if new_target_brightness != self.target_brightness:
                    self.initiate_brightness_transition(new_target_brightness)

                self.handle_brightness_transition()

                # Add to command queue
                self.command_queue.put((blended_color, self.current_brightness))

            except Exception as e:
                print(f"Error in ambient_loop: {e}")
                pass  # Suppress exceptions to improve performance

            # Wait for the next sample
            elapsed = time.time() - start_time
            sleep_time = max(0, sample_interval - elapsed)
            self.stop_ambient.wait(sleep_time)

    def initiate_brightness_transition(self, to_brightness):
        self.brightness_transition_start_time = time.time()
        self.initial_brightness = self.current_brightness
        self.target_brightness = to_brightness
        self.brightness_step_index = self.brightness_steps.index(int(round(self.initial_brightness)))
        self.target_brightness_step_index = self.brightness_steps.index(int(round(to_brightness)))
        self.brightness_direction = 1 if self.target_brightness_step_index > self.brightness_step_index else -1
        self.next_brightness_step_time = self.brightness_transition_start_time + self.brightness_step_duration

    def handle_brightness_transition(self):
        if self.brightness_transition_start_time is not None:
            current_time = time.time()
            if current_time >= self.next_brightness_step_time:
                # Move to next brightness step
                self.brightness_step_index += self.brightness_direction
                if (self.brightness_direction > 0 and self.brightness_step_index >= self.target_brightness_step_index) or \
                   (self.brightness_direction < 0 and self.brightness_step_index <= self.target_brightness_step_index):
                    # Transition complete
                    self.current_brightness = self.target_brightness
                    self.brightness_transition_start_time = None
                else:
                    self.current_brightness = self.brightness_steps[self.brightness_step_index]
                    self.next_brightness_step_time += self.brightness_step_duration

    def command_sender_loop(self):
        send_interval = 0.128  # 128 milliseconds
        buffer_duration = 0.032  # 32 milliseconds
        last_send_time = time.time()
        command_buffer = []

        while True:
            start_time = time.time()
            try:
                # Collect commands from the queue over the buffer duration
                buffer_end_time = start_time + buffer_duration
                while time.time() < buffer_end_time:
                    try:
                        command = self.command_queue.get(timeout=buffer_duration)
                        command_buffer.append(command)
                    except queue.Empty:
                        break

                # Send commands at the specified send interval
                if time.time() - last_send_time >= send_interval and command_buffer:
                    # Use the latest command in the buffer
                    avg_color, brightness = command_buffer[-1]
                    command_buffer.clear()

                    # Convert avg_color to integer value
                    r, g, b = avg_color
                    color_value = (int(r) << 16) + (int(g) << 8) + int(b)

                    # Send the color command
                    self.send_command(2, 4, int(color_value))

                    # Send the brightness command
                    self.send_command(2, 3, int(brightness))

                    last_send_time = time.time()

            except Exception as e:
                print(f"Error in command_sender_loop: {e}")

            # Sleep until the next send interval
            elapsed = time.time() - start_time
            sleep_time = max(0, send_interval - elapsed)
            time.sleep(sleep_time)

    def focus_monitor_loop(self):
        check_interval = 0.5  # Check every 0.5 seconds
        while True:
            try:
                active_window_title = win32gui.GetWindowText(win32gui.GetForegroundWindow()).lower()
                # Check if any keyword is in the window title
                if any(keyword in active_window_title for keyword in self.focus_keywords):
                    if not self.paused:
                        self.paused = True
                        self.previous_target_brightness = self.target_brightness
                        self.initiate_brightness_transition(to_brightness=0)
                        print(f"Paused capturing because focused window is '{active_window_title}'")
                else:
                    if self.paused:
                        self.paused = False
                        self.initiate_brightness_transition(to_brightness=100)
                        print(f"Resumed capturing because focused window is '{active_window_title}'")
                time.sleep(check_interval)
            except Exception as e:
                print(f"Error in focus_monitor_loop: {e}")
                time.sleep(check_interval)

    def order_points(self, pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]  # Top-left
        rect[2] = pts[np.argmax(s)]  # Bottom-right
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]  # Top-right
        rect[3] = pts[np.argmax(diff)]  # Bottom-left
        return rect

    def compute_max_width_height(self, pts):
        # Compute the width of the new image
        widthA = np.sqrt(((pts[2][0] - pts[3][0]) ** 2) + ((pts[2][1] - pts[3][1]) ** 2))
        widthB = np.sqrt(((pts[1][0] - pts[0][0]) ** 2) + ((pts[1][1] - pts[0][1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))

        # Compute the height of the new image
        heightA = np.sqrt(((pts[1][0] - pts[2][0]) ** 2) + ((pts[1][1] - pts[2][1]) ** 2))
        heightB = np.sqrt(((pts[0][0] - pts[3][0]) ** 2) + ((pts[0][1] - pts[3][1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))

        # Ensure maxWidth and maxHeight are at least 1
        maxWidth = max(maxWidth, 1)
        maxHeight = max(maxHeight, 1)

        return maxWidth, maxHeight

    def increase_saturation(self, color_rgb):
        color_array = np.uint8([[color_rgb]])  # Convert to 1x1 image
        hsv = cv2.cvtColor(color_array, cv2.COLOR_RGB2HSV)
        h, s, v = hsv[0][0]
        s = np.clip(int(s * self.saturation_factor), 0, 255)  # Increase saturation
        hsv[0][0][1] = s
        color_array = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        return color_array[0][0]

    def enhance_specific_colors(self, color_rgb):
        color_array = np.uint8([[color_rgb]])  # Convert to 1x1 image
        hsv = cv2.cvtColor(color_array, cv2.COLOR_RGB2HSV)
        h, s, v = hsv[0][0]

        # Define HSV ranges for specific colors

        # Trees (Green)
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])

        # Campfire (Orange/Red)
        lower_orange = np.array([10, 100, 100])
        upper_orange = np.array([25, 255, 255])

        # Neon lights: High saturation and brightness
        lower_neon = np.array([0, 150, 150])
        upper_neon = np.array([179, 255, 255])  # Full hue range

        # Check if the color falls within any of the ranges
        if self.is_color_in_range(hsv[0][0], lower_green, upper_green) or \
           self.is_color_in_range(hsv[0][0], lower_orange, upper_orange) or \
           self.is_color_in_range(hsv[0][0], lower_neon, upper_neon):
            # Enhance the color by increasing brightness and saturation
            s = np.clip(int(s * 1.5), 0, 255)
            v = np.clip(int(v * 1.2), 0, 255)
            hsv[0][0][1] = s
            hsv[0][0][2] = v
            color_array = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
            return color_array[0][0]
        else:
            return color_rgb

    def is_color_in_range(self, hsv_pixel, lower_hsv, upper_hsv):
        # Handle the case when hue wraps around
        hue = hsv_pixel[0]
        if lower_hsv[0] <= upper_hsv[0]:
            # Normal range
            in_range = lower_hsv[0] <= hue <= upper_hsv[0]
        else:
            # Hue wraps around
            in_range = hue >= lower_hsv[0] or hue <= upper_hsv[0]

        return in_range and np.all(hsv_pixel[1:] >= lower_hsv[1:]) and np.all(hsv_pixel[1:] <= upper_hsv[1:])

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

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.stop_ambient_mode()
        else:
            super().keyPressEvent(event)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DeviceControlApp()
    window.show()
    sys.exit(app.exec_())
