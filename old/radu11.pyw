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
    QWidget, QLabel, QComboBox, QSlider, QColorDialog, QMessageBox
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor
from PIL import ImageGrab
from miio.miot_device import MiotDevice
from miio.exceptions import DeviceException
from pynput import mouse, keyboard

from PyQt5 import QtGui, QtCore  # Import for dark mode

# Set CPU affinity to the last core
process = psutil.Process(os.getpid())
num_cores = os.cpu_count()
last_core_index = num_cores - 1  # Zero-based index
process.cpu_affinity([last_core_index])

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

        # Saturation factor
        self.saturation_factor = 1.3  # Increase saturation by 30%

        # Focus Monitoring
        self.paused = False
        self.paused_reasons = set()  # To track multiple pause reasons
        self.focus_keywords = ["discord", "slack", "notepad", "reddit", "code", "chatgpt"]
        self.previous_target_brightness = self.target_brightness

        # Start ALT key listener
        self.start_alt_key_listener()

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
        sample_interval = 0.064  # 64 milliseconds
        color_transition_interval = 0.128  # 128 milliseconds
        sample_buffer = []
        last_color_update_time = time.time()
        while not self.stop_ambient.is_set():
            start_time = time.time()
            try:
                if self.paused:
                    # Handle paused state
                    self.handle_fade_transition()
                    # Use the previous color to prevent abrupt changes
                    blended_color = self.previous_color.astype(np.uint8)
                    # Add to command queue
                    self.command_queue.put((blended_color, self.current_brightness))
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
                if overall_brightness < 30:
                    new_target_brightness = 0
                else:
                    new_target_brightness = 100

                # Handle fade transition when necessary
                if new_target_brightness != self.target_brightness:
                    self.initiate_fade_transition(new_target_brightness)

                self.handle_fade_transition()

                # Filter out near-white pixels
                threshold = 240
                pixels = pixels[~np.all(pixels >= threshold, axis=1)]

                # Additional filtering for dark scenes
                if overall_brightness < 60:
                    pixels = pixels[~np.any(pixels >= 200, axis=1)]

                if len(pixels) == 0:
                    dominant_color = np.array([0, 0, 0], dtype=np.uint8)
                else:
                    # Find the dominant color
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

                # Smooth color transition
                blended_color = self.blend_colors(self.previous_color, dominant_color, alpha=0.16)
                self.previous_color = blended_color

                # Ensure color values are valid
                blended_color = np.clip(blended_color, 0, 255).astype(np.uint8)

                # Add to sample buffer
                sample_buffer.append((blended_color, self.current_brightness))

                # Send color updates every 128ms
                if time.time() - last_color_update_time >= color_transition_interval:
                    # Average the samples
                    avg_color = np.mean([s[0] for s in sample_buffer], axis=0).astype(np.uint8)
                    avg_brightness = self.current_brightness

                    # Reset sample buffer and update last_color_update_time
                    sample_buffer = []
                    last_color_update_time = time.time()

                    # Add to command queue
                    self.command_queue.put((avg_color, avg_brightness))

            except Exception as e:
                print(f"Error in ambient_loop: {e}")
                pass  # Suppress exceptions to improve performance

            # Wait for the next sample
            elapsed = time.time() - start_time
            sleep_time = max(0, sample_interval - elapsed)
            self.stop_ambient.wait(sleep_time)

    def initiate_fade_transition(self, to_brightness):
        if to_brightness < self.current_brightness:
            # Transition to black in 2 stages: 100% -> 50% -> 0%
            self.brightness_steps = [100, 50, 0]
            self.fade_transition_duration = 1.0  # Total duration of 1 second
        else:
            # Transition from black to color in 4 stages: 0% -> 10% -> 40% -> 80% -> 100%
            self.brightness_steps = [0, 10, 40, 80, 100]
            self.fade_transition_duration = 2.0  # Total duration for fade-in

        self.fade_transition_start_time = time.time()
        self.initial_brightness = self.current_brightness
        self.target_brightness = to_brightness

        self.brightness_step_index = self.find_nearest_brightness_step_index(self.initial_brightness)
        self.target_brightness_step_index = self.find_nearest_brightness_step_index(self.target_brightness)
        self.brightness_direction = 1 if self.target_brightness_step_index > self.brightness_step_index else -1
        steps_count = abs(self.target_brightness_step_index - self.brightness_step_index)
        self.fade_step_duration = self.fade_transition_duration / max(steps_count, 1)
        self.next_brightness_step_time = self.fade_transition_start_time + self.fade_step_duration

    def find_nearest_brightness_step_index(self, brightness):
        brightness = int(round(brightness))
        brightness_steps = self.brightness_steps
        differences = [abs(b - brightness) for b in brightness_steps]
        nearest_index = differences.index(min(differences))
        return nearest_index

    def handle_fade_transition(self):
        if hasattr(self, 'fade_transition_start_time') and self.fade_transition_start_time is not None:
            current_time = time.time()
            if current_time >= self.next_brightness_step_time:
                # Move to next brightness step
                self.brightness_step_index += self.brightness_direction
                # Ensure index stays within bounds
                self.brightness_step_index = max(0, min(self.brightness_step_index, len(self.brightness_steps) - 1))
                if self.brightness_step_index == self.target_brightness_step_index:
                    # Transition complete
                    self.current_brightness = self.target_brightness
                    self.fade_transition_start_time = None
                else:
                    self.current_brightness = self.brightness_steps[self.brightness_step_index]
                    self.next_brightness_step_time += self.fade_step_duration

    def command_sender_loop(self):
        send_interval = 0.128  # 128 milliseconds for color updates
        fade_send_interval = 1.2  # 1200 milliseconds for fade transitions
        last_send_time = time.time()
        last_avg_color = None
        last_avg_brightness = None

        while True:
            start_time = time.time()
            try:
                # Get the latest command from the queue
                while not self.command_queue.empty():
                    avg_color, brightness = self.command_queue.get_nowait()

                # Determine send interval based on whether a fade transition is in progress
                if hasattr(self, 'fade_transition_start_time') and self.fade_transition_start_time is not None:
                    current_send_interval = fade_send_interval
                else:
                    current_send_interval = send_interval

                # Send commands at the appropriate interval
                if time.time() - last_send_time >= current_send_interval:
                    # Check for significant changes
                    if last_avg_color is not None:
                        color_diff = np.linalg.norm(avg_color - last_avg_color)
                    else:
                        color_diff = 999  # Ensure first command is sent

                    brightness_diff = abs(brightness - (last_avg_brightness or 0))

                    # Send command if there is a significant change or during fade transitions
                    if (hasattr(self, 'fade_transition_start_time') and self.fade_transition_start_time is not None) \
                            or color_diff > 5 or brightness_diff > 5:
                        # Convert avg_color to integer value
                        r, g, b = avg_color
                        color_value = (int(r) << 16) + (int(g) << 8) + int(b)

                        # Send the color command
                        self.send_command(2, 4, int(color_value))

                        # Send the brightness command
                        self.send_command(2, 3, int(brightness))

                        last_avg_color = avg_color
                        last_avg_brightness = brightness

                    last_send_time = time.time()

            except Exception as e:
                print(f"Error in command_sender_loop: {e}")

            # Sleep until the next send interval
            elapsed = time.time() - start_time
            sleep_time = max(0, send_interval - elapsed)
            time.sleep(sleep_time)

    def focus_monitor_loop(self):
        check_interval = 0.5  # Check every 0.5 seconds
        focus_delay = 0.5     # Reduced delay for quicker response
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
            if self.paused:
                self.initiate_fade_transition(to_brightness=0)
            else:
                self.initiate_fade_transition(to_brightness=100)

    def start_alt_key_listener(self):
        def on_press(key):
            try:
                if key == keyboard.Key.alt:
                    self.paused_reasons.add('alt_key')
                    self.update_paused_state()
                    print("Paused capturing because ALT key is pressed")
            except AttributeError:
                pass

        def on_release(key):
            try:
                if key == keyboard.Key.alt:
                    self.paused_reasons.discard('alt_key')
                    self.update_paused_state()
                    print("Resumed capturing because ALT key is released")
            except AttributeError:
                pass

        self.alt_listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        self.alt_listener.start()

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
        if self.is_color_in_range(h, lower_green, upper_green) or \
           self.is_color_in_range(h, lower_orange, upper_orange) or \
           self.is_color_in_range(h, lower_neon, upper_neon):
            # Enhance the color by increasing brightness and saturation
            s = np.clip(int(s * 1.5), 0, 255)
            v = np.clip(int(v * 1.2), 0, 255)
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
