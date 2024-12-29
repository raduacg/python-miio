import sys
import threading
import numpy as np
import warnings
from sklearn.cluster import KMeans
from sklearn.exceptions import ConvergenceWarning
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QPushButton,
    QWidget, QLabel, QComboBox, QSlider, QColorDialog
)
from PyQt5.QtCore import Qt, QRect, QPoint
from PyQt5.QtGui import QColor
from PIL import ImageGrab
from miio.miot_device import MiotDevice
from miio.exceptions import DeviceException
from pynput import mouse
import pickle
import os
import json
import cv2
import time
import queue

class DeviceControlApp(QMainWindow):
    def __init__(self):
        super().__init__()
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

        # Escape Button Info
        escape_label = QLabel("Press 'Escape' to stop Ambient Sync")
        self.layout.addWidget(escape_label)

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
        self.command_sender_thread.start()

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
            # Start ambient mode
            self.select_region_with_mouse_clicks()
            if self.selected_region:
                self.stop_ambient.clear()
                # Start the ambient loop in a new thread
                self.ambient_thread = threading.Thread(target=self.ambient_loop, name="AmbientThread")
                self.ambient_thread.start()
            else:
                # If no region was selected, uncheck the button
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

        print("Please click four points to define the quadrilateral region")
        with mouse.Listener(on_click=on_click) as listener:
            listener.join()

        self.show()  # Show the main window again

        if len(self.mouse_click_positions) >= 4:
            # Use the four clicks to define the quadrilateral
            self.selected_region = self.mouse_click_positions[:4]
            print("Selected coordinates:")
            for idx, (x, y) in enumerate(self.selected_region):
                print(f"Point {idx+1}: ({x}, {y})")
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

    def ambient_loop(self):
        colors_buffer = []
        previous_brightness = None
        while not self.stop_ambient.is_set():
            start_time = time.time()
            try:
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
                img_small = cv2.resize(warped, (100, 100))

                # Convert image to numpy array
                pixels = img_small.reshape(-1, 3)

                # Filter out near-white pixels
                threshold = 240
                pixels = pixels[~np.all(pixels >= threshold, axis=1)]

                if len(pixels) == 0:
                    # If all pixels are near-white, skip updating
                    continue

                # Color biasing
                color_biases = {
                    'forest_green': {'lower': np.array([34, 139, 34]), 'upper': np.array([50, 205, 50])},
                    'red': {'lower': np.array([200, 0, 0]), 'upper': np.array([255, 60, 60])},
                    'orange': {'lower': np.array([255, 165, 0]), 'upper': np.array([255, 200, 100])},
                    'pink': {'lower': np.array([255, 105, 180]), 'upper': np.array([255, 182, 193])},
                    'neon_violet': {'lower': np.array([138, 43, 226]), 'upper': np.array([148, 0, 211])},
                }
                total_pixels = len(pixels)
                color_counts = {}
                for color_name, bounds in color_biases.items():
                    lower = bounds['lower']
                    upper = bounds['upper']
                    mask = np.all((pixels >= lower) & (pixels <= upper), axis=1)
                    count = np.sum(mask)
                    percentage = count / total_pixels
                    color_counts[color_name] = percentage

                max_percentage = 0
                dominant_bias_color = None
                for color_name, percentage in color_counts.items():
                    if percentage >= 0.2 and percentage > max_percentage:
                        max_percentage = percentage
                        dominant_bias_color = color_name

                if dominant_bias_color:
                    dominant_colors = {
                        'forest_green': [34, 139, 34],
                        'red': [255, 0, 0],
                        'orange': [255, 165, 0],
                        'pink': [255, 105, 180],
                        'neon_violet': [148, 0, 211],
                    }
                    dominant_color = dominant_colors[dominant_bias_color]
                else:
                    # Use K-Means clustering to find the dominant color
                    unique_colors = np.unique(pixels, axis=0)
                    num_unique_colors = len(unique_colors)
                    num_clusters = min(num_unique_colors, 4)
                    if num_clusters == 0:
                        continue
                    kmeans = KMeans(n_clusters=num_clusters, n_init=1)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=ConvergenceWarning)
                        kmeans.fit(pixels)
                    colors = kmeans.cluster_centers_
                    labels = kmeans.labels_
                    counts = np.bincount(labels)
                    dominant_color = colors[counts.argmax()].astype(int)

                # Compute brightness
                brightness = np.mean(dominant_color)

                # If brightness drops significantly, adjust over time
                if previous_brightness is not None and brightness < previous_brightness * 0.5:
                    steps = 10
                    brightness_values = np.linspace(previous_brightness, brightness, steps)
                    for b in brightness_values:
                        adjusted_color = dominant_color * (b / brightness)
                        colors_buffer.append(adjusted_color)
                        time.sleep(0.1)
                else:
                    colors_buffer.append(dominant_color)

                previous_brightness = brightness

                # Update every 32 milliseconds
                if len(colors_buffer) >= 4:
                    avg_color = np.mean(colors_buffer, axis=0).astype(int)
                    colors_buffer = []
                    self.command_queue.put((avg_color, start_time))
            except Exception:
                pass  # Suppress exceptions to improve performance

            # Wait for the remaining time
            elapsed = time.time() - start_time
            sleep_time = max(0, 0.008 - elapsed)
            self.stop_ambient.wait(sleep_time)

    def command_sender_loop(self):
        while not self.stop_ambient.is_set():
            try:
                # Get the command from the queue
                avg_color, timestamp = self.command_queue.get(timeout=0.1)

                # Drop commands that are more than 2s old
                if time.time() - timestamp > 2:
                    continue

                # Convert avg_color to integer value
                r, g, b = avg_color
                color_value = (int(r) << 16) + (int(g) << 8) + int(b)

                # Send the command
                threading.Thread(target=self.send_command, args=(2, 4, int(color_value)), daemon=True).start()
            except queue.Empty:
                continue

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

        return maxWidth, maxHeight

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
