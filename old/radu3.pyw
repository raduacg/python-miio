import sys
import threading
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QPushButton,
    QWidget, QMessageBox, QLabel, QComboBox, QSlider, QColorDialog
)
from PyQt5.QtCore import Qt, QRect, QPoint
from PyQt5.QtGui import QColor
from PIL import ImageGrab
from miio.miot_device import MiotDevice
from miio.exceptions import DeviceException
from pynput import mouse


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

        # Action Buttons
        self.add_button("Brightness Up", lambda: self.send_action(3, 1))
        self.add_button("Brightness Down", lambda: self.send_action(3, 2))
        self.add_button("Toggle TV", lambda: self.send_action(3, 6))
        self.add_button("Toggle Rhythm", lambda: self.send_action(3, 12))

        # Central widget
        container = QWidget()
        container.setLayout(self.layout)
        self.setCentralWidget(container)

        # Region Selection
        self.selected_region = None
        self.ambient_thread = None
        self.stop_ambient = threading.Event()

    def add_button(self, label, command):
        button = QPushButton(label)
        button.clicked.connect(command)
        self.layout.addWidget(button)

    def send_command(self, siid, piid, value):
        try:
            response = self.device.set_property_by(siid, piid, value)
            print(f"Command response: {response}")
        except DeviceException as e:
            QMessageBox.critical(self, "Error", f"Failed to send command: {e}")
            print(f"Failed to send command: {e}")

    def send_action(self, siid, aiid, params=None):
        try:
            if params is None:
                params = []
            response = self.device.call_action_by(siid, aiid, params)
            print(f"Action response: {response}")
        except DeviceException as e:
            QMessageBox.critical(self, "Error", f"Failed to execute action: {e}")
            print(f"Failed to execute action: {e}")

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
            # Stop ambient mode
            self.stop_ambient.set()
            if self.ambient_thread:
                self.ambient_thread.join()

    def select_region_with_mouse_clicks(self):
        self.mouse_click_positions = []
        self.hide()  # Hide the main window while selecting

        def on_click(x, y, button, pressed):
            if pressed:
                self.mouse_click_positions.append((x, y))
                print(f"Mouse clicked at ({x}, {y})")
                if len(self.mouse_click_positions) >= 2:
                    # Stop listener after two clicks
                    return False

        print("Please click two points to define the rectangle")
        with mouse.Listener(on_click=on_click) as listener:
            listener.join()

        self.show()  # Show the main window again

        if len(self.mouse_click_positions) >= 2:
            # Use the first two clicks to define the rectangle
            x1, y1 = self.mouse_click_positions[0]
            x2, y2 = self.mouse_click_positions[1]
            self.selected_region = QRect(QPoint(x1, y1), QPoint(x2, y2))
            print(f"Selected region: {self.selected_region}")
        else:
            self.selected_region = None
            print("Region selection was not completed.")

    def ambient_loop(self):
        while not self.stop_ambient.is_set():
            try:
                # Capture the selected region
                x = self.selected_region.x()
                y = self.selected_region.y()
                w = self.selected_region.width()
                h = self.selected_region.height()
                img = ImageGrab.grab(bbox=(x, y, x + w, y + h))

                # Resize image to speed up processing
                img = img.resize((100, 100))

                # Convert image to numpy array
                img_np = np.array(img)

                # Reshape the image to be a list of pixels
                pixels = img_np.reshape(-1, 3)

                # Filter out white pixels (where R, G, B are all 255)
                pixels = pixels[~np.all(pixels == [255, 255, 255], axis=1)]

                if len(pixels) == 0:
                    # If all pixels are white, skip updating
                    print("All pixels are white; skipping this frame.")
                    continue

                # Find the dominant color
                unique, counts = np.unique(pixels, axis=0, return_counts=True)
                dominant_color = unique[counts.argmax()]

                # Convert to integer value
                r, g, b = dominant_color
                color_value = (int(r) << 16) + (int(g) << 8) + int(b)

                # Update the LED strip color
                self.send_command(2, 4, int(color_value))
            except Exception as e:
                print(f"Error in ambient_loop: {e}")

            # Wait for 20 milliseconds
            self.stop_ambient.wait(0.02)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DeviceControlApp()
    window.show()
    sys.exit(app.exec_())
