import sys
import threading
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QPushButton,
    QWidget, QMessageBox, QLabel, QComboBox, QSlider, QColorDialog, QDesktopWidget
)
from PyQt5.QtCore import Qt, QRect
from PyQt5.QtGui import QPainter, QPen
from PIL import ImageGrab
from miio import MiotDevice
from miio.exceptions import DeviceException


class ScreenRegionSelector(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Select Region")
        self.screen_geometry = QApplication.primaryScreen().geometry()
        self.setGeometry(self.screen_geometry)
        self.start_point = None
        self.end_point = None
        self.setWindowOpacity(0.3)
        self.setWindowFlag(Qt.WindowStaysOnTopHint)
        self.setWindowFlag(Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, False)
        self.setAttribute(Qt.WA_NoSystemBackground, True)
        self.setAttribute(Qt.WA_TranslucentBackground, True)

    def mousePressEvent(self, event):
        self.start_point = event.pos()

    def mouseMoveEvent(self, event):
        self.end_point = event.pos()
        self.update()

    def mouseReleaseEvent(self, event):
        self.end_point = event.pos()
        self.close()

    def paintEvent(self, event):
        if self.start_point and self.end_point:
            painter = QPainter(self)
            painter.setPen(QPen(Qt.red, 2, Qt.SolidLine))
            rect = QRect(self.start_point, self.end_point)
            painter.drawRect(rect.normalized())

    def get_selection(self):
        if self.start_point and self.end_point:
            return QRect(self.start_point, self.end_point).normalized()
        else:
            return None

    def exec_(self):
        loop = QEventLoop()
        self.finished.connect(loop.quit)
        loop.exec_()


class DeviceControlApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Smart Device Controller")
        self.setGeometry(100, 100, 400, 500)

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
        # Add more action buttons as needed

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
            # QMessageBox.information(self, "Success", f"Command sent successfully! Response: {response}")
        except DeviceException as e:
            QMessageBox.critical(self, "Error", f"Failed to send command: {e}")

    def send_action(self, siid, aiid, params=None):
        try:
            if params is None:
                params = []
            response = self.device.call_action(siid, aiid, params)
            # QMessageBox.information(self, "Success", f"Action executed successfully! Response: {response}")
        except DeviceException as e:
            QMessageBox.critical(self, "Error", f"Failed to execute action: {e}")

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
            self.select_region()
            if self.selected_region:
                self.stop_ambient.clear()
                self.ambient_thread = threading.Thread(target=self.ambient_loop)
                self.ambient_thread.start()
        else:
            # Stop ambient mode
            self.stop_ambient.set()
            if self.ambient_thread:
                self.ambient_thread.join()

    def select_region(self):
        self.selector = ScreenRegionSelector()
        self.selector.show()
        QApplication.processEvents()
        while self.selector.isVisible():
            QApplication.processEvents()
        self.selected_region = self.selector.get_selection()

    def ambient_loop(self):
        while not self.stop_ambient.is_set():
            # Capture the selected region
            x = self.selected_region.x()
            y = self.selected_region.y()
            w = self.selected_region.width()
            h = self.selected_region.height()
            img = ImageGrab.grab(bbox=(x, y, x + w, y + h))

            # Convert image to numpy array
            img_np = np.array(img)

            # Reshape the image to be a list of pixels
            pixels = img_np.reshape(-1, 3)

            # Find the dominant color
            unique, counts = np.unique(pixels, axis=0, return_counts=True)
            dominant_color = unique[counts.argmax()]

            # Convert to integer value
            r, g, b = dominant_color
            color_value = (r << 16) + (g << 8) + b

            # Update the LED strip color
            self.send_command(2, 4, int(color_value))

            # Wait for 80 milliseconds
            self.stop_ambient.wait(0.08)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DeviceControlApp()
    window.show()
    sys.exit(app.exec_())
