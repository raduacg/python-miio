import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QPushButton, QWidget, QMessageBox
from miio import Device

class DeviceControlApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Smart Device Controller")
        self.setGeometry(100, 100, 400, 300)

        # Device details
        self.device_ip = "192.168.1.212"
        self.device_token = "4ee353cd722f4e791dfb9481526d411a"
        self.device = Device(self.device_ip, self.device_token)

        # Main layout
        self.layout = QVBoxLayout()

        # Add buttons for commands
        self.add_button("Set Blue", lambda: self.send_command(2, 4, 255))  # Blue
        self.add_button("Set Red", lambda: self.send_command(2, 4, 16711680))  # Red
        self.add_button("Set Green", lambda: self.send_command(2, 4, 65280))  # Green
        self.add_button("Turn On", lambda: self.send_command(2, 1, True))  # Turn On
        self.add_button("Turn Off", lambda: self.send_command(2, 1, False))  # Turn Off

        # Central widget
        container = QWidget()
        container.setLayout(self.layout)
        self.setCentralWidget(container)

    def add_button(self, label, command):
        button = QPushButton(label)
        button.clicked.connect(command)
        self.layout.addWidget(button)

    def send_command(self, siid, piid, value):
        try:
            response = self.device.raw_command("set_properties", [{"siid": siid, "piid": piid, "value": value}])
            QMessageBox.information(self, "Success", f"Command sent successfully! Response: {response}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to send command: {e}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DeviceControlApp()
    window.show()
    sys.exit(app.exec_())
