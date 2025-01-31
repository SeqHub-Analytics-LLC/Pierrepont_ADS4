"""### **Challenge 5: Smart Device Management System (Using Multiple Inheritance)**

#### **Objective**:
Design a system to manage **Smart Devices** that are either **Power-Operated** or **WiFi-Enabled**, or both. Use **multiple inheritance** to combine features from different base classes. Implement a `SmartDevice` class that inherits attributes and behaviors from the relevant parent classes.


#### **Requirements**:
1. Create the following classes:
   - `PowerDevice`: Represents devices that require power to operate.
     - Attribute: `power_status` (e.g., "On" or "Off").
     - Method: `toggle_power()` to turn the device on or off.
   - `WiFiDevice`: Represents devices that can connect to WiFi.
     - Attribute: `connected` (Boolean, whether it’s connected to WiFi).
     - Method: `connect_wifi()` and `disconnect_wifi()`.
   - `SmartDevice`: A device that can be both power-operated and WiFi-enabled.
     - Inherits from both `PowerDevice` and `WiFiDevice`.

2. The `SmartDevice` class should:
   - Combine attributes from both parent classes.
   - Add its own attribute: `device_name`.
   - Include a method `device_info()` to print the device’s name, power status, and WiFi connection status.

---

#### **Example Input/Output**:

##### **Example 1**:
```python
# Create a SmartDevice
smart_bulb = SmartDevice("Smart Bulb")

# Turn the device on
smart_bulb.toggle_power()
# Connect it to WiFi
smart_bulb.connect_wifi()

# Display device information
smart_bulb.device_info()
```

Output:
```plaintext
Device: Smart Bulb
Power Status: On
WiFi Status: Connected
```

##### **Example 2**:
```python
# Create another SmartDevice
smart_thermostat = SmartDevice("Smart Thermostat")

# Display initial status
smart_thermostat.device_info()

# Turn on power but leave WiFi disconnected
smart_thermostat.toggle_power()

# Display updated status
smart_thermostat.device_info()
```

Output:
```plaintext
Device: Smart Thermostat
Power Status: On
WiFi Status: Not Connected
```

---

#### **Detailed Explanation**:

1. **PowerDevice Class**:
   - Manages the device's power status (`On` or `Off`).
   - Provides a method `toggle_power()` to switch the state.

2. **WiFiDevice Class**:
   - Manages WiFi connectivity using a boolean `connected`.
   - Provides methods to connect or disconnect from WiFi.

3. **SmartDevice Class**:
   - Inherits from both `PowerDevice` and `WiFiDevice`.
   - Combines functionality from both parents and adds its own attribute (`device_name`) and method (`device_info()`).

4. **Multiple Inheritance**:
   - Demonstrates how to combine functionality from two different base classes.
   - Shows how to initialize multiple parent classes using `super()` or explicit calls.


#### **Edge Cases to Test**:
1. A device is powered off but connected to WiFi—ensure this is reflected in the output.
2. A device is created but neither powered on nor connected to WiFi.
3. Test multiple devices with independent states to ensure proper attribute handling.

"""

class PowerDevice:
    """Represents a device that requires power to operate."""
    
    def __init__(self):
        self.power_status = "Off"

    def toggle_power(self) -> None:
        """Toggles the power status of the device."""
        self.power_status = "On" if self.power_status == "Off" else "Off"


class WiFiDevice:
    """Represents a device that can connect to WiFi."""
    
    def __init__(self):
        self.connected = False

    def connect_wifi(self) -> None:
        """Connects the device to WiFi."""
        self.connected = True

    def disconnect_wifi(self) -> None:
        """Disconnects the device from WiFi."""
        self.connected = False


class SmartDevice(PowerDevice, WiFiDevice):
    """A device that is both power-operated and WiFi-enabled."""
    
    def __init__(self, device_name: str):
        PowerDevice.__init__(self)  # Initialize PowerDevice
        WiFiDevice.__init__(self)   # Initialize WiFiDevice
        self.device_name = device_name

    def device_info(self) -> None:
        """Displays the device's status."""
        wifi_status = "Connected" if self.connected else "Not Connected"
        print(f"\nDevice: {self.device_name}")
        print(f"Power Status: {self.power_status}")
        print(f"WiFi Status: {wifi_status}")



# Test cases
smart_bulb = SmartDevice("Smart Bulb")
smart_bulb.toggle_power()
smart_bulb.connect_wifi()
smart_bulb.device_info()

smart_thermostat = SmartDevice("Smart Thermostat")
smart_thermostat.device_info()
smart_thermostat.toggle_power()
smart_thermostat.device_info()