import ctypes
import time

# Direct Input scan codes
# Reference: https://docs.microsoft.com/en-us/previous-versions/visualstudio/visual-studio-6.0/aa299374(v=vs.60)
W = 0x11
A = 0x1E
S = 0x1F
D = 0x20
UP = 0xC8
LEFT = 0xCB
RIGHT = 0xCD
DOWN = 0xD0
ENTER = 0x1C
ESC = 0x01

# C struct for input
PUL = ctypes.POINTER(ctypes.c_ulong)

class KeyBdInput(ctypes.Structure):
    _fields_ = [("wVk", ctypes.c_ushort),
                ("wScan", ctypes.c_ushort),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", PUL)]

class HardwareInput(ctypes.Structure):
    _fields_ = [("uMsg", ctypes.c_ulong),
                ("wParamL", ctypes.c_short),
                ("wParamH", ctypes.c_ushort)]

class MouseInput(ctypes.Structure):
    _fields_ = [("dx", ctypes.c_long),
                ("dy", ctypes.c_long),
                ("mouseData", ctypes.c_ulong),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", PUL)]

class Input_I(ctypes.Union):
    _fields_ = [("ki", KeyBdInput),
                ("mi", MouseInput),
                ("hi", HardwareInput)]

class Input(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong),
                ("ii", Input_I)]

class XboxController:
    """
    Simulates Xbox controller inputs in Forza Horizon 4
    """
    def __init__(self):
        # Load the SendInput function from user32.dll
        self.SendInput = ctypes.windll.user32.SendInput
        # Current state
        self.throttle = 0.0
        self.brake = 0.0
        self.steering = 0.0
        # Keys currently pressed
        self.keys_pressed = set()
    
    def press_key(self, key_code):
        """
        Simulates pressing a key
        """
        if key_code in self.keys_pressed:
            return  # Key already pressed
        
        extra = ctypes.c_ulong(0)
        ii_ = Input_I()
        ii_.ki = KeyBdInput(0, key_code, 0x0008, 0, ctypes.pointer(extra))
        x = Input(ctypes.c_ulong(1), ii_)
        self.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))
        self.keys_pressed.add(key_code)
    
    def release_key(self, key_code):
        """
        Simulates releasing a key
        """
        if key_code not in self.keys_pressed:
            return  # Key not pressed
        
        extra = ctypes.c_ulong(0)
        ii_ = Input_I()
        ii_.ki = KeyBdInput(0, key_code, 0x0008 | 0x0002, 0, ctypes.pointer(extra))
        x = Input(ctypes.c_ulong(1), ii_)
        self.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))
        self.keys_pressed.remove(key_code)
    
    def set_throttle(self, value):
        """
        Sets throttle value (0.0 to 1.0)
        """
        value = max(0.0, min(1.0, value))  # Clamp between 0 and 1
        self.throttle = value
        
        if value > 0.5:  # Press W if throttle is more than half
            self.press_key(W)
        else:
            self.release_key(W)
    
    def set_brake(self, value):
        """
        Sets brake value (0.0 to 1.0)
        """
        value = max(0.0, min(1.0, value))  # Clamp between 0 and 1
        self.brake = value
        
        if value > 0.5:  # Press S if brake is more than half
            self.press_key(S)
        else:
            self.release_key(S)
    
    def set_steering(self, value):
        """
        Sets steering value (-1.0 to 1.0, where -1 is left, 1 is right)
        """
        value = max(-1.0, min(1.0, value))  # Clamp between -1 and 1
        self.steering = value
        
        if value < -0.2:  # Press A if steering left is more than 20%
            self.press_key(A)
            self.release_key(D)
        elif value > 0.2:  # Press D if steering right is more than 20%
            self.press_key(D)
            self.release_key(A)
        else:  # Release both if steering is centered
            self.release_key(A)
            self.release_key(D)
    
    def reset(self):
        """
        Resets all controls to zero and releases all keys
        """
        for key in list(self.keys_pressed):
            self.release_key(key)
        self.throttle = 0.0
        self.brake = 0.0
        self.steering = 0.0