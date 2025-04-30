import win32api as wapi
import time

# List of keys we're interested in monitoring
keyList = [
    'W', 'A', 'S', 'D',
    'Q',  # For quitting
    'P',  # For pausing
    'R',  # For resetting/restarting
    'E'   # For pause/resume recording
]

# Virtual key codes for special keys
VK_UP = 0x26
VK_DOWN = 0x28
VK_LEFT = 0x25 
VK_RIGHT = 0x27

def key_check():
    """
    Returns a list of keys that are currently being pressed
    """
    keys = []
    # Check regular single-character keys
    for key in keyList:
        if wapi.GetAsyncKeyState(ord(key)):
            keys.append(key)
    
    # Check special keys with virtual key codes
    if wapi.GetAsyncKeyState(VK_UP):
        keys.append("UP")
    if wapi.GetAsyncKeyState(VK_DOWN):
        keys.append("DOWN")
    if wapi.GetAsyncKeyState(VK_LEFT):
        keys.append("LEFT")
    if wapi.GetAsyncKeyState(VK_RIGHT):
        keys.append("RIGHT")
    
    return keys