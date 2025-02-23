import asyncio
import websockets
import json
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# File paths for Web Attacks and DDoS Attacks
WEB_ATTACKS_FILE = "/home/haneen/GP-latest/results.txt"
DDOS_ATTACKS_FILE = "/home/haneen/GP-latest/ddos_results.txt"

CLIENTS = {"web_attacks": set(), "ddos_attacks": set()}  # Track clients per category
LAST_CONTENT = {"web_attacks": None, "ddos_attacks": None}  # Store last file content

# Function to read the file content
def read_file(file_path):
    try:
        with open(file_path, "r") as f:
            return f.read().strip()
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None

# Function to send updates to WebSocket clients
async def send_update(file_type):
    file_path = WEB_ATTACKS_FILE if file_type == "web_attacks" else DDOS_ATTACKS_FILE
    new_content = read_file(file_path)

    if new_content is not None and new_content != LAST_CONTENT[file_type]:
        LAST_CONTENT[file_type] = new_content
        message = json.dumps({"update": new_content})
        print(f"File changed! Sending update to {file_type} clients.")  # Debugging print

        if CLIENTS[file_type]:
            await asyncio.gather(*(client.send(message) for client in CLIENTS[file_type]))

# Watchdog handler for file changes
class FileChangeHandler(FileSystemEventHandler):
    def __init__(self, file_type):
        self.file_type = file_type

    def on_modified(self, event):
        if event.src_path in [WEB_ATTACKS_FILE, DDOS_ATTACKS_FILE]:
            print(f"Detected change in {self.file_type} file.")  # Debugging print
            loop.call_soon_threadsafe(asyncio.create_task, send_update(self.file_type))

# WebSocket connection handler
async def handler(websocket, path):
    # Determine which file the client wants to subscribe to
    if path == "/web_attacks":
        file_type = "web_attacks"
    elif path == "/ddos_attacks":
        file_type = "ddos_attacks"
    else:
        await websocket.close()
        return

    CLIENTS[file_type].add(websocket)
    print(f"New client connected to {file_type}: {websocket.remote_address}")

    try:
        async for _ in websocket:
            pass  # Keep connection open
    finally:
        CLIENTS[file_type].remove(websocket)
        print(f"Client disconnected from {file_type}: {websocket.remote_address}")

# Start WebSocket server
start_server = websockets.serve(handler, "0.0.0.0", 8765)
loop = asyncio.get_event_loop()
loop.run_until_complete(start_server)

# Start watchdog observers for both files
observer = Observer()
observer.schedule(FileChangeHandler("web_attacks"), path=WEB_ATTACKS_FILE, recursive=False)
observer.schedule(FileChangeHandler("ddos_attacks"), path=DDOS_ATTACKS_FILE, recursive=False)
observer.start()

print("WebSocket server running at ws://0.0.0.0:8765")  # Debugging print

try:
    loop.run_forever()
except KeyboardInterrupt:
    observer.stop()
    observer.join()