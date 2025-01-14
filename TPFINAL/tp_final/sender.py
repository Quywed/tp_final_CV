import socket
import numpy as np
import json
import time
import threading
import queue

class BlenderAudioSender:
    def __init__(self, host='127.0.0.1', port=65432):
        self.host = host
        self.port = port
        self.audio_queue = queue.Queue()
        self.running = True
        self.sock = None
        self.connected = False
        
        # Start sender thread
        self.sender_thread = threading.Thread(target=self._sender_loop)
        self.sender_thread.daemon = True
        self.sender_thread.start()
    
    def _connect(self):
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.sock.bind((self.host, self.port))
            self.sock.listen()
            print(f"Waiting for Blender connection on {self.host}:{self.port}")
            self.conn, addr = self.sock.accept()
            print(f"Connected to Blender at {addr}")
            self.connected = True
        except Exception as e:
            print(f"Connection error: {e}")
            self.connected = False
    
    def _sender_loop(self):
        self._connect()
        while self.running:
            try:
                if not self.connected:
                    self._connect()
                    continue
                
                # Get audio data from queue
                try:
                    data = self.audio_queue.get(timeout=0.1)
                    data_json = json.dumps(data)
                    self.conn.sendall(data_json.encode() + b'\n')
                except queue.Empty:
                    # Send zero data when no sound is playing
                    zero_data = [0.0] * 256
                    data_json = json.dumps(zero_data)
                    self.conn.sendall(data_json.encode() + b'\n')
                    time.sleep(0.01)
                
            except Exception as e:
                print(f"Sender error: {e}")
                self.connected = False
                time.sleep(0.1)
    
    def send_audio_data(self, audio_data):
        """Add audio data to the queue for sending to Blender"""
        self.audio_queue.put(audio_data)
    
    def cleanup(self):
        """Clean up resources"""
        self.running = False
        if self.sock:
            self.sock.close()