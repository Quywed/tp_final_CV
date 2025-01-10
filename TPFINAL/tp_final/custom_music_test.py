import os
import tkinter as tk
from tkinter import messagebox
import pygame

def list_sound_files(directory):
    """List all sound files in the given directory."""
    return [f for f in os.listdir(directory) if f.lower().endswith(('.wav', '.mp3', '.ogg', '.flac'))]

def play_sound():
    """Play the sound file entered in the textbox."""
    file_name = entry.get().strip()
    if not file_name:
        messagebox.showerror("Error", "Please enter a file name.")
        return
    
    file_path = os.path.join(current_directory, file_name)
    if not os.path.isfile(file_path):
        messagebox.showerror("Error", f"File '{file_name}' not found.")
        return
    
    try:
        pygame.mixer.init()
        pygame.mixer.music.load(file_path)
        pygame.mixer.music.play()
    except Exception as e:
        messagebox.showerror("Error", f"Could not play sound: {str(e)}")

def stop_sound():
    """Stop any currently playing sound."""
    if pygame.mixer.music.get_busy():
        pygame.mixer.music.stop()

def on_file_selected(event):
    """Fill the entry with the selected file name."""
    selection = listbox.curselection()
    if selection:
        entry.delete(0, tk.END)
        entry.insert(0, listbox.get(selection))

def update_file_list():
    """Update the listbox with sound files from the current directory."""
    listbox.delete(0, tk.END)
    files = list_sound_files(current_directory)
    for file in files:
        listbox.insert(tk.END, file)

# Initial setup
current_directory = './custom_music'
if not os.path.exists(current_directory):
    os.makedirs(current_directory)

pygame.init()

# Create the main window
root = tk.Tk()
root.title("Sound Player")

# Create and pack the widgets
frame = tk.Frame(root)
frame.pack(pady=10)

entry = tk.Entry(frame, width=40)
entry.grid(row=0, column=0, padx=5)

play_button = tk.Button(frame, text="Play", command=play_sound)
play_button.grid(row=0, column=1, padx=5)

stop_button = tk.Button(frame, text="Stop", command=stop_sound)
stop_button.grid(row=0, column=2, padx=5)

listbox = tk.Listbox(root, width=60, height=15)
listbox.pack(pady=10)
listbox.bind("<<ListboxSelect>>", on_file_selected)

update_file_list()

# Run the Tkinter event loop
root.mainloop()