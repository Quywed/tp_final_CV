
# Interactive Music Creation Through Motion Detection

  

This project enables users to create music interactively using motion detection, face tracking, and hand gesture recognition technology. Users can perform American Sign Language (ASL) gestures to produce musical notes and control various aspects of the music creation process.

  

## Prerequisites

  

### Hardware Requirements

- Webcam (built-in or external)

- Computer with sufficient processing power for real-time video analysis

  

### Software Requirements

- Python 3.8.10

- Visual Studio Code (recommended IDE)

- Blender

- Git  - For repository cloning 

  
## Steps-by-Step Project Installation
In this section we're going to explore the steps needed to get the project up and running.

# 1º Step - Installing VSCode (Optional, but recommended)
To have the best experience possible with this project, installing the IDE VSCode is highly recomended.
To do so, we recommend following the tutorial linked down below:

<a href="https://www.youtube.com/watch?v=D2cwvpJSBX4" target="_blank"><img src="https://i.ytimg.com/vi/D2cwvpJSBX4/maxresdefault.jpg" 
alt="IMAGE ALT TEXT HERE" width="981" height="552" border="10" /></a>

# 2º Step - Creating Virtual Environment

### Install Python Libraries
Use the command an
- threading

- scikit-learn

- mediapipe

- opencv-python

- pygame

- tkinter

- ultralytics (YOLOv11)

  

## Installation

  

1. Clone the repository

2. Create a virtual environment:

```bash

python  -m  venv  venv

```

3. Activate the virtual environment:

- Windows: `venv\Scripts\activate`

- Linux/Mac: `source venv/bin/activate`

4. Install dependencies:

```bash

pip  install  -r  requirements.txt

```

  

## Usage

  

1. Start the program:

```bash

python  main.py

```

  

2. Basic Controls:

- Display ASL gestures for letters A, B, C, D, and I to play different musical notes

- Show letter U to pause any playing instrument

- Tilt face up/down to change piano octave

- Show two hands simultaneously to start/stop metronome

- Hover index finger over the question mark (?) for help menu

  

### Instrument Selection

Type 'obj' in the terminal and press Enter to activate object detection. Present the following objects to change instruments:

  

| Object to Show | Instrument/Function |

|----------------|-------------------|

| Bottle | Bongo |

| Phone | Piano |

| Potted Plant | Drums |

| Cup | Random Music |

| Backpack | Background Music |

  

To stop object detection, type 'stop' and press Enter.

  

### Default Piano Notes

| ASL Letter | Musical Note |

|------------|-------------|

| A | Do |

| B | Re |

| C | Mi |

| D | Fa |

| I | Sol |

  

## Features

  

- ASL gesture recognition for musical note creation

- Face tracking for octave control

- Object detection for instrument selection

- Background music selection

- Built-in metronome (100 BPM)

- Interactive help menu

- Multiple instrument support

- Real-time audio playback

  

## Project Structure

  

```

project/

├── bongo/ # Bongo sound files

├── drums/ # Drum sound files

├── metronome/ # Metronome sound files

├── piano/ # Piano sound files

├── custom_music/ # Background music files

├── main.py # Main program file

└── requirements.txt

```

  

## Limitations

  

- The program currently supports one-handed instrument play only

- Rapid gesture transitions may occasionally trigger unintended notes

- Limited to 5 ASL letters to ensure gesture recognition accuracy

  

## Credits

  

- ASL detection adapted from SohamPrajapati's sign-language-detector-flask-python

- Sound samples from Freesound.org

- Object detection powered by YOLOv11 from Ultralytics

- Face and hand tracking powered by Google's MediaPipe

  

## License

  

[Include your license information here]

  

## Contributing

  

[Include contribution guidelines if applicable]
