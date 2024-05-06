# Text To Speech - Speech To Text - For The Win
---
This is a python application meant to be run from the command line.  It's goal is to analyze audio (wav files present in directory, or realitime audio via microphone)
It was some homework I was given and the description.txt provides more details on that.

## Prerequisites
** Python version 3.11 **
For the mac, you will need to have portaudio and ffmpeg installed on your system.
The suggested way is to use brew

`brew install portaudio`
`brew install ffmpeg`

## Installation
1. Git clone this repo to your computer.  Create a virtual environment `python3 -m venv homework`

2. Activate the venv `source homework/bin/activate`

3. Install python packages `pip install -r requirements.txt`

## Use
With the venv activated just run `python -m app`
You will be presented with a choice of using prerecorded audio files (1) or using real-time transcription with your computer's microphone (2).  If choosing option 2 make sure to just say 'Goodbye' to exit the program.
**On exit the final output for the assignment will be printed to the terminal**
