# Text To Speech - Speech To Text - For The Win
---
This is a python application meant to be run from the command line.  It's goal is to analyze audio (wav files present in directory, or realitime audio via microphone)
It was some homework I was given and the description.txt provides more details on that.

## Prerequisites
**Python version 3.11**
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

### Why didn't you?
- Develop this app as a container (Docker)?  Since using devices (speaker/microphone) is both iffy and laggy between a container and the host computer, there was no value added in containerizing.
- Make a web-based app so that you don't have to install stuff?  My assignment was to build a local python app, so that's what I did.  I do like the idea of using webrtc and the browser to build something like this for a future project with my favorite stack (React/FastAPI).. maybe later.
- Train a foundation model to handle the speech recognition and NER?  I had only 1 day to do this assignment and didn't think I could beat what was available off-the-shelf in that timeframe.
- Create an ML pipeline to fine-tune the models in use over time.  I had only 1 day to do this... and I didn't want to miss the core requiremnents or write a bunch of code I would never use (and because a small amount of training data risks an overfit as well)
- Use any emojis?  Well, I'm getting older, I drive a minivan and wear cargo pants, why even pretend to be cool anymore 🤓
