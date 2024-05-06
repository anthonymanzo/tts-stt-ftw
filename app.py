import asyncio
from dotenv import load_dotenv
import shutil
import subprocess
import requests
import time
import os
import argparse
import logging, verboselogs
from datetime import datetime
import httpx

from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    PromptTemplate,
)
from langchain.output_parsers import PydanticOutputParser

# the desired output as a pydantic model
from models import ContactInfo

from langchain.chains import LLMChain

from deepgram import (
    DeepgramClient,
    DeepgramClientOptions,
    LiveTranscriptionEvents,
    LiveOptions,
    Microphone,
    FileSource,
    PrerecordedOptions,
)


load_dotenv()
# example of setting up a client config. logging values: WARNING, VERBOSE, DEBUG, SPAM
config = DeepgramClientOptions(options={"keepalive": "true"})
deepgram: DeepgramClient = DeepgramClient(os.getenv("DEEPGRAM_API_KEY"), config)

# make a new file in the conversations folder, with a unique timestamped name to keep track of all transcriptions for future training and evaluation.  We will append to this file as we go.
file_name = f"conversations/convo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
convo_file = open(file_name, "w")


def get_final_output() -> ContactInfo:
    """
    Parse the final output from the conversation using an LLM and return the desired output as a pydantic model.
    """
    llm = ChatGroq(
        temperature=0,
        model_name=os.getenv("GROQ_MODEL_NAME"),
        groq_api_key=os.getenv("GROQ_API_KEY"),
    )

    cf = open(file_name, "r")
    full_convo = cf.read()
    parser = PydanticOutputParser(pydantic_object=ContactInfo)
    prompt = PromptTemplate(
        template="Read the full conversation transcript between the Human and the LLM to find the human's contact info.\n{format_instructions}\n{query}\n",
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    chain = prompt | llm | parser

    r = chain.invoke({"query": full_convo})
    # print(r)
    return r.dict()


class LLMProc:

    def __init__(self):
        self.llm = ChatGroq(
            temperature=0,
            model_name=os.getenv("GROQ_MODEL_NAME"),
            groq_api_key=os.getenv("GROQ_API_KEY"),
        )

        self.memory = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True
        )

        # Load the system prompt from a file so it's a little easier to version
        with open("system_prompt.txt", "r") as file:
            system_prompt = file.read().strip()

        self.prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                HumanMessagePromptTemplate.from_template("{text}"),
            ]
        )

        self.conversation = LLMChain(
            llm=self.llm, prompt=self.prompt, memory=self.memory
        )

    def process(self, text):
        self.memory.chat_memory.add_user_message(text)  # Add user message to memory

        start_time = time.time()

        # Go get the response from the LLM
        response = self.conversation.invoke({"text": text})
        end_time = time.time()

        self.memory.chat_memory.add_ai_message(
            response["text"]
        )  # Add AI response to memory

        elapsed_time = int((end_time - start_time) * 1000)
        print(f"LLM ({elapsed_time}ms): {response['text']}")
        return response["text"]


class TextToSpeech:
    # Deepgram API key and model name
    DG_API_KEY = os.getenv("DEEPGRAM_API_KEY")
    MODEL_NAME = "aura-helios-en"  # Example model name, change as needed

    @staticmethod
    def is_installed(lib_name: str) -> bool:
        lib = shutil.which(lib_name)
        return lib is not None

    def speak(self, text):
        if not self.is_installed("ffplay"):
            raise ValueError("ffplay not found, necessary to stream audio.")

        DEEPGRAM_URL = f"https://api.deepgram.com/v1/speak?model={self.MODEL_NAME}&performance=some&encoding=linear16&sample_rate=24000"
        headers = {
            "Authorization": f"Token {self.DG_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {"text": text}

        player_command = ["ffplay", "-autoexit", "-", "-nodisp"]
        player_process = subprocess.Popen(
            player_command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        start_time = time.time()  # Record the time before sending the request
        first_byte_time = None  # Initialize a variable to store the time when the first byte is received

        with requests.post(
            DEEPGRAM_URL, stream=True, headers=headers, json=payload
        ) as r:
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:
                    if (
                        first_byte_time is None
                    ):  # Check if this is the first chunk received
                        first_byte_time = (
                            time.time()
                        )  # Record the time when the first byte is received
                        ttfb = int(
                            (first_byte_time - start_time) * 1000
                        )  # Calculate the time to first byte
                        print(f"TTS Time to First Byte (TTFB): {ttfb}ms\n")
                    player_process.stdin.write(chunk)
                    player_process.stdin.flush()

        if player_process.stdin:
            player_process.stdin.close()
        player_process.wait()


class TranscriptCollector:
    def __init__(self):
        self.reset()

    def reset(self):
        self.transcript_parts = []

    def add_part(self, part):
        self.transcript_parts.append(part)

    def get_full_transcript(self):
        return " ".join(self.transcript_parts)


transcript_collector = TranscriptCollector()


async def get_transcript(callback):
    transcription_complete = asyncio.Event()  # Event to signal transcription completion

    try:
        # # example of setting up a client config. logging values: WARNING, VERBOSE, DEBUG, SPAM
        # config = DeepgramClientOptions(options={"keepalive": "true"})
        # deepgram: DeepgramClient = DeepgramClient(os.getenv("DEEPGRAM_API_KEY"), config)

        dg_connection = deepgram.listen.asynclive.v("1")
        print("Listening...")

        async def on_message(self, result, **kwargs):
            sentence = result.channel.alternatives[0].transcript

            if not result.speech_final:
                transcript_collector.add_part(sentence)
            else:
                # This is the final part of the current sentence
                transcript_collector.add_part(sentence)
                full_sentence = transcript_collector.get_full_transcript()
                # Check if the full_sentence is not empty before printing
                if len(full_sentence.strip()) > 0:
                    full_sentence = full_sentence.strip()
                    print(f"Human: {full_sentence}")
                    convo_file.write(f"Human: {full_sentence}\n")
                    callback(full_sentence)  # Call the callback with the full_sentence
                    transcript_collector.reset()
                    transcription_complete.set()  # Signal to stop transcription and exit

        dg_connection.on(LiveTranscriptionEvents.Transcript, on_message)

        options = LiveOptions(
            model="nova-2",
            punctuate=True,
            language="en-US",
            encoding="linear16",
            channels=1,
            sample_rate=16000,
            endpointing=300,
            smart_format=True,
        )

        await dg_connection.start(options)

        # Open a microphone stream on the default input device
        microphone = Microphone(dg_connection.send)
        microphone.start()

        await transcription_complete.wait()  # Wait for the transcription to complete instead of looping indefinitely

        # Wait for the microphone to close
        microphone.finish()

        # Indicate that we've finished
        await dg_connection.finish()

    except Exception as e:
        print(f"Could not open socket: {e}")
        return


class ConversationManager:
    def __init__(self):
        self.transcription_response = ""
        self.llm = LLMProc()

    async def main(self):
        def handle_full_sentence(full_sentence):
            self.transcription_response = full_sentence

        # Loop indefinitely until "goodbye" is detected
        while True:
            await get_transcript(handle_full_sentence)

            # Check for "goodbye" to exit the loop
            if "goodbye" in self.transcription_response.lower():
                convo_file.close()
                # Now trigger a separate llm call to parse the conversation and
                # return the desired json output (e.g. policy number, name, phone , etc.)
                final_output = get_final_output()
                break

            llm_response = self.llm.process(self.transcription_response)
            print(f"LLM: {llm_response}")
            convo_file.write(f"LLM: {llm_response}\n")
            tts = TextToSpeech()
            tts.speak(llm_response)

            # Reset transcription_response for the next loop iteration
            self.transcription_response = ""


class WavFile:
    def __init__(self):

        self.transcription_response = ""
        self.llm = LLMProc()

        """Initialize with command line arguments parsing."""
        parser = argparse.ArgumentParser(
            description="Choose a .wav file from the current directory."
        )
        parser.add_argument(
            "--dir", type=str, default=".", help="Directory to search for .wav files"
        )
        self.args = parser.parse_args()
        self.directory = self.args.dir

    def list_wav_files(self):
        """List all .wav files in the specified directory."""
        files = [f for f in os.listdir(self.directory) if f.endswith(".wav")]
        return files

    def choose_file(self):
        """Display .wav files and prompt the user to choose one."""
        wav_files = self.list_wav_files()
        if not wav_files:
            print("No .wav files found in the directory.")
            return None

        for index, file in enumerate(wav_files):
            print(f"{index + 1}: {file}")

        choice = int(input("Enter the number of the file you want to select: ")) - 1
        return wav_files[choice]

    def process_file(self):
        """Process the chosen file."""
        print("Processing the file...")
        # Implement the processing logic here
        try:

            with open(self.chosen_file, "rb") as file:
                buffer_data = file.read()

            payload: FileSource = {
                "buffer": buffer_data,
            }

            options: PrerecordedOptions = PrerecordedOptions(
                model="nova-2",
                smart_format=True,
                utterances=True,
                punctuate=True,
                diarize=True,
            )

            before = datetime.now()
            response = deepgram.listen.prerecorded.v("1").transcribe_file(
                payload, options, timeout=httpx.Timeout(300.0, connect=10.0)
            )
            after = datetime.now()
            self.transcription_response = (
                response.results.channels[0].alternatives[0].transcript
            )
            convo_file.write(f"Human: {self.transcription_response }\n")
            print(f"Human: {self.transcription_response}")
            llm_response = self.llm.process(self.transcription_response)
            print(f"LLM: {llm_response}")
            convo_file.write(f"LLM: {llm_response}\n")
            tts = TextToSpeech()
            tts.speak(llm_response)

            convo_file.close()
            final_output = get_final_output()
            print(f"Final output: {final_output}")
            print("")
            difference = after - before
            print(f"time: {difference.seconds}")

        except Exception as e:
            print(f"Exception: {e}")
        pass

    def run(self):
        """Execute the file selection process."""
        chosen_file = self.choose_file()
        if chosen_file:
            print(f"You have selected: {chosen_file}")
            self.chosen_file = chosen_file  # Save the chosen file
            self.process_file()


if __name__ == "__main__":
    print(
        """--- Welcome to my homework assignment! ---
    This program does some basic NER on either a .wav file or live audio input.
    It then uses a language model to generate a response.
          """
    )
    choice = input(
        "Do you want to (1) select a .wav file or (2) use your computer's microphone for real-time transcription? Enter 1 or 2: "
    )

    if choice == "1":
        wav_file = WavFile()
        wav_file.run()
    elif choice == "2":
        manager = ConversationManager()
        asyncio.run(manager.main())
