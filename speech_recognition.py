from time import time
from dotenv import load_dotenv
from openai import OpenAI
import pyaudio
import wave

load_dotenv()

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100

def timer(func):
    def wrapper(*args, **kwargs):
        start_time = time()
        result = func(*args, **kwargs)
        end_time = time()
        print(f'Function {func.__name__!r} executed in {(end_time - start_time):4f}s')
        return result
    return wrapper

def record_voice(seconds=5):
    p = pyaudio.PyAudio()
    try:
        stream = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK,
        )

        print("Start recording...")
        frames = []
        for i in range(int(RATE / CHUNK * seconds)):
            data = stream.read(CHUNK)
            frames.append(data)

        print("Recording stopped")

        stream.stop_stream()
        stream.close()
        p.terminate()

        file_name = 'voice_file.wav'
        wf = wave.open(file_name, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()
        return file_name
    
    except OSError as e:
        print(e)
        p.terminate()
        return None

@timer
def transcribe_text(filename):
    client = OpenAI()
    try:
        print("Transcribing...")
        with open(filename, 'rb') as audio_file:
            transcript = client.audio.transcriptions.create(
                model='whisper-1',
                file=audio_file,
            )
        text = transcript.text
        print("Transcription:")
        print(text)
        return text

    except Exception as e:
        print(f"Transcription error: {e}\n")
        return None

def load_prompt(prompt_name):
    try:
        with open(prompt_name, 'r') as f:
            return f.read()
    except FileNotFoundError:
        raise ValueError("Incorrect file name")


@timer
def generate_text_analysis(text, prompt):
    client = OpenAI()
    completion = client.chat.completions.create(
        model='gpt-4o-mini',
        messages=[
            {'role': 'system', 'content': prompt},
            {'role': 'user', 'content': text}
        ],
        temperature=0.1,
        max_tokens=500,
    )
    return completion.choices[0].message.content

def voice_to_text_with_analysis(prompt):
    while True:
        record = input("Press r to record or q to quit: ")
        record = record.lower()
        if record == 'r':
            file_name = record_voice(10)
            text = transcribe_text(file_name)
            result = generate_text_analysis(text, prompt)

            print("RESULT:")
            print(result)
            print()

        elif record == 'q':
            break

if __name__ == '__main__':
    PROMPT = load_prompt('prompt.txt')
    voice_to_text_with_analysis(PROMPT)
