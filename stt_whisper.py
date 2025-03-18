from pydub import AudioSegment
from transformers import pipeline
import os
import warnings

# Suppress Warnings
warnings.filterwarnings("ignore")

# Now we can select the language
language = input("Enter language code (e.g., 'en' for English): ").strip()

#Using whisper model we can load and set language
stt_pipeline = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-small",
    generate_kwargs={"language": language}  
)

#This code will load the audio file
audio = AudioSegment.from_mp3("becky-dating-voice-message-28843.mp3")

#we have to split to 30 second chunks
chunk_length = 30 * 1000  # 30 seconds in milliseconds
chunks = [audio[i: i + chunk_length] for i in range(0, len(audio), chunk_length)]

#This is the process of chunk
full_transcription = ""
for i, chunk in enumerate(chunks):
    chunk_path = f"chunk_{i}.wav"
    chunk.export(chunk_path, format="wav")
    result = stt_pipeline(chunk_path)
    full_transcription += result["text"] + " "
    
    os.remove(chunk_path)  

#Now it will save transcription to a text file
output_file = f"transcription_{language}.txt"
with open(output_file, "w", encoding="utf-8") as f:
    f.write(full_transcription.strip())

print(f"Transcription saved to {output_file}")
