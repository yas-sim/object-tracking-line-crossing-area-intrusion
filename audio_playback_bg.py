import time

import pyaudio
import wave
import threading

# Audio background playback class (wav format is supported)
class audio_playback_bg:
    def __init__(self, wavfile:str, audio):   # audio = pyaudio object
        with wave.open(wavfile, 'rb') as wav:
            if wav.getsampwidth() != 2:       # Checking bits/sampling (bytes/sampling)
                raise RuntimeError("wav file {} does not have int16 format".format(wavfile))
            if wav.getframerate() != 16000:   # Checking sampling rate
                raise RuntimeError("wav file {} does not have 16kHz sampling rate".format(wavfile))
            self.wavdata = wav.readframes(wav.getnframes())

        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self.play_thread)
        self.exit_flag = False
        self.play_flag = False
        self.play_buf = None            # Current playback buffer
        self.audio = audio              # PyAudio object
        self.frame_size = 2048          # Audio frame size (samples / frame)
        self.sampling_rate = 16000      # Audio sampling rate
        self.playback_stream = self.audio.open(format=pyaudio.paInt16, channels=1, rate=self.sampling_rate, output=True, frames_per_buffer=self.frame_size)
        self.thread.start()

    def __del__(self):
        self.terminate_thread()

    def terminate_thread(self):
        self.exit_flag = True
        self.thread.join()

    def play_thread(self):
        while self.exit_flag == False:
            if self.play_flag == False:
                time.sleep(0.1)
                continue
            if self.play_buf is None:
                self.play_buf = self.wavdata[:]
            # Cut out an audio frame from the playback buffer
            if len(self.play_buf) > self.frame_size*2:
                play_data = self.play_buf[:self.frame_size*2]
                self.play_buf = self.play_buf[self.frame_size*2:]
            else:
                play_data = (self.play_buf+b'\0\0'*self.frame_size)[:self.frame_size*2]
                self.lock.acquire()
                self.play_flag = False
                self.lock.release()
                self.play_buf = None
            # Playback an audio frame
            self.playback_stream.write(frames=play_data, num_frames=self.frame_size)
            time.sleep(0.1)     # 16KHz, 2048samples = 128ms. Wait must be shorter than 128ms.
    
    def play(self):
        self.lock.acquire()
        self.play_flag = True
        self.lock.release()

    def stop(self):
        self.play_buf = None
        self.lock.acquire()
        self.play_flag = False
        self.lock.release()
