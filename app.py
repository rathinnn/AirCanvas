

import multiprocessing
import io
from pydub import AudioSegment
import speech_recognition as sr
import whisper
import tempfile
import os
import click
import cv2
#cv2.namedWindow('MediaPipe Hands')
#cv2.namedWindow('Canvas')
import numpy as np

import mediapipe as mp
def audio(q):
    temp_dir = tempfile.mkdtemp()
    save_path = os.path.join(temp_dir, "temp.wav")
    audio_model = whisper.load_model("medium")    
    
    r = sr.Recognizer()
    r.energy_threshold = 200
    r.pause_threshold = 1
    r.dynamic_energy_threshold = True

    with sr.Microphone(sample_rate=16000) as source:
        print("Online")
 
        while True:

            audio = r.listen(source)
            data = io.BytesIO(audio.get_wav_data())
            audio_clip = AudioSegment.from_file(data)
            audio_clip.export(save_path, format="wav")


            result = audio_model.transcribe(save_path,language='english', fp16 = False)



            predicted_text = result["text"]
            print(predicted_text)
            if("canvas" in predicted_text.lower()):
                process(predicted_text.lower(), q)
            print("Here")

def video(q):
    color = {}
    color['red'] = (0,0,255)
    color['blue'] = (255,0,0)
    color['green'] = (0,255,0)
    cur_color = 'red'

    size = {}
    size['small'] = 1
    size['medium'] = 2
    size['large'] = 4
    cur_size = 'medium'
    start = True
    save_ = False
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands

    # For webcam input:
    cap = cv2.VideoCapture(0)
    save = np.zeros((480, 640, 3), dtype = 'uint8')
    rpoints = []
    def paint(rpoints):
        points = rpoints
        if(len(points) > 1):
            cv2.line(save, points[-1], points[-2], color[cur_color], size[cur_size])
    with mp_hands.Hands(
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
        while cap.isOpened():
            #print("media")
            success, image = cap.read()
            h, w, _ = image.shape
            try:
                task = q.get_nowait()
                if task.color:
                    cur_color = task.color
                if task.size:
                    cur_size = task.size
                if task.start:
                    start = task.start
                if task.save:
                    save_ = task.save
            except:
                pass
            if not success:
                print("Ignoring empty camera frame.")

                continue


            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image)

 
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())
                    cx, cy = int(hand_landmarks.landmark[8].x * w), int(hand_landmarks.landmark[8].y * h)
                    rpoints.append((cx, cy))
                    #cv2.circle(save, (cx, cy), 3, (250, 250, 0), 6)
                    if(start):
                        paint(rpoints)

            cv2.imshow('Canvas', cv2.flip(save, 1))
            if save_:
                cv2.imwrite("canvas.jpg", save)
                save_ = False
            cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
            if cv2.waitKey(5) & 0xFF == 27:
                break

    cap.release()
class Task(object):
    
    def __init__(self, color, size, start, save):
        self.color = color
        self.size = size
        self.start = start
        self.save = save




def process(text, q):
    if("start" in text):
        q.put(Task(None, None, True, None))
    elif("pause" in text):
        q.put(Task(None, None, False, None))
    elif("save" in text):
        q.put(Task(None, None, None, True))
    elif("color" in text):
        if("red"  in text):
            color = "red"
        elif("blue" in text):
            color = "blue"
        elif("green" in text):
            color = "green"
        q.put(Task(color, None, None, None))
    elif("size" in text):
        if("large"  in text):
            size = "large"
        elif("medium" in text):
            size = "medium"
        elif("small" in text):
            size = "small"
        q.put(Task(None, size, None, None))

if __name__ == '__main__':
    queue = multiprocessing.Queue()

    p = multiprocessing.Process(target=audio, args=(queue,))
    p.start()
    video(queue)
