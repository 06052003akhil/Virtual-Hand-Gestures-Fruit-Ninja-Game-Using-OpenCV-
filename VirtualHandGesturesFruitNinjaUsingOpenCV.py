import cv2
import time
import random
import mediapipe as mp
import math
import numpy as np

# Initialize MediaPipe components
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

class FruitSlicingGame:
    def __init__(self):
        # Use updated parameter names
        self.hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
        self.curr_Frame = 0
        self.prev_Frame = 0
        self.delta_time = 0

        self.next_Time_to_Spawn = 0
        self.Speed = [0, 5]
        self.Fruit_Size = 30
        self.Spawn_Rate = 1
        self.Score = 0
        self.Lives = 15
        self.Difficulty_level = 1
        self.game_Over = False

        self.slash = []
        self.slash_Color = (255, 255, 255)
        self.slash_length = 19

        self.Fruits = []

    def Spawn_Fruits(self):
        fruit = {}
        random_x = random.randint(15, 600)
        random_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        fruit["Color"] = random_color
        fruit["Curr_position"] = [random_x, 440]
        fruit["Next_position"] = [0, 0]
        self.Fruits.append(fruit)

    def Fruit_Movement(self):
        for fruit in self.Fruits:
            if fruit["Curr_position"][1] < 20 or fruit["Curr_position"][0] > 650:
                self.Lives -= 1
                self.Fruits.remove(fruit)
            else:
                cv2.circle(self.img, tuple(fruit["Curr_position"]), self.Fruit_Size, fruit["Color"], -1)
                fruit["Next_position"][0] = fruit["Curr_position"][0] + self.Speed[0]
                fruit["Next_position"][1] = fruit["Curr_position"][1] - self.Speed[1]
                fruit["Curr_position"] = fruit["Next_position"]

    def distance(self, a, b):
        return int(math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2))

    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not access the webcam.")
            return

        while cap.isOpened():
            success, self.img = cap.read()
            if not success:
                print("Skipping frame")
                continue

            h, w, _ = self.img.shape
            self.img = cv2.cvtColor(cv2.flip(self.img, 1), cv2.COLOR_BGR2RGB)
            self.img.flags.writeable = False
            results = self.hands.process(self.img)
            self.img = cv2.cvtColor(self.img, cv2.COLOR_RGB2BGR)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        self.img,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())

                    for id, lm in enumerate(hand_landmarks.landmark):
                        if id == 8:
                            index_pos = (int(lm.x * w), int(lm.y * h))
                            cv2.circle(self.img, index_pos, 18, self.slash_Color, -1)
                            self.slash.append(index_pos)
                            if len(self.slash) > self.slash_length:
                                self.slash.pop(0)

                            for fruit in self.Fruits:
                                d = self.distance(index_pos, fruit["Curr_position"])
                                cv2.putText(self.img, str(d), fruit["Curr_position"], cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 2, 3)
                                if d < self.Fruit_Size:
                                    self.Score += 100
                                    self.slash_Color = fruit["Color"]
                                    self.Fruits.remove(fruit)

            if self.Score % 1000 == 0 and self.Score != 0:
                self.Difficulty_level = int(self.Score / 1000) + 1
                self.Spawn_Rate = self.Difficulty_level * 4 / 5
                self.Speed[0] *= self.Difficulty_level
                self.Speed[1] = int(5 * self.Difficulty_level / 2)

            if self.Lives <= 0:
                self.game_Over = True

            if not self.game_Over:
                if time.time() > self.next_Time_to_Spawn:
                    self.Spawn_Fruits()
                    self.next_Time_to_Spawn = time.time() + (1 / self.Spawn_Rate)
                self.Fruit_Movement()
            else:
                cv2.putText(self.img, "GAME OVER", (int(w * 0.1), int(h * 0.6)), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 3)
                self.Fruits.clear()

            self.slash_array = np.array(self.slash, np.int32).reshape((-1, 1, 2))
            cv2.polylines(self.img, [self.slash_array], False, self.slash_Color, 15, 0)

            self.curr_Frame = time.time()
            self.delta_Time = self.curr_Frame - self.prev_Frame
            FPS = int(1 / self.delta_Time) if self.delta_Time > 0 else 0
            cv2.putText(self.img, "FPS : " + str(FPS), (int(w * 0.82), 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 250, 0), 2)
            cv2.putText(self.img, "Score: " + str(self.Score), (int(w * 0.35), 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 5)
            cv2.putText(self.img, "Level: " + str(self.Difficulty_level), (int(w * 0.01), 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 150), 5)
            cv2.putText(self.img, "Lives remaining : " + str(self.Lives), (200, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            self.prev_Frame = self.curr_Frame
            cv2.imshow("img", self.img)

            if cv2.waitKey(5) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    game = FruitSlicingGame()
    game.run()
