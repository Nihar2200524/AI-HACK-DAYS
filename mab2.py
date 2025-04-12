import pygame
import sys
import time
import random
import threading
import torch
import cv2
from datetime import datetime

# ----------------- Pygame Setup ----------------- #
pygame.init()
siren_sound = pygame.mixer.Sound("siren.wav")
siren_playing = False

screen_width = 1000
screen_height = 800
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("4-Way Traffic Simulation")

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)
GREEN = (0, 255, 0)
GREY = (169, 169, 169)
BLUE = (0, 0, 255)
ORANGE = (255, 165, 0)

clock = pygame.time.Clock()
log_file = open("traffic_log.txt", "a")

# ----------------- Signal Class ----------------- #
class Signal:
    def __init__(self, x, y):  # Fixed constructor
        self.x = x
        self.y = y
        self.red = True
        self.green = False
        self.yellow = False

    def draw(self):
        pygame.draw.rect(screen, BLACK, (self.x, self.y, 20, 60))
        pygame.draw.circle(screen, RED if self.red else WHITE, (self.x + 10, self.y + 10), 7)
        pygame.draw.circle(screen, YELLOW if self.yellow else WHITE, (self.x + 10, self.y + 30), 7)
        pygame.draw.circle(screen, GREEN if self.green else WHITE, (self.x + 10, self.y + 50), 7)

    def setRed(self):
        self.red = True
        self.green = False
        self.yellow = False

    def setGreen(self):
        self.red = False
        self.green = True
        self.yellow = False

# ----------------- Define Signals ----------------- #
signalNorth = Signal(480, 130)
signalSouth = Signal(500, 630)
signalWest = Signal(300, 380)
signalEast = Signal(660, 400)
signals = [signalNorth, signalSouth, signalWest, signalEast]
currentGreen = 0
signals[currentGreen].setGreen()

# ----------------- Vehicle Class ----------------- #
class Vehicle:
    def __init__(self, x, y, direction, color, is_ambulance=False):  # Fixed constructor
        self.x = x
        self.y = y
        self.direction = direction
        self.color = color
        self.is_ambulance = is_ambulance
        self.speed = 4 if is_ambulance else 2

    def move(self):
        if self.direction == 'down':
            if self.is_ambulance or signalNorth.green or self.y < 120:
                self.y += self.speed
            if self.y > screen_height:
                vehicles.remove(self)
        elif self.direction == 'up':
            if self.is_ambulance or signalSouth.green or self.y > 670:
                self.y -= self.speed
            if self.y < -50:
                vehicles.remove(self)
        elif self.direction == 'right':
            if self.is_ambulance or signalWest.green or self.x < 290:
                self.x += self.speed
            if self.x > screen_width:
                vehicles.remove(self)
        elif self.direction == 'left':
            if self.is_ambulance or signalEast.green or self.x > 710:
                self.x -= self.speed
            if self.x < -50:
                vehicles.remove(self)

    def draw(self):
        if self.direction in ['left', 'right']:
            pygame.draw.rect(screen, self.color, (self.x, self.y, 50, 30))
        else:
            pygame.draw.rect(screen, self.color, (self.x, self.y, 30, 50))

vehicles = []
last_spawn_time = 0
spawn_delay = 2

def spawnVehicle():
    direction = random.choice(['left', 'right', 'up', 'down'])
    color = random.choice([BLUE, ORANGE, GREY])
    if direction == 'left':
        vehicles.append(Vehicle(screen_width + 10, 410, direction, color))
    elif direction == 'right':
        vehicles.append(Vehicle(-60, 390, direction, color))
    elif direction == 'up':
        vehicles.append(Vehicle(490, screen_height + 10, direction, color))
    elif direction == 'down':
        vehicles.append(Vehicle(510, -60, direction, color))

def drawIntersection():
    screen.fill(WHITE)
    pygame.draw.rect(screen, BLACK, (0, 375, 1000, 100))
    pygame.draw.rect(screen, BLACK, (475, 0, 50, 800))

    for i in range(0, 1000, 40):
        pygame.draw.rect(screen, WHITE, (i, 422, 20, 6))
    for i in range(0, 800, 40):
        pygame.draw.rect(screen, WHITE, (492, i, 6, 20))

    for signal in signals:
        signal.draw()

    for vehicle in vehicles[:]:
        vehicle.move()
        vehicle.draw()

def updateSignals():
    global currentGreen
    for signal in signals:
        signal.setRed()
    currentGreen = (currentGreen + 1) % len(signals)
    signals[currentGreen].setGreen()

def checkAmbulancePresence():
    return any(v.is_ambulance for v in vehicles)

def log_event(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_file.write(f"[{timestamp}] {message}\n")
    log_file.flush()

# ----------------- YOLO Detection Thread ----------------- #
def detect_ambulance_with_yolo():
    global siren_playing
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    model.conf = 0.4
    cap = cv2.VideoCapture(0)

    print("🚑 YOLOv5 Detection Thread Started")
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        results = model(frame)
        ambulance_detected = False
        for *box, conf, cls in results.xyxy[0]:
            label = model.names[int(cls)]
            if label == "truck":  # Replace with 'ambulance' if your model has it
                ambulance_detected = True
                break

        if ambulance_detected and not checkAmbulancePresence():
            log_event("Ambulance detected via YOLO")
            direction = random.choice(['left', 'right', 'up', 'down'])
            for signal in signals:
                signal.setRed()
            index = ['up', 'down', 'right', 'left'].index(direction)
            signals[index].setGreen()

            y_pos = 390 if direction == 'right' else 410 if direction == 'left' else -60 if direction == 'down' else screen_height + 10
            x_pos = -60 if direction == 'right' else screen_width + 10 if direction == 'left' else 510 if direction == 'down' else 490

            vehicles.append(Vehicle(x_pos, y_pos, direction, RED, True))

        cv2.imshow("YOLOv5 Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# ----------------- Main Loop ----------------- #
def main():
    global last_spawn_time, siren_playing
    threading.Thread(target=detect_ambulance_with_yolo, daemon=True).start()
    changeTime = time.time() + 5

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                log_file.close()
                pygame.quit()
                sys.exit()

        if checkAmbulancePresence():
            if not siren_playing:
                siren_sound.play(-1)
                siren_playing = True
        else:
            if siren_playing:
                siren_sound.stop()
                siren_playing = False

        if time.time() - last_spawn_time > spawn_delay:
            spawnVehicle()
            last_spawn_time = time.time()

        drawIntersection()

        if time.time() >= changeTime and not checkAmbulancePresence():
            updateSignals()
            changeTime = time.time() + 5

        pygame.display.update()
        clock.tick(60)

main()
