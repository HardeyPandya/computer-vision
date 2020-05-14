import nxt.locator
from nxt.motor import *

class Mindstorms:
    def __init__(self):
        self.brick = nxt.locator.find_one_brick()
        self.motorLeft = Motor(self.brick, PORT_A)
        self.motorRight = Motor(self.brick, PORT_B)
        self.motorTrigger = Motor(self.brick, PORT_C)

    def moveHor(self, steps):
        if steps>0: 
            speed = +20
            #right
            self.brick.play_tone_and_wait(400.0,100)
            self.brick.play_tone_and_wait(600.0,100)
        else:
            speed = -20
            #left
            time.sleep(1)
            self.brick.play_tone_and_wait(600.0,100)
            self.brick.play_tone_and_wait(400.0,100)
        degrees = 50
        self.motorLeft.turn(speed, degrees)

    def moveVer(self, steps):
        if steps<0: 
            speed = +50
            #up
            self.brick.play_tone_and_wait(800.0,100)
            self.brick.play_tone_and_wait(1000.0,100)
        else:
            speed = -50
            #down
            self.brick.play_tone_and_wait(1000.0,100)
            self.brick.play_tone_and_wait(800.0,100)
        degrees = 50
        self.motorRight.turn(speed, degrees)	

    def shoot(self):
        self.brick.play_tone_and_wait(300.0,100)
        time.sleep(0.1)
        self.brick.play_tone_and_wait(100.0,100)

        self.motorTrigger.turn(100, 360)
        #self.motorTrigger.turn( 50, 50)
