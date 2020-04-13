import nxt
import nxt.usbsock
from nxt.sensor import *

b = nxt.locator.find_one_brick(debug=True)
b.play_tone_and_wait(440.0, 100)
