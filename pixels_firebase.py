import firebase_admin
from firebase_admin import credentials
from firebase_admin import db

#!/usr/bin/python
# Copyright (c) 2017 Adafruit Industries
# Author: Dean Miller
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

# Can enable debug output by uncommenting:
#import logging
#logging.basicConfig(level=logging.DEBUG)

from Adafruit_AMG88xx import Adafruit_AMG88xx
from time import sleep
import pymysql
import datetime
import numpy as np
#import pandas as pd


#import Adafruit_AMG88xx.Adafruit_AMG88xx as AMG88

# Default constructor will pick a default I2C bus.
#
# For the Raspberry Pi this means you should hook up to the only exposed I2C bus
# from the main GPIO header and the library will figure out the bus number based
# on the Pi's revision.
#
# For the Beaglebone Black the library will assume bus 1 by default, which is
# exposed with SCL = P9_19 and SDA = P9_20.
sensor = Adafruit_AMG88xx()
#add_db = pymysql.connect(
#    user='root',
#    passwd='12345',
#    host='192.168.35.137',
#    db='temp',
#    charset='utf8'
#)
#cursor = add_db.cursor(pymysql.cursors.DictCursor)
#sql = "SELECT * FROM datas;"
#cursor.execute(sql)
#result = cursor.fetchall()
#result = pd.DataFrame(result)


# Optionally you can override the bus number:
#sensor = AMG88.Adafruit_AMG88xx(busnum=2)

cred = credentials.Certificate('mykey.json')
firebase_admin.initialize_app(cred,{
    'databaseURL' : 'https://trespasser-detection-default-rtdb.firebaseio.com/' 
})

ref = db.reference()
#wait for it to boot
sleep(.1)
with open('data.txt', 'a') as u :

    while(1):
    
        #sql = "INSERT INTO 'datas' (data) VALUES (sensor.readPixels());"
    
        readPixels = sensor.readPixels()
        
        now = datetime.datetime.now()
        nowDatetime = now.strftime('%Y-%m-%d %H:%M:%S')
        print(nowDatetime)
        #sql = "INSERT INTO datas VALUES ('{}', '{}')".format(readPixels, nowDatetime)
        
        readPixel = str(readPixels)
        #u.write(nowDatetime)
        #u.write(", ")
        #u.write(readPixel)
        #u.write("\n")
        ref.update({ nowDatetime : readPixel})

        #cursor.execute(sql)
    
        #sql1 = "INSERT INTO datas(time) VALUES ('{}')".format(nowDatetime)
        #cursor.execute(sql1)
    
        #add_db.commit()
        print(readPixels)
        
        #sleep(1)
        sleep(1)        
        
    u.close()


