# This script will detect faces via your webcam and push the data to a InfluxDB instance.
# By Ignacio Van Droogenbroeck @hectorivand

import cv2

from datetime import datetime

from influxdb_client import Point, InfluxDBClient
from influxdb_client.client.write_api import SYNCHRONOUS

# Specify the InfluxDB bucket

bucket = "crowd-counter"

# In this file, you must set the parameters to connect to InfluxDB.

client = InfluxDBClient.from_config_file('influxdb_config.ini')

write_api = client.write_api(write_options=SYNCHRONOUS)
query_api = client.query_api()

cap = cv2.VideoCapture(0)

# Create the haar cascade
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

while(True):
	# Capture frame-by-frame
	ret, frame = cap.read()

	# Our operations on the frame come here
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# Detect faces in the image
	faces = faceCascade.detectMultiScale(
		gray,
		scaleFactor=1.2,
		minNeighbors=5,
		minSize=(30, 100)
	)

	# Write the quantity of faces detected on InfluxDB

	p = Point("public-count").tag("cameras", "entry").field("people", '{0}'.format(len(faces)))
	write_api.write(bucket=bucket, record=p)

	# Draw a rectangle around the faces
	for (x, y, w, h) in faces:
		cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

	# Display the resulting frame
	cv2.imshow('frame', frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
