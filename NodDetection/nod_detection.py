import cv2
import numpy as np
import os
from datetime import datetime
from ffpyplayer.player import MediaPlayer

# Output video codec and writer
fourcc = cv2.VideoWriter_fourcc(*"XVID")
out = cv2.VideoWriter("/home/sm/Desktop/nodcontrol.avi",fourcc, 20.0, (640,480))
# Path root for image files on Y/N gesture trigger
# Dev
# img_root = os.path.join(os.path.expanduser("~"), "Desktop/learnopencv/NodDetection/resources")
# Production
img_root = os.path.join(os.path.expanduser("~"), "Desktop/learnopencv/NodDetection/prodResources")

# Capture source video and fix the dimensions
cap = cv2.VideoCapture(0)
MEDIA_WIDTH = 600
MEDIA_HEIGHT = 400
cap.set(cv2.CAP_PROP_FRAME_WIDTH, MEDIA_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, MEDIA_HEIGHT)

#params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )


# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                 maxLevel = 2,
                 criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

#path to face cascde
basepath = "C:/Users/anatu/Documents/GitHub/opencv/data/haarcascades"
face_cascade = cv2.CascadeClassifier(os.path.join(basepath, "haarcascade_frontalface_alt.xml"))
#######################################################################


# Helper functions
#######################################################################
# Function to build the decision-tree data structure which stores the media we want to be showing
def build_tree():

  # root = "C:/users/anatu/desktop"
  # result = dict()
  # # Initialize the current state to root
  # result["current"] = "root"
  # result["root"] = os.path.join(img_root, "waiter_test.jpg")
  # result["Y"] = os.path.join(root, "Nod_Demo.mp4")
  # result["N"] = os.path.join(root, "sound_test.mp4")

  result = dict()
  result["root"] = os.path.join(img_root, "Background.mp4")
  result["current"] = "root"
  result["Y"] = os.path.join(img_root, "I2_Yes.mp4")
  result["N"] = os.path.join(img_root, "I2_No.mp4")
  result["NN"] = os.path.join(img_root, "I3_No_No.mp4")
  result["YY"] = os.path.join(img_root, "I3_Yes_Yes.mp4")
  result["NY"] = os.path.join(img_root, "I3_No_Yes.mp4")
  result["YN"] = os.path.join(img_root, "I3_Yes_No.mp4")
  result["YNN"] = os.path.join(img_root, "I4.mp4")
  result["YNY"] = os.path.join(img_root, "I4.mp4")
  result["NNY"] = os.path.join(img_root, "I4.mp4")
  result["NNN"] = os.path.join(img_root, "I4.mp4")
  result["NYN"] = os.path.join(img_root, "I4.mp4")
  result["NYY"] = os.path.join(img_root, "I4.mp4")


  return result

# Euclidean distance between two points in (x,y) space
def distance(x,y):
   import math
   return math.sqrt((x[0]-y[0])**2+(x[1]-y[1])**2)

# Function to get coordinates of point
def get_coords(p1):
   try: return int(p1[0][0][0]), int(p1[0][0][1])
   except: return int(p1[0][0]), int(p1[0][1])
#######################################################################

def main():
  #######################################################################
  # Set the model parameters
  font = cv2.FONT_HERSHEY_SIMPLEX
  #define movement thresholds to recognize gestures
  max_head_movement = 20
  movement_threshold = 50
  gesture_threshold = 100
  # Set the number of frames for which to display the recognized gesture
  # (i.e. how long to show Yes or No after gesture is detected) 
  gesture_show = 30
  # Max number of frames for which a gesture can be detected
  gesture_timescale = 120

  # Build the tree with all the media paths
  mediaTree = build_tree()

  #######################################################################

  #######################################################################
  # Detect gestures once face has been identified in the image
  gesture = False
  x_movement = 0
  y_movement = 0
  gesture_fc = 0

  # Flag to force face detection in the first run
  start_flag = True
  face_found = False
  wait = False

  while True:
    if start_flag == True or face_found == False:
      start_flag = False
      frame_num = 0

      # Face finding step - Places the points which track
      # x and y displacement on the face for gesture detection
      #######################################################
      while frame_num < 30:
        # Take first frame and find corners in it
        frame_num += 1
        ret, frame = cap.read()
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(frame_gray, 1.3, 5)
        for (x,y,w,h) in faces:
          cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
          face_found = True
        cv2.putText(img=frame, text='Finding face...',org=(0,0), fontFace=font, fontScale=2, color=(255,255,255))
        cv2.imshow("face_video",frame)
        out.write(frame)
        cv2.waitKey(1)
      face_center = x+w/2, y+h/3
      p0 = np.array([[face_center]], np.float32)

    # Press the "z" key once the face has been found
    # to reset the face detection flow in case there's a bug
    if cv2.waitKey(5) & 0xFF == ord('z'):
      face_found = False
    #######################################################

    # Reset x/y displacements if the max number
    # of frames have passed
    gesture_fc += 1
    if gesture_fc == gesture_timescale:
      x_movement = 0
      y_movement = 0
      gesture_fc = 0

    ret,frame = cap.read()
    old_gray = frame_gray.copy()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    cv2.circle(frame, get_coords(p1), 4, (0,0,255), -1)
    cv2.circle(frame, get_coords(p0), 4, (255,0,0))

    #get the xy coordinates for points p0 and p1
    a,b = get_coords(p0), get_coords(p1)
    x_movement += abs(a[0]-b[0])
    y_movement += abs(a[1]-b[1])

    text = "x_movement: " + str(x_movement)
    if not gesture: cv2.putText(frame,text,(50,50), font, 0.8,(0,0,255),2)
    text = "y_movement: " + str(y_movement)
    if not gesture: cv2.putText(frame,text,(50,100), font, 0.8,(0,0,255),2)

    if x_movement > gesture_threshold:
      gesture = "No"
    if y_movement > gesture_threshold:
      gesture = "Yes"

    if gesture:
      if mediaTree["current"] == "root":
        mediaTree["current"] = ""

      if gesture == "Yes":
        newCurrent = mediaTree["current"] + "Y"
        media_path = mediaTree[mediaTree["current"] + "Y"]
      if gesture == "No":
        newCurrent = mediaTree["current"] + "N"
        media_path = mediaTree[mediaTree["current"] + "N"]

      mediaTree["current"] = newCurrent

      media_cap = cv2.VideoCapture(media_path)
      audio_cap = MediaPlayer(media_path)
      while(media_cap.isOpened()):
          _, mFrame = media_cap.read()
          audio_frame, val = audio_cap.get_frame()

          try:
            cv2.imshow('media',mFrame)
          except cv2.error:
            cv2.destroyWindow("media")
            break

          if val != 'eof' and audio_frame is not None:
            img, t = audio_frame

          if cv2.waitKey(20) & 0xFF == ord('q'):
            break
      wait = True  


      gesture = False
      x_movement = 0
      y_movement = 0

    # Play the root "waiter state" which shows the starting prompt 
    elif mediaTree["current"] == "root" and wait == False:
      waiterPath = mediaTree["root"]
      media_cap = cv2.VideoCapture(waiterPath)
      audio_cap = MediaPlayer(waiterPath)
      while(media_cap.isOpened()):
          _, mFrame = media_cap.read()
          audio_frame, val = audio_cap.get_frame()

          try:
            cv2.imshow('media',mFrame)
          except cv2.error:
            # media_cap.release()
            cv2.destroyWindow("media")
            break

          if val != 'eof' and audio_frame is not None:
            img, t = audio_frame

          if cv2.waitKey(30) & 0xFF == ord('q'):
            break
      wait = True

    if gesture_show == 0:
      gesture = False
      x_movement = 0
      y_movement = 0
      gesture_show = 60 #number of frames a gesture is shown

    p0 = p1

    # final = cv2.hconcat([newimg, newimg])
    cv2.imshow("face_video",frame)
    out.write(frame)
    cv2.waitKey(1)

  cv2.destroyAllWindows()
  cap.release()
  #######################################################################

if __name__ == "__main__":
  main()