import cv2
import time
import numpy as np
import os
import itertools

# Notes
# EDGE DETECTION FOR VIDEO: https://www.geeksforgeeks.org/real-time-edge-detection-using-opencv-python/


class Skeletonizer():

	def set_mode(self, mode):
		# Specifies the type of model to use.
		
		# The protoFile defines the architecture of the neural net (how layers are arranged)
		
		# The .caffemodel file stores the pre-trained weights of the model
		
		# n_points is the number of data points used in that training dataset
		
		# kpNames are the names of the body parts intended to be mapped by each keypoint
		# as described in https://www.learnopencv.com/deep-learning-based-human-pose-estimation-using-opencv-cpp-python/

		# MPII and COCO are two different training datasets

		# Model architecture for VGGNet is as follows:
		# First 10 layers perform convolution and pooling to create image feature vectors
		# These are fed into a two-branch CNN
		# The first branch computes confidence maps of body part locations
		# The second branch computes affinity maps which encode the degree of association
		# between body parts
		MODE = mode

		if MODE is "COCO":
		    self.protoFile = "pose/coco/pose_deploy_linevec.prototxt"
		    self.weightsFile = "pose/coco/pose_iter_440000.caffemodel"
		    self.nPoints = 18
		    self.kpNames = ["Nose", "Shoulder_R", "Elbow_R", "Wrist_R", "Shoulder_L", "Elbow_L", "Wrist_L", \
		    "Hip_R", "Knee_R", "Ankle_R", "Hip_L", "Knee_L", "Ankle_L", "Eye_R", "Eye_L", "Ear_R", "Ear_L", "Bkgrd"]
		    self.POSE_PAIRS = [ [1,0],[1,2],[1,5],[2,3],[3,4],[5,6],[6,7],[1,8],[8,9],[9,10],[1,11],[11,12],[12,13],[0,14],[0,15],[14,16],[15,17]]

		elif MODE is "MPI" :
		    self.protoFile = "pose/mpi/pose_deploy_linevec_faster_4_stages.prototxt"
		    self.weightsFile = "pose/mpi/pose_iter_160000.caffemodel"
		    self.nPoints = 15
		    self.kpNames = ["Head", "Neck", "Shoulder_R", "Elbow_R", "Wrist_R", "Shoulder_L", "Elbow_L", "Wrist_L",\
		    "Hip_R", "Knee_R", "Ankle_R", "Hip_L", "Knee_L", "Ankle_L", "Chest", "Background"]
		    self.POSE_PAIRS = [[0,1], [1,2], [2,3], [3,4], [1,5], [5,6], [6,7], [1,14], [14,8], [8,9], [9,10], [14,11], [11,12], [12,13] ]


	def set_params(self):
		# Specify width and height of the imput image
		inWidth = 368
		inHeight = 368
		# Define the probability threshold used to draw circles onto the images
		# to mark out keypoints
		threshold = 0.1
		return inWidth, inHeight, threshold

	def prepare_out_data_file(self, outname):
		outfile = open(outname, "w")
		kp_X = [elem + "_X" for elem in self.kpNames]
		kp_Y = [elem + "_Y" for elem in self.kpNames]
		join = list(zip(kp_X, kp_Y))
		outfile.write(",".join(list(sum(join, ()))) + "\n")
		return outfile

	def track_video_from_file(self, filename, outname):

		self.set_mode("MPI")
		inWidth, inHeight, threshold = self.set_params()
		
		input_source = filename
		cap = cv2.VideoCapture(input_source)

		# cap = cv2.VideoCapture(0)
		hasFrame, frame = cap.read()
		vid_writer = cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame.shape[1],frame.shape[0]))
		# Load the neural network using the prototxt architecture spec and the pre-trained weights
		net = cv2.dnn.readNetFromCaffe(self.protoFile, self.weightsFile)

		# Prepare the file to store points
		outfile = self.prepare_out_data_file(outname)

		while cv2.waitKey(1) < 0:
		    t = time.time()
		    hasFrame, frame = cap.read()
		    frameCopy = np.copy(frame)
		    if not hasFrame:
		        cv2.waitKey()
		        break

		    frameWidth = frame.shape[1]
		    frameHeight = frame.shape[0]

		    inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight),
		                              (0, 0, 0), swapRB=False, crop=False)
		    net.setInput(inpBlob)
		    output = net.forward()

		    H = output.shape[2]
		    W = output.shape[3]
		    # Empty list to store the detected keypoints
		    points = []

		    for i in range(self.nPoints):
		        # confidence map of corresponding body's part.
		        probMap = output[0, i, :, :]

		        # Find global maxima of the probMap.
		        minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
		        
		        # Scale the point to fit on the original image
		        x = (frameWidth * point[0]) / W
		        y = (frameHeight * point[1]) / H

		        if prob > threshold : 
		            cv2.circle(frameCopy, (int(x), int(y)), 8, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
		            cv2.putText(frameCopy, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, lineType=cv2.LINE_AA)

		            # Add the point to the list if the probability is greater than the threshold
		            points.append((int(x), int(y)))
		        else :
		            points.append(None)

		    flatPoints = [str(item) for sublist in points for item in sublist]
		    outfile.write(",".join(flatPoints) + "\n")


		    # Draw Skeleton
		    for pair in self.POSE_PAIRS:
		        partA = pair[0]
		        partB = pair[1]

		        if points[partA] and points[partB]:
		            cv2.line(frame, points[partA], points[partB], (0, 255, 255), 3, lineType=cv2.LINE_AA)
		            cv2.circle(frame, points[partA], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
		            cv2.circle(frame, points[partB], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)

		    cv2.putText(frame, "time taken = {:.2f} sec".format(time.time() - t), (50, 50), cv2.FONT_HERSHEY_COMPLEX, .8, (255, 50, 0), 2, lineType=cv2.LINE_AA)
		    # cv2.putText(frame, "OpenPose using OpenCV", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 50, 0), 2, lineType=cv2.LINE_AA)
		    # cv2.imshow('Output-Keypoints', frameCopy)
		    cv2.imshow('Output-Skeleton', frame)

		    vid_writer.write(frame)
		outfile.close()
		vid_writer.release()


	# def track_video_from_feed(self):

	# 	self.set_mode("MPI")
	# 	inWidth, inHeight, threshold = self.set_params()
		
	# 	input_source = filename
	# 	cap = cv2.VideoCapture(input_source)

	# 	# cap = cv2.VideoCapture(0)
	# 	hasFrame, frame = cap.read()
	# 	vid_writer = cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame.shape[1],frame.shape[0]))
	# 	net = cv2.dnn.readNetFromCaffe(self.protoFile, self.weightsFile)


if __name__ == "__main__":
	skel = Skeletonizer()
	skel.track_video_from_file("reddy_video.mp4", "output_points.csv")
