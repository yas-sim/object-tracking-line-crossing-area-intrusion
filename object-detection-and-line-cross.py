import sys
import time
import random

import yaml

import numpy as np
from numpy import linalg as LA, true_divide

import cv2
from scipy.spatial import distance
from munkres import Munkres               # Hungarian algorithm for ID assignment

import openvino as ov

from line_boundary_check import *

num_line_colors = 100
line_colors = [ [random.randint(64,255), random.randint(64,255), random.randint(64,255)] for _ in range(num_line_colors) ]

class BoundaryLine:
    def __init__(self, p0=(0,0), p1=(0,0)):
        self.p0 = p0
        self.p1 = p1
        self.color = (0,255,255)
        self.lineThinkness = 4
        self.textColor = (0,255,255)
        self.textSize = 4
        self.textThinkness = 2
        self.count1 = 0
        self.count2 = 0

# Draw single boundary line
def drawBoundaryLine(img, line):
    x1, y1 = line.p0
    x2, y2 = line.p1
    cv2.line(img, (x1, y1), (x2, y2), line.color, line.lineThinkness)
    cv2.putText(img, str(line.count1), (x1, y1), cv2.FONT_HERSHEY_PLAIN, line.textSize, line.textColor, line.textThinkness)
    cv2.putText(img, str(line.count2), (x2, y2), cv2.FONT_HERSHEY_PLAIN, line.textSize, line.textColor, line.textThinkness)
    cv2.drawMarker(img, (x1, y1),line.color, cv2.MARKER_TRIANGLE_UP, 16, 4)
    cv2.drawMarker(img, (x2, y2),line.color, cv2.MARKER_TILTED_CROSS, 16, 4)

# Draw multiple boundary lines
def drawBoundaryLines(img, boundaryLines):
    for line in boundaryLines:
        drawBoundaryLine(img, line)

# in: boundary_line = boundaryLine class object
#     trajectory   = (x1, y1, x2, y2)
def checkLineCross(boundary_line, trajectory):
    traj_p0  = (trajectory[0], trajectory[1])    # Trajectory of an object
    traj_p1  = (trajectory[2], trajectory[3])
    bLine_p0 = (boundary_line.p0[0], boundary_line.p0[1]) # Boundary line
    bLine_p1 = (boundary_line.p1[0], boundary_line.p1[1])
    intersect = checkIntersect(traj_p0, traj_p1, bLine_p0, bLine_p1)      # Check if intersect or not
    if intersect == True:
        angle = calcVectorAngle(traj_p0, traj_p1, bLine_p0, bLine_p1)   # Calculate angle between trajectory and boundary line
        if angle<180:
            boundary_line.count1 += 1
        else:
            boundary_line.count2 += 1
        #cx, cy = calcIntersectPoint(traj_p0, traj_p1, bLine_p0, bLine_p1) # Calculate the intersect coordination

# Multiple lines cross check
def checkLineCrosses(boundaryLines, objects):
    for obj in objects:
        traj = obj.trajectory
        if len(traj)>1:
            p0 = traj[-2]
            p1 = traj[-1]
            for line in boundaryLines:
                checkLineCross(line, [p0[0],p0[1], p1[0],p1[1]])


#------------------------------------
# Area intrusion detection
class Area:
    def __init__(self, contour):
        self.contour  = np.array(contour, dtype=np.int32)
        self.count    = 0

warning_obj = None


# Area intrusion check
def checkAreaIntrusion(areas, objects):
    for area in areas:
        area.count = 0
        for obj in objects:
            p0 = (obj.pos[0] + obj.pos[2])//2
            p1 = (obj.pos[1] + obj.pos[3])//2
            #if cv2.pointPolygonTest(area.contour, (p0, p1), False)>=0:
            if pointPolygonTest(area.contour, (p0, p1)):
                area.count += 1

# Draw areas (polygons)
def drawAreas(img, areas):
    for area in areas:
        if area.count > 0:
            color = (0,0,255)
        else:
            color = (255,0,0)
        cv2.polylines(img, [area.contour], True, color,4)
        cx, cy = np.mean(area.contour, axis=0).astype(int)
        cv2.putText(img, str(area.count), (cx, cy), cv2.FONT_HERSHEY_PLAIN, 4, color, 2)


#------------------------------------
# Object tracking

class object:
    def __init__(self, pos, feature, id=-1):
        self.feature = feature
        self.id = id
        self.trajectory = []
        self.time = time.monotonic()
        self.pos = pos

class objectTracker:
    def __init__(self):
        self.objectid = 0
        self.timeout  = 3   # sec
        self.clearDB()
        self.similarityThreshold = 0.4
        pass

    def clearDB(self):
        self.objectDB = []

    def set_timeout(self, timeout):
        self.timeout = timeout

    def evictTimeoutObjectFromDB(self):
        # discard time out objects
        now = time.monotonic()
        for object in self.objectDB:
            if object.time + self.timeout < now:
                self.objectDB.remove(object)     # discard feature vector from DB
                print("Discarded  : id {}".format(object.id))

    # objects = list of object class
    def trackObjects(self, objects):
        # if no object found, skip the rest of processing
        if len(objects) == 0:
            return

        # If any object is registred in the db, assign registerd ID to the most similar object in the current image
        if len(self.objectDB)>0:
            # Create a matix of cosine distance
            cos_sim_matrix=[ [ distance.cosine(objects[j].feature, self.objectDB[i].feature) 
                            for j in range(len(objects))] for i in range(len(self.objectDB)) ]
            # solve feature matching problem by Hungarian assignment algorithm
            hangarian = Munkres()
            combination = hangarian.compute(cos_sim_matrix)

            # assign ID to the object pairs based on assignment matrix
            for dbIdx, objIdx in combination:
                if distance.cosine(objects[objIdx].feature, self.objectDB[dbIdx].feature)<self.similarityThreshold:
                    objects[objIdx].id = self.objectDB[dbIdx].id                               # assign an ID
                    self.objectDB[dbIdx].feature = objects[objIdx].feature                     # update the feature vector in DB with the latest vector (to make tracking easier)
                    self.objectDB[dbIdx].time    = time.monotonic()                            # update last found time
                    xmin, ymin, xmax, ymax = objects[objIdx].pos
                    self.objectDB[dbIdx].trajectory.append([(xmin+xmax)//2, (ymin+ymax)//2])   # record position history as trajectory
                    objects[objIdx].trajectory = self.objectDB[dbIdx].trajectory

        # Register the new objects which has no ID yet
        for obj in objects:
            if obj.id==-1:           # no similar objects is registred in feature_db
                obj.id = self.objectid
                self.objectDB.append(obj)  # register a new feature to the db
                self.objectDB[-1].time = time.monotonic()
                xmin, ymin, xmax, ymax = obj.pos
                self.objectDB[-1].trajectory = [[(xmin+xmax)//2, (ymin+ymax)//2]]  # position history for trajectory line
                obj.trajectory = self.objectDB[-1].trajectory
                self.objectid+=1

    def drawTrajectory(self, img, objects):
        for obj in objects:
            if len(obj.trajectory)>1:
                line_col = line_colors[obj.id % num_line_colors]  # Use different color for each object
                cv2.polylines(img, np.array([obj.trajectory], np.int32), False, line_col, 4)


#------------------------------------


# DL models for pedestrian detection and person re-identification
model_det  = 'pedestrian-detection-adas-0002'           # 1,3,384,672 -> 1,1,200,7
model_reid = 'person-reidentification-retail-0277'      # 1,3,256,128 -> 1,256

model_det  = './intel/{0}/FP16/{0}'.format(model_det)
model_reid = './intel/{0}/FP16/{0}'.format(model_reid)

# Load boundary lines and areas from config
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)
boundaryLines = []
areas = []
for item in config:
    if item['type'] == 'wire':
        boundaryLines.append(BoundaryLine(item['points'][0], item['points'][1]))
    elif item['type'] == 'area':
        areas.append(Area(item['points']))

_N, _C, _H, _W = 0, 1, 2, 3

def main():
    global boundaryLines, areas
    global model_det, model_reid

    # Open USB webcams (or a movie file)
    '''
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH , 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)   
    '''
    infile = 'people-detection.264'
    cap = cv2.VideoCapture(infile)
    #'''
    ret, img = cap.read()
    ih, iw, _ = img.shape

    # Device name could be 'CPU', 'GPU', 'GPU.0', 'NPU', etc.
    device_det = 'CPU'
    device_reid = 'CPU'

    ov_config = {'CACHE_DIR' : './cache'}
    # Prep for face/pedestrian detection
    model_det  = ov.Core().read_model(model_det+'.xml')         # model=pedestrian-detection-adas-0002
    model_det_shape = model_det.input().get_shape()
    compiled_model_det    = ov.compile_model(model_det, device_det, ov_config)
    ireq_det = compiled_model_det.create_infer_request()

    # Preparation for face/pedestrian re-identification
    model_reid = ov.Core().read_model(model_reid+'.xml')        # person-reidentificaton-retail-0079
    model_reid_shape = model_reid.input().get_shape()
    compiled_model_reid    = ov.compile_model(model_reid, device_reid, ov_config)
    ireq_reid = compiled_model_reid.create_infer_request()

    tracker = objectTracker()
    tracker.set_timeout(10)  # Set timeout for object tracking

    fps = 0
    fps_count = 10
    loop_count = 0
    start_time = time.perf_counter()

    try:
        while cv2.waitKey(1)!=27:           # 27 == ESC
            ret, image = cap.read()
            if ret==False:
                del cap
                cap = cv2.VideoCapture(infile)
                continue
            inBlob = cv2.resize(image, (model_det_shape[3], model_det_shape[2]))
            inBlob = inBlob.transpose((2,0,1))
            inBlob = inBlob.reshape(list(model_det_shape))
            res = ireq_det.infer({0: inBlob})
            detObj = ireq_det.get_tensor('detection_out').data.reshape((200,7)) 
   
            objects = []
            for obj in detObj:                # obj = [ image_id, label, conf, xmin, ymin, xmax, ymax ]
                if obj[2] > 0.75:             # Confidence > 75% 
                    xmin = abs(int(obj[3] * image.shape[1]))
                    ymin = abs(int(obj[4] * image.shape[0]))
                    xmax = abs(int(obj[5] * image.shape[1]))
                    ymax = abs(int(obj[6] * image.shape[0]))
                    class_id = int(obj[1])

                    obj_img=image[ymin:ymax,xmin:xmax].copy()             # Crop the found object

                    # Obtain feature vector of the detected object using re-identification model
                    inBlob = cv2.resize(obj_img, (model_reid_shape[3], model_reid_shape[2]))
                    inBlob = inBlob.transpose((2,0,1))
                    inBlob = inBlob.reshape(model_reid_shape)
                    res = ireq_reid.infer({0: inBlob})
                    featVec = res[0][0].ravel()  # Get the feature vector of the object
                    objects.append(object([xmin,ymin, xmax,ymax], featVec, -1))

            outimg = image.copy()

            tracker.trackObjects(objects)
            tracker.evictTimeoutObjectFromDB()
            tracker.drawTrajectory(outimg, objects)

            checkLineCrosses(boundaryLines, objects)
            drawBoundaryLines(outimg, boundaryLines)

            checkAreaIntrusion(areas, objects)
            drawAreas(outimg, areas)

            # Draw bounding boxes, IDs and trajectory
            for obj in objects:
                id = obj.id
                color = ( (((~id)<<6) & 0x100)-1, (((~id)<<7) & 0x0100)-1, (((~id)<<8) & 0x0100)-1 )
                xmin, ymin, xmax, ymax = obj.pos
                cv2.rectangle(outimg, (xmin, ymin), (xmax, ymax), color, 2)
                cv2.putText(outimg, 'ID='+str(id), (xmin, ymin - 7), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

            # Calculate and Draw FPS
            loop_count += 1
            if loop_count % fps_count == 0:
                end_time = time.perf_counter()
                fps = fps_count / (end_time - start_time)
                start_time = end_time
            out_msg = f'FPS: {fps:.2f} Det={device_det}, ReID={device_reid}'
            cv2.putText(outimg, out_msg, (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 6)
            cv2.putText(outimg, out_msg, (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            outimg = cv2.resize(outimg, (0, 0), fx=2.0, fy=2.0)

            cv2.imshow('image', outimg)
    except KeyboardInterrupt:
        pass

    cv2.destroyAllWindows()

if __name__ == '__main__':
    sys.exit(main() or 0)
