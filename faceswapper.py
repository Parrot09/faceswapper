from clarifai import rest
from clarifai.rest import ClarifaiApp
from clarifai.rest import Image as ClImage
import cv2
import numpy as np
import dlib

def showimage(image, caption='image'):
	cv2.imshow(caption, image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	
def shape_to_np(shape, dtype="int"):
	coords = np.zeros((68, 2), dtype=dtype)
	for i in range(0, 68):
		coords[i] = (shape.part(i).x, shape.part(i).y)
	return coords


source_file = 'bill-gates.jpg'
target_file = 'steve-jobs.jpg'

app = ClarifaiApp(api_key='a2689fbac33a4b96b49daab05d323cb5')

model = app.models.get('face-v1.3')

## Use a global file name
source_send = app.inputs.create_image_from_filename(source_file)
target_send = app.inputs.create_image_from_filename(target_file)
source_image = cv2.imread(source_file)
target_image = cv2.imread(target_file)

t_rows, t_cols, _ = target_image.shape
s_rows, s_cols, _ = source_image.shape

## Dlib stuff
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
source_gray = cv2.cvtColor(source_image, cv2.COLOR_BGR2GRAY)
target_gray = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY)

source_rects = detector(source_gray, 1)
target_rects = detector(source_gray, 1)

## image = ClImage(url='https://samples.clarifai.com/face-det.jpg')
response = model.predict([target_send, source_send])

target_bbox = []
for resu in response['outputs'][0]['data']['regions']:
	target_bbox.append(resu['region_info']['bounding_box'])	

source_bbox = []
for resu in response['outputs'][1]['data']['regions']:
	source_bbox.append(resu['region_info']['bounding_box'])

showimage(source_image, 'source image original')

showimage(target_image, 'target image original')


## Replace face(s) in target_image with face in source_image
# Extract face from source image, right now source image should have only one face
for face in source_bbox:
	top_row = int(face['top_row'] * s_rows)
	bottom_row = int(face['bottom_row'] * s_rows)
	left_col = int(face['left_col'] * s_cols)
	right_col = int(face['right_col'] * s_cols)
	source_face = source_image[top_row:bottom_row, left_col:right_col]

# For each face in target_image, scale the source image and then apply
for face in target_bbox:
	top_row = int(face['top_row'] * t_rows)
	bottom_row = int(face['bottom_row'] * t_rows)
	left_col = int(face['left_col'] * t_cols)
	right_col = int(face['right_col'] * t_cols)
	n_rows = bottom_row - top_row
	n_cols = right_col - left_col
	
	dst_face = target_image[top_row:bottom_row, left_col:right_col]
	src_face = cv2.resize(source_face, (n_cols, n_rows), interpolation=cv2.INTER_CUBIC)
	
	
	# Perform landmarks detection on src_face and dst_face
	src_gray = cv2.cvtColor(src_face, cv2.COLOR_BGR2GRAY)
	src_rects = detector(src_gray, 1)
	for i, rect in enumerate(src_rects):
		shape = predictor(src_gray, rect)
		src_points = shape_to_np(shape) # Now we have the points
	hull = []
	hullIndex = cv2.convexHull(src_points, returnPoints = False)
	for i in hullIndex:
		hull.append(src_points[i])		
	
	new_mask = np.zeros_like(src_face)
	cv2.fillConvexPoly(new_mask, np.int32(hull), (255, 255, 255))
	
	#showimage(new_mask)
	result = np.bitwise_and(src_face, new_mask)
	#showimage(result)
	
	#mask = 255 * np.ones(src_face.shape, src_face.dtype)
	
	width, height, channels = dst_face.shape
	center = (height/2, width/2)
	
	#clone = cv2.seamlessClone(src_face, dst_face, mask, (int(center[0]), int(center[1])), cv2.NORMAL_CLONE)
	clone = cv2.seamlessClone(result, dst_face, new_mask, (int(center[0]), int(center[1])), cv2.NORMAL_CLONE)
	#showimage(clone, 'swapping image')
	
	# Fit a line on clone
	#rows, cols = clone.shape[:2]
	#[vx,vy,x,y] = cv2.fitLine(src_points, cv2.DIST_L2, 0, 0.01, 0.01)
	#lefty = int((-x*vy/vx) + y)
	#righty = int(((cols-x)*vy/vx)+y)
	#cv2.line(clone,(cols-1,righty),(0,lefty),(0,255,0),2)
	
	#showimage(clone, 'line on clone')
	
	target_image[top_row:bottom_row, left_col:right_col] = clone
	

showimage(target_image, 'target image modified')