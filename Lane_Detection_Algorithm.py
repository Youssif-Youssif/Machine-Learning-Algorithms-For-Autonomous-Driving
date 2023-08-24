#Team 13
#Lane Detection using CV
#This program will emulate the POV of a camera mounted infront of a vehicle
#Thefore it will allow the camera to centre upon the curent lane of travel to ensure the system stays within these bounds


import cv2
#imported cv2 for computer vision related tasks (reading and processing videos or images)
import numpy as np
#imported numpy for linear algebra calculations involving matrices and array operations
import matplotlib.pyplot as plt

#Importing the image used for testing the program
#The image must be the POV of a centered front end camera
img = cv2.imread(r'C:\Users\youss\PycharmProjects\capstone\StaticImageTest.jpg')
#Convert the image to greyscale, this is done to easily recognize the change in intensity
#and idenitfy edges
grayscale_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

#Here begins the first step of the canny edge detector
# We applied a gaussian blue inorder to reduce noise and smoothen out the intesnity values
#This is done to prevent false edge detection caused by noise
Gaussian_noise_smoothing = cv2.GaussianBlur(grayscale_img, (7, 7), 0.2)

# original image
plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

# grayscale image
plt.subplot(1, 3, 2)
plt.imshow(grayscale_img, cmap='gray')
plt.title('Grayscale Image')
plt.axis('off')

# Gaussian Filtered Image
plt.subplot(1, 3, 3)
plt.imshow(Gaussian_noise_smoothing, cmap='gray')
plt.title('Gaussian Filtered Image')
plt.axis('off')
plt.show()

#comparing the work to the function
Canny_edge_detector = cv2.Canny(Gaussian_noise_smoothing, 50, 150)

plt.imshow(Canny_edge_detector)
plt.title("Canny Edge Detection Image")
plt.show()

# create a zero mask that will be applied later
mask = np.zeros_like(Canny_edge_detector)

#As seen in the plot above, there are other white objects that are identified, but are not within our interested lane
#Therefore, we must eliminate them buy focusing only on our lane
#Next, we will need to define the area in which the lane is visible for a camera
#Since the camera is front and center mounted, we will assume it covers the coordinates below
#Note: these coordinates are not a cartesian system, but an Image Coordinate system for pixels obtained from the input image
#Define the 4 side figure to define the region of interest
heightY, widthX = img.shape[:2]
bottom_left=np.array([0, heightY])
top_left=np.array([449, 315])
top_right=np.array([490, 315])
bottom_right=np.array([widthX, heightY])


four_side_Area_of_Interest = np.array(
   [[
       tuple(bottom_left),
       tuple(top_left),
       tuple(top_right),
       tuple(bottom_right)
   ]],
   dtype=np.int32
)
#Next, we will convolute the area of interest on the zero mask, create a white 4 side figure to represent the lane
cv2.fillPoly(mask, four_side_Area_of_Interest, 255)
plt.imshow(mask)
plt.title("Relevant Area")
plt.show()

#The line below will be used to isolate for the lane in the canny image
Lane_in_Canny = cv2.bitwise_and(Canny_edge_detector, mask)
plt.imshow(Lane_in_Canny)
plt.title("Detected edges in Mask")
plt.show()


# Define the Hough transform parameters

#Dictoronary for the defined hough trasnsform paramters
hough_params = {
   'ρ': 1,
   'θ': 3.14 / 180,
   'threshold': 3,
   'min # of pixels for line': 10,
   'max # pixels between line': 10
}

# Run Hough Transform on the edge-detected image
#The Hough Transform will store the result in the form of line segments with start points (x1,y1) and
#end point segments of the line as (x2,y2)
Hough_Transform = cv2.HoughLinesP(
   Lane_in_Canny,
   hough_params['ρ'],
   hough_params['θ'],
   hough_params['threshold'],
   np.array([]),
   hough_params['min # of pixels for line'],
   hough_params['max # pixels between line'])


#Create a blank image to isolate and display the location of the detected lanes on
dup_img_1 = np.copy(img) * 0

dup_img_1 = np.zeros_like(
    img, dtype=np.uint8
)
dup_img_2 = np.zeros_like(
    img, dtype=np.uint8
)

# Draw lines on the blank image that will then be planted on the original image
for line in Hough_Transform:
   first_x_point, first_y_point, second_x_point, second_y_point = line[0]      #extract the current line detected in the hough transform
   cv2.line(dup_img_1,  #on the blank image
            (first_x_point, first_y_point),  #draw line segment from this starting point
            (second_x_point, second_y_point),  #to this end point
            (255, 0, 0), 10)

plt.imshow(dup_img_1, cmap='gray')
plt.title("Hough Transform - Line Detection")
plt.show()
# merge the canny greyscale canny image to obtain the 3-channel RGB image
dup_img_2[:, :, 0] = Canny_edge_detector  #blue channel
dup_img_2[:, :, 1] = Canny_edge_detector  #green channel
dup_img_2[:, :, 2] = Canny_edge_detector  #red channel

plt.imshow(dup_img_2)
plt.title("dummy picture for testing")
plt.show()

# Draw the lines on the edge image
lanes_Detected_Before_AOI = cv2.addWeighted(dup_img_2, 0.8, dup_img_1, 1, 0)

#Draw the 4 side figure to represent the area of interest
lanes_Detected_In_AOI = cv2.polylines(lanes_Detected_Before_AOI, four_side_Area_of_Interest, True, (0, 0, 255), 10)
plt.imshow(lanes_Detected_In_AOI)
plt.title("lanes_Detected_In_AOI")
plt.show()

# Draw the lines on the original image
def superimpose_lines(image, lines, color=[255, 0, 0], thickness=10):
    for line in lines:
        (
            p1, s1, pp2, ss2
        ) = line[0]
        cv2.line(
            image,
            (p1, s1),
            (pp2,ss2),
            color, thickness
        )

# Create a blank image to isolate and display the location of the detected lanes on
lanes_detected_img = np.copy(img)
superimpose_lines(lanes_detected_img, Hough_Transform, color=[255, 0, 0], thickness=5)

# isplay the original image with detected lanes
plt.imshow(cv2.cvtColor(lanes_detected_img, cv2.COLOR_BGR2RGB))
plt.title("Lanes Detected on Original Image")
plt.show()
# For Video/ real time navigation use the code below
#α=0.7 means the original image will have 70% influence in the final output.
#β=1.0 means the Hough lines will have 100% influence in the final outpu

def main(image, α=1, β=1., γ=0.):
   edges_img = edge_detection(image)
   masked_img = Relevant_Area(edges_img, Obatin_4sided_Polygon_Coords(image))
   houghed_lines = hough_lines(masked_img, ρ=1, θ=np.pi / 180, threshold=20, line_pixels=20, distancepixels=180)

   # Perform weighted addition of the Hough lines image and the initial image directly
   output = cv2.addWeighted(image, α, houghed_lines, β, γ)
   return output

#To itterate through multiple frames, aka a video, we will creata a function for each step

def Obatin_4sided_Polygon_Coords(image):
    rows, cols = image.shape[:2]  # Get the number of rows and columns (height and width) of the input image
    # Define fractions of the image width and height to determine the vertices
    bw = 0.2
    tw = 0.5
    h = 0.5
    vone = [cols * bw, rows]  # Bottom-left vertex @ 15% width and at the bottom of the image
    vtwo = [cols * tw, rows * h]  # top-left vertex @ 45% width and 60% height
    vthree = [cols * (1 - bw), rows]  # Bottom-right vertex @ 85% width and at the bottom of the image
    vfour = [cols * (1 - tw), rows * h]  # top-right vertex @ 55% width and 60% height


    v = np.array([[vone, vtwo, vfour, vthree]], dtype=np.int32)
    return v

def edge_detection(img, low_threshold=180, high_threshold=240):
   # Convert the image to greyscale, this is done to easily recognize the change in intensity
   # and idenitfy edges
   gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
   # We applied a gaussian blue inorder to reduce noise and smoothen out the intesnity values
   # This is done to prevent false edge detection caused by noise
   blurred_img = cv2.GaussianBlur(gray_img, (5, 5), 0) #5x5 kernel with 0 std
   #apply canny edge detector, which will be used to detect areas of rapid intensity change
   #values above high threshold are edges, values below are not, while values inbetween are edges based on their negbours
   canny_img = cv2.Canny(blurred_img, low_threshold, high_threshold)
   return canny_img     # Return the resulting edge-detected image.

def Relevant_Area(img, vertices):

   # Create a blank mask image that is the same size as the original image
   mask = np.zeros_like(img)
   # Define the fill color (transparency) for the mask
   fill_color = (255, 255, 255, 100)  # Transparent white (R, G, B, Alpha)
   # create the white polygon figure on the blank image
   cv2.fillPoly(mask, vertices, fill_color)
   # AND the polygon figure and the canny img
   #if both pixels are non zero, the resulting pixel is retain its orginal value
   #otherwisse, it will be set to black
   masked_image = cv2.bitwise_and(img, mask)
   return masked_image

#The Hough Transform is a feature extraction technique used in computer vision
# it was used to detect lines, to identify lane segments in this code
# it converts a cartesian plane to the hough space, where a line in image is a point in hough
#ρ, rho, represents the perpendicular distance from the origin
#θ,theta, represents the angle (in radians) between the x-axis and the line perpendicular to the detected line.
#the hough transform will consdier all possible linea for an edge point, with each edge point voting for lines that can go through it
#The accumulator array where votes are stored will have a threhold, and votes above this will be considered a line with rho and theta
#that will be converted to cartesian coordinates, tielding the endpoints of a detected line
def hough_lines(img, ρ, θ, threshold, line_pixels, distancepixels):
   #the HoughLines function returns an array of line segments represented by their endpoints (x1, y1, x2, y2)
   lines = cv2.HoughLinesP(img, ρ, θ, threshold, np.array([]), minLineLength=line_pixels,
                           maxLineGap=distancepixels)
   hough_output = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
   #using the endpoints foind ine "lines", the slope function will create a line from the endpoint and
   #display it on the image
   hough_output = line_fitting(hough_output, lines)
   return hough_output


#the following function will be used to identify which lane is the left lane and which is the right lane from the hough transform output
def find_lane_side(img_lines):
   #iterate over each line detected by the hough transform
   LL = [
       (
           (y2 - y1) / (x2 - x1),
           y1 - ((y2 - y1) / (x2 - x1)) * x1
       ) #calcaulted slope and y intercept of each line
                 #consider only the line segments with non-vertical slopes (x1 != x2) and negative slopes (indicating the left lane)
                 for l in img_lines
                 for (
           x1,
           y1,
           x2,
           y2) in l
                 if x1 != x2 and (y2 - y1) / (x2 - x1) < 0]
   #
   #do the same but for right lane
   RL = [
       (
           (y2 - y1) / (x2 - x1),
           y1 - ((y2 - y1) / (x2 - x1)) * x1)
                  for line in img_lines
                  for (
           x1,
           y1,
           x2,
           y2) in line
                  if x1 != x2 and (y2 - y1) / (x2 - x1) >= 0
   ]

   left_l = np.mean(LL, axis=0)
   right_l = np.mean(RL, axis=0)
   return left_l, right_l

#fitting the left and right lane lines to the detected lane segments and then drawing and highlighting
 # these fitted lines on the input image
def line_fitting(img, l):
   #create a copy image
   img = img.copy()
   #store vertices of the polygon that will be filled to highlight the lane
   poly_vertices = []
   order = [
       0,
       1,
       3,
       2]
   (
       LL, RL
   ) = find_lane_side(l)
   # Iterate over the fitted lines (left and right) to draw them on the image
   for m, b in [LL, RL]:
       rows, cols = img.shape[:2]
       s1 = int(rows)
       ss2 = int(rows * 0.6)
       p1 = int((s1 - b) / m)
       pp2 = int((ss2 - b) / m)
       poly_vertices.append(
           (p1, s1)
       )
       poly_vertices.append(
           (pp2, ss2)
       )
       # Draw lane line on image
       Line_Overlay(img, np.array(
           [
               [[p1, s1, pp2, ss2]]])
                    )
   # Rearrange the order of vertices to form a polygon for filling
   poly_vertices = [
       poly_vertices[i] for i in order
   ]
   cv2.fillPoly(img, pts=np.array([poly_vertices], 'int32'), color=(254, 135, 0))
   return cv2.addWeighted(img, 0.7, img, 0.4, 0.)

def Line_Overlay(img, lines, color=[0, 255, 0], thickness=10):
   for line in lines:
       for (
               p1,s1,pp2,ss2) in line:
           cv2.line(img,(p1, s1),(pp2, ss2),color,thickness)

fig, axes = plt.subplots(1, 2, figsize=(20, 10))

# Plot the input image
axes[0].imshow(img)
axes[0].set_title("Input Image")
axes[0].set_xticks([])
axes[0].set_yticks([])


# Plot the Lane Detection Result
result_img = main(img)
axes[1].imshow(result_img)
axes[1].set_title("Lane Detection Result")
axes[1].set_xticks([])
axes[1].set_yticks([])


input_video_path = r"C:\Users\youss\PycharmProjects\capstone\VideoTest.mp4"
output_video_path = r'C:\Users\youss\PycharmProjects\capstone\Result.mp4'

# Load the video file
cap = cv2.VideoCapture(input_video_path)

# Get the video's frame rate and size
fps = int(cap.get(cv2.CAP_PROP_FPS))
fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create the output video file
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (fw, fh))

# Process each frame of the video and save the processed frames to the output video
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Apply the lane detection algorithm to the frame (replace main with your lane detection function)
    processed_frame = main(frame)

    # Write the processed frame to the output video
    out.write(processed_frame)
