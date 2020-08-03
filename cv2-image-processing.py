import cv2 as cv2
import numpy as np


# initialising variables
font = cv2.FONT_HERSHEY_SIMPLEX
crosshair_size = 10
scale_factor = 0.3

# define range of blue color in HSV
lower_blue = np.array([80,50,50])
upper_blue = np.array([130,255,255])


def scale(img, scale_factor):
    height, width = img.shape[:2]
    new_h, new_w = int(height * scale_factor), int(width * scale_factor)
    img = cv2.resize(img, (new_w,new_h), interpolation = cv2.INTER_AREA)
    return img, new_h, new_w

def crosshair(img, centre_X, centre_Y, crosshair_size):
    cv2.line(img, (int(centre_X - crosshair_size), centre_Y), (int(centre_X + crosshair_size), centre_Y), (0, 255, 0), 2)
    cv2.line(img, (centre_X, int(centre_Y - crosshair_size)), (centre_X, int(centre_Y + crosshair_size)), (0, 255, 0), 2)

def get_displacement(change_x, change_y):
    x_direction = 'LEFT' if change_x < 0 else 'RIGHT'
    y_direction = 'UP' if change_y < 0 else 'DOWN'

    return x_direction, y_direction

def get_yaw_angle(lm, rm, bm, tm, left_base, right_base):
    #############################################
    # do refer to the picture above !
    # lm: left-most point onf minArea boundary
    # rm: right-most point onf minArea boundary
    # bm: bottom-most point onf minArea boundary
    # tm: tm-most point onf minArea boundary
    # left_base: basepoint for when turning CCW
    # right_base: basepoint for when turning CW
    #############################################

    # calculate the length of the minArea rect
    nl_m, nr_m, nb_m, nt_m = (np.array(i) for i in [lm, rm, bm, tm])
    top_right_len = np.linalg.norm(nt_m - nr_m)
    top_left_len = np.linalg.norm(nt_m - nl_m)

    # rotate CCW
    if top_right_len < top_left_len:
        rotation = 'CCW'
        tri_base = left_base[0] - bm[0]
        tri_height = left_base[1] - rm[1]

    # rotate CW
    elif top_right_len > top_left_len:
        rotation = 'CW'
        tri_base = bm[0] - right_base[0]
        tri_height = right_base[1] - lm[1]

    else:
        pass

    # if theta is either 0 or 90
    try:
        theta = np.rad2deg(np.arctan(tri_base/tri_height)).astype('int0')

    except:
        if top_right_len < top_left_len:
            theta = 0
            rotation = '-'

        elif top_right_len > top_left_len:
            theta = 90
            rotation = 'CW' # can be CCW too

    return theta, rotation


cap = cv2.VideoCapture('data/test2.mp4')

# uncomment to save video
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter('data/result2.avi', fourcc, 20.0, (int(cap.get(3)*scale_factor), int(cap.get(4)*scale_factor)))

while(cap.isOpened()):
    # Take each frame
    ret, frame = cap.read()

    k = cv2.waitKey(1)
    if not ret or k & 0xFF == ord('q'):
        break

    # frame is too big so scale down using udf
    frame, height, width = scale(frame, scale_factor)

    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(frame, frame, mask = mask)

    # blur
    blur = cv2.GaussianBlur(res,(5,5),0)

    # canny detection
    edges = cv2.Canny(blur,75,175)

    # find and draw contours
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    img_with_contours = frame.copy();
    cv2.drawContours(img_with_contours, contours[0], -1, (0,255,0), 3)

    # isolate one contour and draw straight rect
    cnt = contours[0]
    rect_bound = frame.copy()
    x,y,w,h = cv2.boundingRect(cnt)
    cv2.rectangle(rect_bound, (x,y), (x+w,y+h), (0,255,0), 2)

    # get centre of frame and draw marker
    centre_X, centre_Y = int(width/2), int(height/2)
    crosshair(rect_bound, centre_X, centre_Y, crosshair_size)

    # draw minArea rect
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    cv2.drawContours(rect_bound,[box],0,(0,0,255),2)

    # get boundary centre and endpoints
    M = cv2.moments(cnt)
    l_m = tuple(box[box[:,0].argmin()])
    r_m = tuple(box[box[:,0].argmax()])
    t_m = tuple(box[box[:,1].argmin()])
    b_m = tuple(box[box[:,1].argmax()])

    try:  # sometimes boundary doesnt exist, will encounter zero division error
        boundary_centre_x = int(M['m10']/M['m00'])
        boundary_centre_y = int(M['m01']/M['m00'])
        crosshair(rect_bound, boundary_centre_x, boundary_centre_y, crosshair_size)
    except ZeroDivisionError:
        continue

    # compute difference
    change_x = boundary_centre_x - centre_X
    change_y = boundary_centre_y - centre_Y
    x_direction, y_direction = get_displacement(change_x, change_y)

    # get bases of triangle to calculate yaw angle
    left_base = (x+w, y+h)
    right_base = (x, y+h)

    angle, rotation = get_yaw_angle(l_m, r_m, b_m, t_m, left_base, right_base)

    # put text
    cv2.putText(rect_bound, x_direction, (10,30), font, 0.5, (255,0,0), 1, cv2.LINE_AA)
    cv2.putText(rect_bound, y_direction, (10,60), font, 0.5, (255,0,0), 1, cv2.LINE_AA)
    cv2.putText(rect_bound, rotation, (10,90), font, 0.5, (255,0,0), 1, cv2.LINE_AA)
    cv2.putText(rect_bound, f'{abs(change_x)}', (70,30), font, 0.5, (255,0,0), 1, cv2.LINE_AA)
    cv2.putText(rect_bound, f'{abs(change_y)}', (70,60), font, 0.5, (255,0,0), 1, cv2.LINE_AA)
    cv2.putText(rect_bound, f'{angle}', (70,90), font, 0.5, (255,0,0), 1, cv2.LINE_AA)

    # show frame
    cv2.imshow('frame',frame)
    cv2.imshow('mask',mask)
    cv2.imshow('res',res)
    cv2.imshow('edges',edges)
    cv2.imshow('img_with_contours', img_with_contours)
    cv2.imshow('rect_bound', rect_bound)
#     out.write(rect_bound)

cap.release()
# out.release()
cv2.destroyAllWindows()
