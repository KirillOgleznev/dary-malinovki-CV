## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2017 Intel Corporation. All Rights Reserved.

#####################################################
##              Align Depth to Color               ##
#####################################################

# First import the library
import pyrealsense2 as rs
# Import Numpy for easy array manipulation
import numpy as np
# Import OpenCV for easy image rendering
import cv2

import matplotlib.pyplot as plt

# Create a pipeline
pipeline = rs.pipeline()

# Create a config and configure the pipeline to stream
#  different resolutions of color and depth streams
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

if device_product_line == 'L500':
    config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
else:
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
profile = pipeline.start(config)

# Getting the depth sensor's depth scale (see rs-align example for explanation)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: ", depth_scale)

# We will be removing the background of objects more than
#  clipping_distance_in_meters meters away
clipping_distance_in_meters = 1  # 1 meter
clipping_distance = clipping_distance_in_meters / depth_scale

# Create an align object
# rs.align allows us to perform alignment of depth frames to others frames
# The "align_to" is the stream type to which we plan to align depth frames.
align_to = rs.stream.color
align = rs.align(align_to)


# Streaming loop

def nothing(x):
    pass


def draw_circle(event, x, y, flags, param):
    if (event == cv2.EVENT_LBUTTONDOWN):
        dist = aligned_depth_frame.get_distance(x, y)

        dist = round(dist * 100, 2)
        print(dist, 'см')


def show_plot(img):
    # plt.plot(img[240])
    # a = [aligned_depth_frame.get_distance(i, 240) * 100 for i in range(len(img[0]))]
    #
    # for i in range(len(img[0])):
    #     print(int(a[i]))
    #     bg_removed[int(a[i]*3)][i] = (0, 0, 255)
    pass
    # plt.plot(a)
    # plt.show()
    # cv2.imshow('tmp', bg_removed)
    # cv2.waitKey(1)
    #
    # breakpoint()
    # pass


cv2.namedWindow('Align Example', cv2.WINDOW_NORMAL)
cv2.setMouseCallback('Align Example', draw_circle)
# 50 - 100 (67)
cv2.createTrackbar("distance", "Align Example", 665, 1000, nothing)

colorizer = rs.colorizer()
colorizer.set_option(rs.option.visual_preset, 0)  # 0=Dynamic, 1=Fixed, 2=Near, 3=Far
colorizer.set_option(rs.option.min_distance, 0.6)
colorizer.set_option(rs.option.max_distance, 0.7)
colorizer.set_option(rs.option.color_scheme, 2)


try:
    while True:
        try:
            a = cv2.getTrackbarPos("distance", "Align Example")

            clipping_distance = (cv2.getTrackbarPos("distance", "Align Example") / 1000) / depth_scale
        except:
            pass
        # Get frameset of color and depth
        frames = pipeline.wait_for_frames()
        # frames.get_depth_frame() is a 640x360 depth image

        # Align the depth frame to color frame
        aligned_frames = align.process(frames)

        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame()  # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()

        # Validate that both frames are valid
        if not aligned_depth_frame or not color_frame:
            continue

        # 0.6 - 0.8
        print(aligned_depth_frame.get_units())
        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Remove background - Set pixels further than clipping_distance to grey
        grey_color = 153
        depth_image_3d = np.dstack(
            (depth_image, depth_image, depth_image))  # depth image is 1 channel, color is 3 channels
        bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), grey_color, color_image)
        print(clipping_distance)
        # Render images:
        #   depth align to color on left
        #   depth on right
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        images = np.hstack((bg_removed, depth_colormap))

        depth_colormap = np.asanyarray(colorizer.colorize(aligned_depth_frame).get_data())
        depth_colormap_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), 0, depth_colormap)

        show_plot(depth_colormap)
        cv2.imshow('Align Example', bg_removed)
        cv2.imshow('RealSense', depth_colormap_removed)

        key = cv2.waitKey(1)
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break
finally:
    pipeline.stop()
