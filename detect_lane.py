import cv2
import numpy as np
from binarize import warp, get_combined_binary
from calibrate import calibrate_camera

mtx = None
dist = None
left_fit = None
right_fit = None
found = False


def slide_and_fit(warped_image):
    """
    Finds out lanes in a given warped image using histogram and sliding window.
    :param warped_image: Undistorted and perspective transformed image
    :return: dictionary which contains left lane indices, right lane indices, polynomials for left and right lanes
    """
    histogram = np.sum(warped_image[warped_image.shape[0] / 2:, :], axis=0)
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((warped_image, warped_image, warped_image)) * 255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] / 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(warped_image.shape[0] / nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = warped_image.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = warped_image.shape[0] - (window + 1) * window_height
        win_y_high = warped_image.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (
        nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (
        nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Return everything we have
    ret = {}
    ret['left_fit'] = left_fit
    ret['right_fit'] = right_fit
    ret['nonzerox'] = nonzerox
    ret['nonzeroy'] = nonzeroy
    ret['left_lane_inds'] = left_lane_inds
    ret['right_lane_inds'] = right_lane_inds

    return ret


def fit(binary_warped, left_fit, right_fit):
    """
    Tries to find lane lines in a given image, given a previously lane-found image along with polynomials
    for left and right lanes
    :param binary_warped: Undistorted, perspective transformed, binary image
    :param left_fit: 2nd degree polynomial for left lane
    :param right_fit: 2nd degree polynomial for right lane
    :return: If number of lane indices detected are more than 7, then a dictionary is returned with polynomials and indices
     if not, None is returned
    """
    # Assume you now have a new warped binary image
    # from the next frame of video (also called "binary_warped")
    # It's now much easier to find line pixels!
    print(type(binary_warped))
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    left_lane_inds = ((nonzerox > (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] - margin)) & (
    nonzerox < (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] + margin)))
    right_lane_inds = (
    (nonzerox > (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] - margin)) & (
    nonzerox < (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] + margin)))

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # If we found less than 7 points, calculate fit again
    if leftx.shape[0] < 7 or rightx.shape[0] < 7:
        return None

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Return everything we got
    ret = {}
    ret['left_fit'] = left_fit
    ret['right_fit'] = right_fit
    ret['nonzerox'] = nonzerox
    ret['nonzeroy'] = nonzeroy
    ret['left_lane_inds'] = left_lane_inds
    ret['right_lane_inds'] = right_lane_inds

    return ret


def calc_curve(ret):
    """
    Figures out lane curvature.
    :param ret: Dictionary which contains left lane, right lane indices along with polynomials for lanes
    :return: left and right lane curvatures
    """
    left_lane_inds = ret['left_lane_inds']
    right_lane_inds = ret['right_lane_inds']
    nonzerox = ret['nonzerox']
    nonzeroy = ret['nonzeroy']

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    y_eval = 719  # Image is 720x1280

    # Define conversions in x and y from pixels space to meters
    y_meters_per_pixel = 30 / 720  # meters per pixel in y dimension
    x_meters_per_pixel = 3.7 / 700  # meters per pixel in x dimension

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(lefty * y_meters_per_pixel, leftx * x_meters_per_pixel, 2)
    right_fit_cr = np.polyfit(righty * y_meters_per_pixel, rightx * x_meters_per_pixel, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * y_meters_per_pixel + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * left_fit_cr[0])
    right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * y_meters_per_pixel + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * right_fit_cr[0])
    # Now our radius of curvature is in meters

    return left_curverad, right_curverad


def get_vehicle_offset(undist, left_fit, right_fit):
    """
    Calculates the vehicle offset in the lane from center of the lane. Assumes that the car is in center of image
    :param undist: Undistorted image
    :param left_fit: 2nd degree left lane polynomial
    :param right_fit: 2nd degree right lane polynomial
    :return: vehicle offset in meters from center of lane
    """
    x_meters_per_pixel = 3.7 / 700  # meters per pixel in x dimension

    # Calculate vehicle center offset in pixels
    bottom_y = undist.shape[0] - 1
    bottom_x_left = left_fit[0] * (bottom_y ** 2) + left_fit[1] * bottom_y + left_fit[2]
    bottom_x_right = right_fit[0] * (bottom_y ** 2) + right_fit[1] * bottom_y + right_fit[2]
    vehicle_offset = undist.shape[1] / 2 - (bottom_x_left + bottom_x_right) / 2

    # Convert pixel offset to meters
    vehicle_offset *= x_meters_per_pixel

    return vehicle_offset


def visualize(undist, left_fit, right_fit, m_inv, left_curverad, right_curverad, vehicle_offset):
    """
    Draws the lane lines and shades the area between lanes
    :param undist: Undistorted image
    :param left_fit: 2nd degree polynomial for left lane
    :param right_fit: 2nd degree polynomial for right lane
    :param m_inv: Inverse matrix for perspective transform
    :param left_curverad: Radius of curvature of left curve
    :param right_curverad: Raidus of curvature of right curve
    :param vehicle_offset: Vehicle offset from center of lane
    :return: Image with lane lines drawn, area between the lanes shaded, curvature of lanes along with vehicle offset
    annotated on the image
    """
    # Generate x and y values for plotting
    ploty = np.linspace(0, undist.shape[0] - 1, undist.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    # Create an empty image
    color_warp = np.zeros((undist.shape[0], undist.shape[1], undist.shape[2]), dtype='uint8')

    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lanes and fill color
    cv2.fillPoly(color_warp, np.int_([pts]), (180, 127, 255))

    # Warp back the image
    newwarp = cv2.warpPerspective(color_warp, m_inv, (undist.shape[1], undist.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.2, 0)

    # Show the lane curvature and vehicle offset on image
    avg_curve = (left_curverad + right_curverad) / 2
    label_str = 'Radius: %.1f m' % avg_curve
    result = cv2.putText(result, label_str, (330, 40), 0, 1, (0, 0, 0), 2, cv2.LINE_AA)

    label_str = 'Vehicle offset: %.1f m' % vehicle_offset
    result = cv2.putText(result, label_str, (330, 70), 0, 1, (0, 0, 0), 2, cv2.LINE_AA)

    return result


def fit_and_plot(image):
    global left_fit, right_fit, found
    global mtx, dist

    if mtx is None or dist is None:
        ret, mtx, dist, rvecs, tvecs = calibrate_camera()

    udi = cv2.undistort(image, mtx, dist)
    warped_image, M, M_inv = warp(get_combined_binary(udi))

    # if found:
    #     ret = fit(warped_image, left_fit, right_fit)
    #     if ret is None:
    #         found = False
    #     else:
    #         left_fit = ret['left_fit']
    #         right_fit = ret['left_fit']
    #
    # if not found:
    #     ret = slide_and_fit(warped_image)
    #     left_fit = ret['left_fit']
    #     right_fit = ret['left_fit']
    #     found = True

    ret = slide_and_fit(warped_image)
    left_curverad, right_curverad = calc_curve(ret)
    offset = get_vehicle_offset(udi, ret['left_fit'], ret['right_fit'])
    return visualize(udi, ret['left_fit'], ret['right_fit'], M_inv, left_curverad, right_curverad, offset)