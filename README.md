## Advanced Lane Finding
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)


In this project, I was able to write a software algorithm pipeline to identify the lane boundaries in a video. For detailed project description, Check out the [writeup](https://github.com/MyadaRoshdi/P4/blob/master/writeup.md). 


The Project
---

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

The images for camera calibration are stored in the folder called `camera_cal`.  The images in `test_images` I used for testing my pipeline on single frames.  If you want to extract more test images from the videos, you can simply use an image writing method like `cv2.imwrite()`, i.e., you can read the video in frame by frame as usual, and for frames you want to save for later you can write to an image file.  

I saved examples of the output from each stage of my pipeline in the folder called `ouput_images`, and with description of what each image shows in my writeup.   The video called `project_video.mp4` is the video I tested my  pipeline on and it worked well.  

The `challenge_video.mp4` video is an extra (and optional) challenge for you if you want to test your pipeline under somewhat trickier conditions.  The `harder_challenge.mp4` video is another optional challenge and is brutal!

If you're feeling ambitious (again, totally optional though), don't stop there!  We encourage you to go out and take video of your own, calibrate your camera and show us how you would implement this project from scratch!

## **My project submission includes the following files/folders:** 
* code.ipynb : this is the notebook I used in implementing and testing my pipeline.
* code.py : script version of code.ipynb.
* output_images: this folder contains my saved examples of the output from each stage of pipeline.
* writeup.md: a report writeup file as markdown.
* output_videos: this folder contains the result of applying my algorithm on supported videos.
* camera_cal: this folder contains all images used in camera caliberation and a saved pickle file contains calibration results for re-useability . 
* README.md: This is the current opened file, it contains breif description of the project submission and dependencies issues.



### Dependencies
This project requires:

* [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)

The project enviroment can be created with CarND Term1 Starter Kit. Click [here](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) for the details.


# P4
