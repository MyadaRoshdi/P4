## Writeup 

### This file is my writeup report that covers my work for the [Advanced lane finding Project](https://github.com/MyadaRoshdi/P4).
 to get my project full package, download it locally:
  - git clone https://github.com/MyadaRoshdi/P4

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.
* Test on supported videos

[//]: # (Image References)

[image0]: ./test_images/straight_lines1.jpg "example"
[image1]: ./output_images/before-after-calibration.png "Undistorted"
[image2]: ./output_images/undist_transformed.png "Road Transformed"
[image3]: ./output_images/threshold_combined_with_grad-Y.png "gradient_combined"
[image4]: ./output_images/threshold_combined_without_grad-Y.png "without_gradient-Y"
[image5]: ./output_images/test_images_pipeline_Gradient_S.png "gradient+S"
[image6]: ./output_images/test_images_pipeline_Gradient_R_L.png "gradient+L+R"
[image7]: ./output_images/test_mages_pipeline_R_L_B.png "L+R+B"
[image8]: ./output_images/test_images_pipeline_R_L.png "L+R"
[image9]: ./output_images/test_images_pipeline_L_B_Best_results.png "Best+L+B"
[image10]: ./output_images/test2_histogram.png "good_histogram"
[image11]: ./output_images/good_histogram_result_test2.png "good_histogram_result"
[image12]: ./output_images/test1_histogram.png "bad_histogram"
[image13]: ./output_images/bad_histogram_result_test1.png "bad_histogram_result"
[image14]: ./test_images/test2.jpg "test2_image"
[image15]: ./output_images/test1_histogram_of_bottom_3rd.png "better_histogram_after_bottm_third"
[image16]: ./output_images/test_image_lane_lines_fit.png "bad_laneline_draw_test_images"
[image17]: ./output_images/best_result_lane_lines_fit.png "best_lanelines_draw_result_test_images"
[image18]: ./output_images/warped_cropped.png "cropped"
[image19]: ./output_images/draw_measurments.png "final_test_result"
[image20]: ./output_images/un_warped_binary.png "Unwarp"
[video1]: ./output_videos/project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.    

You're reading my [writeup](https://github.com/MyadaRoshdi/P4/blob/master/writeup.md) now!. And you can check my README [here](https://github.com/MyadaRoshdi/P4/blob/master/README.md).



### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the 3rd, 4th and 5th code cells of the IPython notebook located in "code.ipynb" (or in lines #40 through #111 of the file called `code.py`).  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![test_image][image0]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.


Here, I will describe in details the steps I followed to get the best combined techniques I used to get the best lane detection binary image:

**a) Undistort and warp the image (Perspective transform)**
Here  I implemented the function (`corners_unwarp()`), i which I applied both distortion correction and warp the image to get bird's view of the image. 

The code for this step is contained in the 6th code cells of the IPython notebook located in "code.ipynb" (or in lines #132 through #168 of the file called `code.py`). 
Here's an example of my output for this step. 

![undistorted_img][image2]


**b) Apply combind thresholds to our un-distorted and warped image, and get a combined binary** 
Gradient threshold + Gradient magnitude threshold + Gradient direction threshold
Here  I implemented the functions (`abs_sobel_thresh(), mag_thresh(), dir_threshold()` ), to apply Gradient threshold + Gradient magnitude threshold + Gradient direction threshold and then combined to get a combined thresholds.

**NOTE** the thresholds values are imperically chosen.

The code for this step is contained in the 8th & 9th code cells of the IPython notebook located in "code.ipynb" (or in lines #209 through #289 of the file called `code.py`). 
Here's an example of my output for this step. 

![combined_Gradient][image3]

Then, I noticed that when removing Y-Gradent threshold, i get a better result as shown below:
![better_without_gradY][image4]

**c) Experimenting different Gradient & Color thresholds to get the best combined binary** 

Here I started experimenting different combinations of color and gradient thresholds to generate a binary image.  I implemented the color thresholds in  (`s_thresh(), r_thresh(), l_thresh() and b_thresh()` ), to apply s_channel threshold , r_channel threshold, l_channel threshold and b_channel threshold
**NOTE** the thresholds values are imperically chosen.

Experiments Summarization:
1) Apply both Gradients + S-threshold
![G+S][image5]
  
  
  
2) Apply both Gradient + R-threshold + L-threshold
![G+S+L][image6]


   
3) Apply only R-threshold + L-threshold
![R+L][image8]



   
4) Apply  R-threshol + L-threshold + B-threshold
![R+L+B][image7]
   
   
5) Apply L-threshold + B-threshold
![L+B][image9]

 ***Best Combination*** It is noticed as shown above, the best combination is just apply **L-threshold + B-threshold**. Since, L-channel filters the effect of brightness or darkness in the image, while B-channel considered best grab the yellow color, since I had problem with yellow color detection as shown in the previous experiments. 
 ![L+B][image9]

**NOTE** A good discution about color extraction can be found [here](https://communities.theiet.org/discussions/viewtopic/348/19442).

The code for all my experiment step is contained from  the 11th to the  26th code cells of the IPython notebook located in "code.ipynb" (or in lines #393 through #799 of the file called `code.py`). 



#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform **(look at section 2.a) above )** includes a function called `corners_unwarp()`, which appears in lines 132 through 168 in the file `code.py`  (or,  in the 6th code cell of the IPython notebook).  The `corners_unwarp()` function takes as inputs an image (`img`), number of x and y points, camera matrix and distortion coefficients (returned values from camera calibration step. then I set source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
# For source points I'm just using  the outer four detected corners
src = np.float32([(575,464),
                   (707,464), 
                   (258,682), 
                   (1049,682)])
                   
dst = np.float32([(offset,0),
                  (img.shape[1]-offset,0),
                  (offset,img.shape[0]),
                  (img.shape[1]-offset,img.shape[0])])                   

   
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 575, 464      | 450, 0        | 
| 707, 464      | 830, 0      |
| 258, 682      | 450, 720      |
| 1049, 682     | 830, 720        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![warped][image2]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did the following step and fit my lane lines with a 2nd order polynomial, here is a detailed description:

**a) Line Finding Method: Peaks in a Histogram:**
I applyied histogram to get the peaks in the bottom half of the image, which considered as a good start point to detect the lane     lines, As shown below, this couldn't work with all images, and some peaks gives false detection, so I modified it to just get the histogram peaks of the bottom third of the image instead of the bottom half,and it worked for all testing images as shown in the next steps images.

Good Histogram : 
![Good_histogram ][image10]

Misleading Histogram (detects the far right lane instead)
![bad_histogram ][image12]

Gettng better Histogram of above image after using bottom 3rd
![bad_histogram ][image15]

**b) Cropping Image**
Since I found that ther are some misleading information on the right of the image, I decided to crop a small part of the right of the image, as shown in the following function:
   `img = bin_img[0:720, 0:1100] #slicing arrays, giving the [startY:endY,startX:endX] coordinates to the slice.`  
      
![cropped warped image ][image18]

**c) Implement Sliding Windows and Fit a Polynomial**
Here, I exactly used the code suggested in the lesson, I just did some slight modification. I implemented that in the function `lane_lines()` that in lines #805 through #894 in my code in `code.py`, also could be found under the Step5-section in IPython notebook located in "code.ipynb"

**d) Skip the sliding windows step once you know where the lines are**
Here again , I exactly used the code suggested in the lesson, I just did some slight modification. I implemented that in the function `lane_lines_fit()` that in lines #947 through #972 in my code in `code.py`, also could be found under the Step5-section in IPython notebook located in "code.ipynb"

**e) Visualize**
Also , I exactly used the code suggested in the lesson, I just did some slight modification. I implemented that in the function `visualize_lane_lines_fit()` that in lines #980 through #1014 in my code in `code.py`, also could be found under the Step5-section in IPython notebook located in "code.ipynb"

**f) testing on all test_images**
testing the full algorithm on all images supported in test_images are shown bellow:
**Note:** the first image shows the algorithm with bad histogram and without cropping and the 2nd shows the final best results,

![Bad histogram and no cropping][image16]
![best lane lines output ][image17]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I implemented that in the function `lane_curvature_vehicle_position()`,  this function is used to measure the curvature radius for left and right lanes, and car position w.r.t. center. This function can be found in lines #1105 through #1129 in my code in `code.py`, also could be found under the Step6-section in IPython notebook located in "code.ipynb". the follwoing section shows an images showing this function output on it.


#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented that in the function `unwarp()`, all what this function does is unwarping a  warped images. This function can be found in lines #1158 through #1164 in my code in `code.py`, also could be found under the Step7-section in IPython notebook located in "code.ipynb". Here's an example image of un-warping a warped image

![best lane lines output ][image20]

Then, I implemented that in the function `draw_lines()`, which draws a trapezoid from the previously detetected polynomials of left and right lines, on the orignal image. This function can be found in lines #1187 through #1210 in my code in `code.py`, also could be found under the Step7-section in IPython notebook located in "code.ipynb". Here's an example of drawing the trapezoid, and the measurments on the original image

![final test output ][image19]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk breifly about the approach I took and described in all above sections:

**problems / issues you faced in your implementation of this project**
The problem I faced in the 1st part of image processing were with the parts with lighter areas and shadows, together with yellow-lines, so as shown above i used L-channel filtering and it worked much much better with the light areas and shadows, and gave a great results compared to S-channel and R-channel thresholding, but then i faced problem that yellow lines disappeared, so I searched for better color extraction, and I found a guide [here](https://communities.theiet.org/discussions/viewtopic/348/19442) which took me to the LAB-color space and doing B-channel thresholding, I experimented different thresholds till I get the best results as shown above.
In the begining I used the suggested Gradient-thresholds, then I discovered that all it does is catching lots of noises in my images, so when I excluded this step, everything worked better.

Another problem I faced, is when I used the bottom half histogram peaks to detect the lane lines, in some images it gave me false detection of the right line, as it detects the far right lane instead, so I modified it to use the bottom third of the image instead, and in worked so well.
Then after I did a slight cropping to the unwanted parts of the right of the image, as when I tested my pipeline on the supported project_video, in some parte it still detects the far right lane instead, after applying cropping, it worked so good.


**where the pipeline might fail**
My pipeline might fail in the following conditions:
1* if my car is in the right lane of the road instead of the left lane
2* If there is snow or the lane is already for any other reason it turned to white or yellow close to the lane-lines colors.
3* Very sharp edges

**and how I might improve it if I were going to pursue this project further**  
To solve the points I discussed in the previous quesion (**where the pipeline might fail**), I suggest the following:

1* Removing more un wanted information from the image, as I used right side cropping, we can also remove some from left side and the top of the image.

2* Not just using a center camera image, I would get another right and left cameras and do a kind of combining results from the 3-images to detect the lane, so if the front camera images shows a full white lane, the left and right cameras can be used to detect the lane lines.

3* getting better smooting lane-lines detecting Algorthm, since I already used the suggested Algorithm in the lesson, I didn't assign more time to this part, but I can see that more smoothing techniques may help in sharp edges


