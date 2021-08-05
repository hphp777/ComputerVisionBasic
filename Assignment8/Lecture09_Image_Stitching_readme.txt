- Purpose of this code
 This code is made to stitch the two images. We will find corresponding points using SIFT.
 
- How to run this code
 Execute "Lecture09_Image_Stitching.cpp" in the Visual Studio with two input images.

 * The same functions with the previous lectures are omitted.
- How to adjust parameters (if any)
 void stitching_RANSAC(int k, float threshold, int loop);
 int k: Give the number of how many pixel you will randomly choose.
 float threshold: Give the threshold value through which we can decern the inliers.
 int loop: Give the number how many times this procedure should be iterated. 

- How to define default parameters

 void stitching_RANSAC(int k, float threshold, int loop);
 int k: Give the number of how many pixel you will randomly choose. In this code, I gave 3, 4.
 float threshold: Give the threshold value through which we can decern the inliers. In this code, I gave 0.5.
 int loop: Give the number how many times this procedure should be iterated.  In this code, I gave 50.
