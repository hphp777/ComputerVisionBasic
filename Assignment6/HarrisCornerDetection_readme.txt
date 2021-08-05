- Purpose of this code
 This code is made to detect corner points in the input image.

- How to run this code
 Load 'HarrisCornerDetection.cpp' with an input "lena.jpg"image. 

- How to adjust parameters (if any)
vector<Point2f> MatToVec(const Mat input);
 const Mat input: Give input Mat. Iterating each pixel, if the value is 1, then push this coordiante into the vector.

Mat NonMaximum_Suppression(const Mat input, Mat corner_mat, int radius);
 const Mat input: Give Mat on which non-maximum-suppression will be applied.
 Mat corner_mat: Give Mat on which corner points were represented. Through this function, the number of corner points in this variable will be decrease.
 int radius : Give the window size to compare intensity reference pixel with its neighboring pixels within window.

Mat Mirroring(const Mat input, int n);
 const Mat input: Give input Mat which will be mirrored.
 int n: Give the amount (how much the input image will be mirrored)

- How to define default parameters
vector<Point2f> MatToVec(const Mat input);
 const Mat input: Give input Mat. Iterating each pixel, if the value is 1, then push this coordiante into the vector.

Mat NonMaximum_Suppression(const Mat input, Mat corner_mat, int radius);
 const Mat input: Give Mat on which non-maximum-suppression will be applied.
 Mat corner_mat: Give Mat on which corner points were represented. Through this function, the number of corner points in this variable will be decrease.
 int radius : Give the window size to compare intensity reference pixel with its neighboring pixels within window. In this code, I gave 2.

Mat Mirroring(const Mat input, int n);
 const Mat input: Give input Mat which will be mirrored.
 int n: Give the amount (how much the input image will be mirrored).  In this code, I gave 2.