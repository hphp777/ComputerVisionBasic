- Purpose of this code
 This code is for increasing image contrast and so that we can make intensities of input image distributed more even. 

- How to run this code
 Through visual studio. Run hist_qe.cpp with 'input.jpg' image.

- How to adjust parameters (if any)
 void hist_eq(Mat &input, Mat &equalized, G *trans_func, float *CDF);
 Mat &input: Give Mat variable which would be equalized(gray image).
 Mat &equalized: Give Mat variable to save equalized image(gray image).
 G *trans_func: Give array (Gray scale type) of which size is 256(the number of levels of intensity). It is isnitialized with 0s.
 float *CDF: To execute histogram equalization, give CDF of input image which is calculated by cal_PDF function.


- How to define default parameters
 void hist_eq(Mat &input, Mat &equalized, G *trans_func, float *CDF);
 Mat &input: Give Mat variable which would be equalized(gray image).
 Mat &equalized: Give Mat variable to save equalized image(gray image).
 G *trans_func: Give array (Gray scale type) of which size is 256(the number of levels of intensity). It is isnitialized with 0s.
 float *CDF: To execute histogram equalization, give CDF of input image which is calculated by cal_PDF function.
