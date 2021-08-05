- Purpose of this code
 This code is made to inhance the quality of image through the histogram matching. 

- How to run this code
 Through the visual studio, run "hist_matching_gs.cpp" with the input image.

- How to adjust parameters (if any)
 void hist_match(Mat& input, Mat& equalized, G* trans_func, float* CDF);
 Mat& input: Give input Mat
 Mat& equalized: Give Mat variable to save matched image. This is intermediate result.
 G* trans_func: Give array to save transformation function.
 float* CDF: Give CDF.

 void hist_match_final(Mat& input, Mat& equalized, G* trans_func, float* CDF); 
 Mat& input: Give input Mat
 Mat& equalized: Give Mat variable to save matched image. This is the final result
 G* trans_func: Give array to save transformation function.
 float* CDF:  Give CDF.

 Mat drawHist(Mat img)
 Mat img: Give Mat which will be drawn.

- How to define default parameters
 void hist_match(Mat& input, Mat& equalized, G* trans_func, float* CDF);
 Mat& input: Give input Mat
 Mat& equalized: Give Mat variable to save matched image. This is intermediate result.
 G* trans_func: Give array to save transformation function.
 float* CDF: Give CDF.

 void hist_match_final(Mat& input, Mat& equalized, G* trans_func, float* CDF); 
 Mat& input: Give input Mat
 Mat& equalized: Give Mat variable to save matched image. This is the final result
 G* trans_func: Give array to save transformation function.
 float* CDF:  Give CDF.

 Mat drawHist(Mat img)
 Mat img: Give Mat which will be drawn.