- Purpose of this code
This code is for stretching input intensity of original image to the more broad output intensity. 
The range which will be stretched is defined manually.  

- How to run this code
Through the visual studio, load the PDF_CDF.cpp and run.

- How to adjust parameters (if any)
 void linear_stretching(Mat &input, Mat &stretched, G *trans_func, G x1, G x2, G y1, G y2);
 Mat &input: Give input Mat of which intensity will be stretched.
 Mat &stretched: Give Mat variable to save the image of which intensities are stretched. (Same size with input image) 
 G *trans_func: Give array which transfer intensities. The size of this is 256. This is filled in linear_stretching function.
 G x1, x2: The range of input intensity which will be stretched to the output intensity. These variables are defined manually.
 G y1, y2: The range of output intensity. These variables are defined manually.
 [x1,x2] -> [y1,y2]

- How to define default parameters
 void linear_stretching(Mat &input, Mat &stretched, G *trans_func, G x1, G x2, G y1, G y2);
 Mat &input: Give input Mat(result of the cvtColor(input, input_gray, CV_RGB2GRAY) function).
 Mat &stretched: Give Mat variable to save the image of which intensities are stretched. (Same size with input image). This Mat is a clone of converted gray image.
 G *trans_func: Give array which transfer intensities. The size of this is 256. It is initialized with 0.
 G x1, x2: The range of input intensity which will be stretched to the output intensity. These variables are defined manually.
 G y1, y2: The range of output intensity. These variables are defined manually.