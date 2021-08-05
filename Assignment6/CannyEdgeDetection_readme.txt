- Purpose of this code
 This code is made to detect edges in the input image.

- How to run this code
 Load 'CannyEdgeDetection.cpp' with an input "lena.jpg"image. 

- How to adjust parameters (if any)
Cv::Canny(src, dst, low, high, sobel_kernel_size, precision)
Src: input image Mat
Dst: output image Mat
Low: low threshold
High: high threshold
Sobel_kernel_size: kernel size of the sobel filter
Precision: whether this function will be operated precisely or not. (True or False)(If this value is False, the number of edge would be decrease)

- How to define default parameters
 Canny(input, output, 100, 127, 3, false)