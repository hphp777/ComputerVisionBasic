#include <iostream>
#include <opencv2/opencv.hpp>
#include <math.h>       /* exp */
#define IM_TYPE	CV_8UC3


/*
	unsharp masking은 원본 이미지의 intensity에 원
	본이미지에 low pass filter를 적용한 결과를
	빼 주어서 원본 이미지의 intensity variation을 향상시키는 방법이다.
	이를 식으로 표현하면 (I-kL)/(1-k)이다.
	I: 원본 이미지 픽셀의 intensity
	L: 원본 이미지 픽셀의 intensity에 low pass filter를 적용한 결과.
	k: 1보다 작은 파라미터. 0.5이하로 하는것 추천.
*/

using namespace cv;

// Image Type
// "G" for GrayScale Image, "C" for Color Image
#if (IM_TYPE == CV_8UC3)
typedef uchar G;
typedef cv::Vec3b C;
#elif (IM_TYPE == CV_16SC3)
typedef short G;
typedef Vec3s C;
#elif (IM_TYPE == CV_32SC3)
typedef int G;
typedef Vec3i C;
#elif (IM_TYPE == CV_32FC3)
typedef float G;
typedef Vec3f C;
#elif (IM_TYPE == CV_64FC3)
typedef double G;
typedef Vec3d C;
#endif

Mat gaussianfilter(const Mat input, int n, float sigmaT, float sigmaS, const char* opt);
Mat unsharp_masking(Mat input, Mat A_G, float k);

int main() {

	Mat input = imread("lena.jpg", CV_LOAD_IMAGE_COLOR);
	Mat input_gray;
	Mat after_gaussian;
	Mat output;


	cvtColor(input, input_gray, CV_RGB2GRAY);



	if (!input.data)
	{
		std::cout << "Could not open" << std::endl;
		return -1;
	}

	namedWindow("Grayscale", WINDOW_AUTOSIZE);
	imshow("Grayscale", input_gray);
	after_gaussian = gaussianfilter(input_gray, 1, 1, 1, "zero-paddle"); //Boundary process: zero-paddle, mirroring, adjustkernel
	output = unsharp_masking(input_gray, after_gaussian, 0.4);
	namedWindow("Unsharp Masking", WINDOW_AUTOSIZE);
	imshow("Unsharp Masking", output);


	waitKey(0);

	return 0;
}


Mat gaussianfilter(const Mat input, int n, float sigmaT, float sigmaS, const char* opt) {

	Mat kernel;

	int row = input.rows;
	int col = input.cols;
	int kernel_size = (2 * n + 1);
	int tempa;
	int tempb;
	float denom;
	float kernelvalue;
	float* Wt = new float[kernel_size];
	float* Ws = new float[kernel_size];

	// Initialiazing Kernel Matrix 
	// 여기서부터는 커널 벨류가 다 다르다.
	// kernelvalue = w(s,t)
	kernel = Mat::zeros(kernel_size, kernel_size, CV_32F);
	denom = 0.0;

	// 일단 포문을 돌면서 각 좌표에 kernel value를 할당한다.
	// 그다음 누적합을 구한다.
	// 각 좌표별로 그 누적합을 나누어준다.
	// 나누는 경우에는 2번 수행하면 된다.
	// x좌표부터 수행
	// 이를 담을 배열이 필요하다. 
	for (int b = -n; b <= n; b++) {
		float value1 = exp(-(pow(b, 2) / (2 * pow(sigmaT, 2))));
		Wt[n + b] = value1;
		denom += value1;
	}

	for (int b = -n; b <= n; b++) {
		Wt[n + b] /= denom;
	}
	denom = 0.0;
	//그다음 y좌표 수행
	for (int a = -n; a <= n; a++) {
		float value2 = exp(-(pow(a, 2) / (2 * pow(sigmaS, 2))));
		Ws[n + a] = value2;
		denom += value2;
	}

	for (int a = -n; a <= n; a++) {
		Ws[n + a] /= denom;
	}

	Mat output = Mat::zeros(row, col, input.type());


	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {


			if (!strcmp(opt, "zero-paddle")) {
				float sum1 = 0.0;
				for (int a = -n; a <= n; a++) {
					for (int b = -n; b <= n; b++) {

						/* Gaussian filter with Zero-paddle boundary process:

						Fill the code:
						*/
						//그냥 안에것만 그대로 더하면 된다. 어차피 0이므로
						if ((i + a <= row - 1) && (i + a >= 0) && (j + b <= col - 1) && (j + b >= 0)) { //if the pixel is not a border pixel
							sum1 += Ws[n + a] * Wt[n + b] * (float)(input.at<G>(i + a, j + b));
						}
					}
				}
				output.at<G>(i, j) = (G)sum1;
			}

			else if (!strcmp(opt, "mirroring")) {
				float sum1 = 0.0;
				for (int a = -n; a <= n; a++) {
					for (int b = -n; b <= n; b++) {

						/* Gaussian filter with "mirroring" process:

						Fill the code:
						*/
						if (i + a > row - 1) {  //mirroring for the border pixels
							tempa = i - a;
						}
						else if (i + a < 0) {
							tempa = -(i + a);
						}
						else {
							tempa = i + a;
						}
						if (j + b > col - 1) {
							tempb = j - b;
						}
						else if (j + b < 0) {
							tempb = -(j + b);
						}
						else {
							tempb = j + b;
						}
						sum1 += Ws[n + a] * Wt[n + b] * (float)(input.at<G>(tempa, tempb));
					}
				}
				output.at<G>(i, j) = (G)sum1;
			}


			else if (!strcmp(opt, "adjustkernel")) {
				float sum1 = 0.0;
				float sum2 = 0.0;
				for (int a = -n; a <= n; a++) {
					for (int b = -n; b <= n; b++) {

						/* Gaussian filter with "adjustkernel" process:

						Fill the code:
						*/
						if ((i + a <= row - 1) && (i + a >= 0) && (j + b <= col - 1) && (j + b >= 0)) {
							sum1 += Ws[n + a] * Wt[n + b] * (float)(input.at<G>(i + a, j + b));
							sum2 += Ws[n + a] * Wt[n + b];
						}
					}
				}
				output.at<G>(i, j) = (G)(sum1 / sum2);
			}
		}
	}
	free(Wt);
	free(Ws);

	return output;
}

Mat unsharp_masking(Mat input, Mat A_G, float k) {
	Mat result = input.clone(); //initialization

	int row = input.rows;
	int col = input.cols;

	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			result.at<G>(i, j) = (input.at<G>(i, j) - k * A_G.at<G>(i, j)) / (1-k);
		}
	}
	return result;
}