#include <iostream>
#include <opencv2/opencv.hpp>

#define IM_TYPE	CV_8UC3

using namespace cv;


// Note that this code is for the case when an input data is a color value.
int main() {

	Mat input = imread("lena.jpg", CV_LOAD_IMAGE_COLOR);
	Mat output;
	Mat input_gray; //convert RGB to Grayscale

	cvtColor(input, input_gray, CV_RGB2GRAY);
	
	if (!input.data)
	{
		std::cout << "Could not open" << std::endl;
		return -1;
	}

	namedWindow("Original", WINDOW_AUTOSIZE);
	imshow("Original", input);


	Mat samples(input.rows * input.cols, 3, CV_32F);

	for (int y = 0; y < input.rows; y++)
		for (int x = 0; x < input.cols; x++)
			for (int z = 0; z < 3; z++)
				samples.at<float>(y + x*input.rows, z) = input.at<Vec3b>(y, x)[z];

	// Clustering is performed for each channel (RGB)
	// Note that the intensity value is not normalized here (0~1). You should normalize both intensity and position when using them simultaneously.
	int clusterCount = 10;
	Mat labels;
	int attempts = 5;
	Mat centers;

	//왜 RGB이미지에 대해서만 clustering을 수행하지?
	kmeans(samples, clusterCount, labels, TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 10000, 0.0001), attempts, KMEANS_PP_CENTERS, centers);
	//centers에는 각 cluster의 대표 intensity가 저장되어있다.
	//cluster의 index를 사용해서 접근해야 한다.
	
	Mat new_image(input_gray.size(), input_gray.type());
	Mat new_image2(input.size(), input.type());


	for (int y = 0; y < input.rows; y++)
		for (int x = 0; x < input.cols; x++)
		{
			//(y,x)의 pixel이 속한 군집번호를 반환한다.
			//BGR의 B값을 가지고 Gray이미지의 Clustering을 수행?
			int cluster_idx = labels.at<int>(y + x*input.rows, 0);
			new_image.at<uchar>(y, x) = centers.at<float>(cluster_idx, 0);
			//Fill code that finds for each pixel of each channel of the output image the intensity of the cluster center.
			for (int z = 0; z < 3; z++)
				new_image2.at<Vec3b>(y, x)[z] = centers.at<float>(cluster_idx,z);
		}
	
	
	imshow("clustered image(Gray)", new_image);
	imshow("clustered image(RGB)", new_image2);

	waitKey(0);

	return 0;
}

