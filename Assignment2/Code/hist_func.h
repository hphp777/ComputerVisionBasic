#pragma once

#include <opencv2/opencv.hpp>
#include <stdio.h>

#define L 256		// # of intensity levels
#define IM_TYPE	CV_8UC3

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

// generate PDF for single channel image
float *cal_PDF(Mat &input) {

	int count[L] = { 0 };
	float *PDF = (float*)calloc(L, sizeof(float));

	// Count
	for (int i = 0; i < input.rows; i++)
		for (int j = 0; j < input.cols; j++)
			count[input.at<G>(i, j)]++;

	// Compute PDF
	for (int i = 0; i < L; i++)
		PDF[i] = (float)count[i] / (float)(input.rows * input.cols);

	return PDF;
}

// generate PDF for color image
float **cal_PDF_RGB(Mat &input) {

	int count[L][3] = { 0 };
	float **PDF = (float**)malloc(sizeof(float*) * L);

	for (int i = 0; i < L; i++)
		PDF[i] = (float*)calloc(3, sizeof(float));

	////////////////////////////////////////////////
	//											  //
	// How to access multi channel matrix element //
	//											  //
	// if matrix A is CV_8UC3 type,				  //
	// A(i, j, k) -> A.at<Vec3b>(i, j)[k]		  //
	//											  //
	////////////////////////////////////////////////

	// Count
	for (int i = 0; i < input.rows; i++)
		for (int j = 0; j < input.cols; j++) {
			count[input.at<Vec3b>(i, j)[0]][0]++; //B
			count[input.at<Vec3b>(i, j)[1]][1]++; //G
			count[input.at<Vec3b>(i, j)[2]][2]++; //R
		}
			

	// Compute PDF
	for (int k = 0; k < 3; k++) {
		for (int i = 0; i < L; i++) {
			PDF[i][k] = (float)count[i][k] / (float)(input.rows * input.cols);
		}
	}
	
	return PDF;
}

// generate CDF for single channel image
float *cal_CDF(Mat &input) {

	int count[L] = { 0 };
	float *CDF = (float*)calloc(L, sizeof(float));

	// Count
	for (int i = 0; i < input.rows; i++)
		for (int j = 0; j < input.cols; j++)
			count[input.at<G>(i, j)]++;

	// Compute CDF
	for (int i = 0; i < L; i++) {
		CDF[i] = (float)count[i] / (float)(input.rows * input.cols);

		if (i != 0)
			CDF[i] += CDF[i - 1];
	}

	return CDF;
}

// generate CDF for color image
float **cal_CDF_RGB(Mat &input) {

	int count[L][3] = { 0 };
	float **CDF = (float**)malloc(sizeof(float*) * L);

	for (int i = 0; i < L; i++)
		CDF[i] = (float*)calloc(3, sizeof(float));

	////////////////////////////////////////////////
	//											  //
	// How to access multi channel matrix element //
	//											  //
	// if matrix A is CV_8UC3 type,				  //
	// A(i, j, k) -> A.at<Vec3b>(i, j)[k]		  //
	//											  //
	////////////////////////////////////////////////

	// Count
	for (int i = 0; i < input.rows; i++)
		for (int j = 0; j < input.cols; j++) {
			count[input.at<Vec3b>(i, j)[0]][0]++; //B
			count[input.at<Vec3b>(i, j)[1]][1]++; //G
			count[input.at<Vec3b>(i, j)[2]][2]++; //R
		}

	// Compute CDF
	for (int k = 0; k < 3; k++) {
		for (int i = 0; i < L; i++) {
			CDF[i][k] = (float)count[i][k] / (float)(input.rows * input.cols);

			if (i != 0)
				CDF[i][k] += CDF[i - 1][k]; //Accumulate
		}
	}

	return CDF;
}

//Drawing plots (PDF, CDF)
Mat drawHist_gray_PDF(Mat img) {
	MatND histogram;
	const int* channel_numbers = { 0 };
	float channel_range[] = { 0.0, 255.0 };
	const float* channel_ranges = channel_range;
	int number_bins = 255;
	calcHist(&img, 1, channel_numbers, Mat(), histogram, 1, &number_bins, &channel_ranges, true, false);

	//Plot the histogram
	int hist_w = 512;
	int hist_h = 400;
	int bin_w = cvRound((double)hist_w / number_bins);

	Mat histImg(hist_h, hist_w, CV_8UC1, Scalar(0, 0, 0));
	normalize(histogram, histogram, 0, histImg.rows, NORM_MINMAX, -1, Mat());

	for (int i = 1; i < number_bins; i++) {
		line(histImg, Point(bin_w * (i - 1), hist_h - cvRound(histogram.at<float>(i - 1))), Point(bin_w * (i), hist_h - cvRound(histogram.at<float>(i))), Scalar(255, 0, 0), 2, 8, 0);
	}

	return histImg;
}

Mat drawHist_gray_CDF(Mat img) {
	MatND histogram;
	const int* channel_numbers = { 0 };
	float channel_range[] = { 0.0, 255.0 };
	const float* channel_ranges = channel_range;
	int number_bins = 255;
	bool uniform = true;
	bool accumulate = true;
	calcHist(&img, 1, channel_numbers, Mat(), histogram, 1, &number_bins, &channel_ranges);
	
	//Accumulate Histogram
	Mat accumulatedHist = histogram.clone();

	for (int i = 1; i < number_bins; i++) {
		accumulatedHist.at<float>(i) += accumulatedHist.at<float>(i - 1);
	}

	//Plot the histogram
	int hist_w = 512;
	int hist_h = 400;
	int bin_w = cvRound((double)hist_w / number_bins);

	Mat histImg(hist_h, hist_w, CV_8UC1, Scalar(0, 0, 0));
	normalize(accumulatedHist, accumulatedHist, 0, histImg.rows, NORM_MINMAX, -1, Mat());

	for (int i = 1; i < number_bins; i++) {
		line(histImg, Point(bin_w * (i - 1), hist_h - cvRound(accumulatedHist.at<float>(i - 1))), Point(bin_w * (i), hist_h - cvRound(accumulatedHist.at<float>(i))), Scalar(255, 0, 0), 2, 8, 0);
	}

	return histImg;
}

Mat* drawHist_RGB_PDF(Mat img) {
	MatND histogramB, histogramR, histogramG;
	const int channel_numberR[] = { 2 };
	const int channel_numberG[] = { 1 };
	const int channel_numberB[] = { 0 };
	float channel_range[] = { 0.0, 255.0 };
	const float* channel_ranges = channel_range;
	int number_bins = 255;

	//R,G,B별로 히스토그램 계산
	calcHist(&img, 1, channel_numberB, Mat(), histogramB, 1, &number_bins, &channel_ranges);
	calcHist(&img, 1, channel_numberR, Mat(), histogramR, 1, &number_bins, &channel_ranges);
	calcHist(&img, 1, channel_numberG, Mat(), histogramG, 1, &number_bins, &channel_ranges);

	//Plot the histogram
	int hist_w = 512;
	int hist_h = 400;
	int bin_w = cvRound((double)hist_w / number_bins);

	Mat histImgB(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));
	normalize(histogramB, histogramB, 0, histImgB.rows, NORM_MINMAX, -1, Mat());

	Mat histImgG(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));
	normalize(histogramG, histogramG, 0, histImgG.rows, NORM_MINMAX, -1, Mat());

	Mat histImgR(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));
	normalize(histogramR, histogramR, 0, histImgR.rows, NORM_MINMAX, -1, Mat());

	for (int i = 1; i < number_bins; i++) {
		line(histImgB, Point(bin_w * (i - 1), hist_h - cvRound(histogramB.at<float>(i - 1))),
			Point(bin_w * (i), hist_h - cvRound(histogramB.at<float>(i))), Scalar(255, 0, 0), 2, 8, 0);
		line(histImgG, Point(bin_w * (i - 1), hist_h - cvRound(histogramG.at<float>(i - 1))),
			Point(bin_w * (i), hist_h - cvRound(histogramG.at<float>(i))), Scalar(0, 255, 0), 2, 8, 0);
		line(histImgR, Point(bin_w * (i - 1), hist_h - cvRound(histogramR.at<float>(i - 1))),
			Point(bin_w * (i), hist_h - cvRound(histogramR.at<float>(i))), Scalar(0, 0, 255), 2, 8, 0);
	}

	//동적할당으로 공간을 생성한 뒤에 하나씩 값을 대입해주어야 한다.
	Mat* hist = new Mat[3];
	hist[0] = histImgB;
	hist[1] = histImgG;
	hist[2] = histImgR;

	return hist;
}

Mat* drawHist_RGB_CDF(Mat img) {

	MatND histogramB, histogramR, histogramG;
	const int channel_numberR[] = { 2 };
	const int channel_numberG[] = { 1 };
	const int channel_numberB[] = { 0 };
	float channel_range[] = { 0.0, 255.0 };
	const float* channel_ranges = channel_range;
	int number_bins = 255;

	//R,G,B별로 히스토그램 계산
	calcHist(&img, 1, channel_numberB, Mat(), histogramB, 1, &number_bins, &channel_ranges, true, true);
	calcHist(&img, 1, channel_numberR, Mat(), histogramR, 1, &number_bins, &channel_ranges, true, true);
	calcHist(&img, 1, channel_numberG, Mat(), histogramG, 1, &number_bins, &channel_ranges, true, true);

	//Accumulate Histogram
	Mat accumulatedHistB = histogramB.clone();
	for (int i = 1; i < number_bins; i++) {
		accumulatedHistB.at<float>(i) += accumulatedHistB.at<float>(i - 1);
	}
	Mat accumulatedHistG = histogramG.clone();
	for (int i = 1; i < number_bins; i++) {
		accumulatedHistG.at<float>(i) += accumulatedHistG.at<float>(i - 1);
	}
	Mat accumulatedHistR = histogramR.clone();
	for (int i = 1; i < number_bins; i++) {
		accumulatedHistR.at<float>(i) += accumulatedHistR.at<float>(i - 1);
	}

	printf("%f\n", accumulatedHistB.at<float>(254));
	printf("%f\n", accumulatedHistG.at<float>(254));
	printf("%f\n", accumulatedHistR.at<float>(254));

	//Plot the histogram
	int hist_w = 512;
	int hist_h = 400;
	int bin_w = cvRound((double)hist_w / number_bins);

	Mat histImgB(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));
	normalize(accumulatedHistB, accumulatedHistB, 0, histImgB.rows, NORM_MINMAX, -1, Mat());

	Mat histImgG(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));
	normalize(accumulatedHistG, accumulatedHistG, 0, histImgG.rows, NORM_MINMAX, -1, Mat());

	Mat histImgR(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));
	normalize(accumulatedHistR, accumulatedHistR, 0, histImgR.rows, NORM_MINMAX, -1, Mat());

	for (int i = 1; i < number_bins; i++) {
		line(accumulatedHistB, Point(bin_w * (i - 1), hist_h - cvRound(accumulatedHistB.at<float>(i - 1))),
			Point(bin_w * (i), hist_h - cvRound(accumulatedHistB.at<float>(i))), Scalar(255, 0, 0), 2, 8, 0);
		line(accumulatedHistG, Point(bin_w * (i - 1), hist_h - cvRound(accumulatedHistG.at<float>(i - 1))),
			Point(bin_w * (i), hist_h - cvRound(accumulatedHistG.at<float>(i))), Scalar(0, 255, 0), 2, 8, 0);
		line(accumulatedHistR, Point(bin_w * (i - 1), hist_h - cvRound(accumulatedHistR.at<float>(i - 1))),
			Point(bin_w * (i), hist_h - cvRound(accumulatedHistR.at<float>(i))), Scalar(0, 0, 255), 2, 8, 0);
	}

	//동적할당으로 공간을 생성한 뒤에 하나씩 값을 대입해주어야 한다.
	Mat* hist = new Mat[3];
	hist[0] = histImgB;
	hist[1] = histImgG;
	hist[2] = histImgR;

	return hist;
}