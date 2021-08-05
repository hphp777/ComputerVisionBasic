#include "hist_func.h"

/*
	grey scale image를 Histogram Matching 방법으로 
	보정하는 방법이다. 
	같은 풍경을 촬영한 여러개의 이미지중 
	contranst가 가장 훌륭한 이미지를 기준으로 두고
	나머지 이미지들을 최대한 그 이미지와 비슷하도록 보정하는 작업이다.
	이 과정을 수행하기 위해 이번에는 transformation function이 두개가 필요하다.
	하나는 변환할 이미지, 그리고 하나는 잘 나온 이미지의 HE를 수행하기 위한 것이다.
	인풋 이미지에 대해 T, G-1을 두번씩 적용해서 결과를 출력한다.
	앞의 HE를 2번 적용하는것이지만 이 과정에서의 차별점은 trasformation함수의 
	역함수를 구하는 과정을 거쳐야 한다는 점이다.
	역함수는 1:1대응이 아니면 구할수 없는 것이 원칙인데, 주어진 
	transformation function은 그렇지 않은 경우가 일반적이다.
	따라서 역함수를 구한 후에 이를 1:1대응으로 다시 조정해주는 과정이 필요하다.
*/

void hist_match(Mat& input, Mat& equalized, G* trans_func, float* CDF);
void hist_match_final(Mat& input, Mat& equalized, G* trans_func, float* CDF);
Mat drawHist(Mat img);

int main() {
	

	//input, output 이미지를 저장할 Matrix를 생성한다.
	Mat input=imread("blurred_input.jpg", CV_LOAD_IMAGE_COLOR); //여기에 바꾸고자 하는 이미지를 넣는다.
	Mat input_grey;
	Mat refer = imread("input.jpg", CV_LOAD_IMAGE_COLOR); //참고할 이미지
	Mat refer_grey;
	

	cvtColor(input, input_grey, CV_RGB2GRAY);
	cvtColor(refer, refer_grey, CV_RGB2GRAY);

	Mat output1 = input_grey.clone();
	Mat output2 = refer_grey.clone();
	
	// PDF or transfer function txt files
	FILE* f_PDF; //PDF of original Image
	FILE* f_equalized_PDF; //PDF of output image

	FILE* f_trans_func_input;
	FILE* f_trans_func_refer;
	FILE* f_trans_func_result;

	float *PDF_input = cal_PDF(input_grey);
	float *PDF_refer = cal_PDF(refer_grey);
	float *CDF_input = cal_CDF(input_grey);
	float *CDF_refer = cal_CDF(refer_grey);

	fopen_s(&f_PDF, "HM_f_PDF_grey.txt", "w+");
	fopen_s(&f_equalized_PDF, "HM_f_equalized_PDF.txt", "w+");

	fopen_s(&f_trans_func_input, "HM_f_trans_func_input.txt", "w+");
	fopen_s(&f_trans_func_refer, "HM_f_trans_func_refer.txt", "w+");
	fopen_s(&f_trans_func_result, "HM_f_trans_func_inverse.txt", "w+");

	// transfer function
	G trans_func_input[L] = { 0 };
	G trans_func_refer[L] = { 0 };
	G trans_func_refer_inverse[L] = { 0 }; //use inverse matrix

	// histogram equalization
	hist_match(input_grey, output1, trans_func_input, CDF_input); //use this output later.
	hist_match(refer_grey, output2, trans_func_refer, CDF_refer);

	Mat result = output1.clone();


	// equalized PDF(중간결과)
	float* equalized_PDF_input = cal_PDF(output1); // input 1차변환 PDF
	float* equalized_PDF_refer = cal_PDF(output2); // reference이미지의 결과

	// Make inverse function
	G i = 0;
	for (int k = 0; k < L; k++) {
		trans_func_refer_inverse[k] = 0;
	}

	for (int k = 0; k < L; k++) {
		//index, data 뒤바꾸기
		while (i <= trans_func_refer[k]) {
			if (trans_func_refer_inverse[i] == 0) {
				trans_func_refer_inverse[i] = k;
			}
			i++;
		}
	}
	//마지막으로 역함수, s(intermediate result)를 이용해서 result를 만드는 과정
	float* PDF_output1 = cal_PDF(output1);
	float* CDF_output1 = cal_CDF(output1);

	hist_match_final(output1, result, trans_func_refer_inverse, CDF_output1);

	//equalized PDF(최종결과)
	float* equalized_PDF_result = cal_PDF(result);

	for (int i = 0; i < L; i++) {
		// write PDF
		fprintf(f_PDF, "%d\t%f\n", i, PDF_input[i]);
		fprintf(f_equalized_PDF, "%d\t%f\n", i, equalized_PDF_result[i]);

		// write transfer functions
		fprintf(f_trans_func_input, "%d\t%d\n", i, trans_func_input[i]);
		fprintf(f_trans_func_refer, "%d\t%d\n", i, trans_func_refer[i]);
		fprintf(f_trans_func_result, "%d\t%d\n", i, trans_func_refer_inverse[i]);
	}
	

	// memory release
	free(PDF_input);
	free(CDF_input);
	free(PDF_refer);
	free(CDF_refer);
	free(PDF_output1);
	free(CDF_output1);
	fclose(f_PDF);
	fclose(f_equalized_PDF);
	fclose(f_trans_func_input);
	fclose(f_trans_func_result);
	fclose(f_trans_func_result);

	////////////////////// Show each image ///////////////////////

	namedWindow("input_gs", WINDOW_AUTOSIZE);
	imshow("input_gs", input_grey);

	namedWindow("intermediate_gs", WINDOW_AUTOSIZE);
	imshow("intermediate_gs", output1);

	namedWindow("refer_gs", WINDOW_AUTOSIZE);
	imshow("refer_gs", refer_grey);

	namedWindow("Matched Image", WINDOW_AUTOSIZE);
	imshow("Matched Image", result);

	//히스토그램 출력
	namedWindow("Histogram_input", WINDOW_AUTOSIZE);
	imshow("Histogram_input", drawHist(input_grey));
	
	namedWindow("Histogram_result", WINDOW_AUTOSIZE);
	imshow("Histogram_result", drawHist(result));
	//////////////////////////////////////////////////////////////

	waitKey(0);

	return 0;
}

void hist_match(Mat& input, Mat& equalized, G* trans_func, float* CDF) {
	// compute transfer function
	for (int i = 0; i < L; i++)
		trans_func[i] = (G)((L - 1) * CDF[i]);

	// perform the transfer function
	for (int i = 0; i < input.rows; i++)
		for (int j = 0; j < input.cols; j++)
			equalized.at<G>(i, j) = trans_func[input.at<G>(i, j)];
}
void hist_match_final(Mat& input, Mat& equalized, G* trans_func, float* CDF) {
	// perform the transfer function
	for (int i = 0; i < input.rows; i++)
		for (int j = 0; j < input.cols; j++)
			equalized.at<G>(i, j) = trans_func[input.at<G>(i, j)];
}

Mat drawHist(Mat img) {
	MatND histogram;
	const int* channel_numbers = { 0 };
	float channel_range[] = { 0.0, 255.0 };
	const float* channel_ranges = channel_range;
	int number_bins = 255;
	calcHist(&img, 1, channel_numbers, Mat(), histogram, 1, &number_bins, &channel_ranges);

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