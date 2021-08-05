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

/*
	컬러버전도 히스토그램 매칭도 YUV를 사용해야 한다.
*/

void hist_match(Mat& input, Mat& equalized, G* trans_func, float* CDF);
void hist_match_final(Mat& input, Mat& equalized, G* trans_func, float* CDF);

Mat* drawHist(Mat img);

int main() {

	//input, output 이미지를 저장할 Matrix를 생성한다.
	Mat input = imread("blurred_input.jpg", CV_LOAD_IMAGE_COLOR); //여기에 바꾸고자 하는 이미지를 넣는다.
	Mat refer = imread("input.jpg", CV_LOAD_IMAGE_COLOR); //참고할 이미지

	Mat input_YUV;
	Mat refer_YUV;

	cvtColor(input, input_YUV, CV_RGB2YUV);	// RGB -> YUV
	cvtColor(refer, refer_YUV, CV_RGB2YUV);	// RGB -> YUV

	// split each channel(Y, U, V)
	Mat input_channels[3];
	split(input, input_channels); //YUV를 각각 분리해서 접근한다.
	Mat refer_channels[3];
	split(refer, input_channels); //YUV를 각각 분리해서 접근한다.
	// Y에 대해서만 HE진행
	// grey scale을 가진 이미지와 유사함.
	Mat input_Y = input_channels[0];			// U = channels[1], V = channels[2]
	Mat refer_Y = refer_channels[0];

	Mat output1 = input_Y.clone();
	Mat output2 = refer_Y.clone();

	// PDF or transfer function txt files
	FILE* f_PDF_RGB; //PDF of original Image
	FILE* f_equalized_PDF_RGB; //PDF of output image

	FILE* f_trans_func_input;
	FILE* f_trans_func_refer;
	FILE* f_trans_func_result;

	float** PDF_input = cal_PDF_RGB(input); //파일에 써주기 위해 컬러로 호출
	float* PDF_refer = cal_PDF(refer_Y);
	float* CDF_input = cal_CDF(input_Y);
	float* CDF_refer = cal_CDF(refer_Y);

	fopen_s(&f_PDF_RGB, "HSRGB_f_PDF_grey.txt", "w+");
	fopen_s(&f_equalized_PDF_RGB, "HSRGB_f_equalized_PDF.txt", "w+");

	fopen_s(&f_trans_func_input, "HSRGB_f_trans_func_input.txt", "w+");
	fopen_s(&f_trans_func_refer, "HSRGB_f_trans_func_refer.txt", "w+");
	fopen_s(&f_trans_func_result, "HSRGB_f_trans_func_inverse.txt", "w+");

	// transfer function
	G trans_func_input[L] = { 0 };
	G trans_func_refer[L] = { 0 };
	G trans_func_refer_inverse[L] = { 0 }; //use inverse matrix

	// histogram equalization
	hist_match(input_Y, output1, trans_func_input, CDF_input); //use this output later.
	hist_match(refer_Y, output2, trans_func_refer, CDF_refer);

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
	Mat result = input.clone();
	Mat output = input.clone();
	float* PDF_output1 = cal_PDF(output1);
	float* CDF_output1 = cal_CDF(output1);

	hist_match_final(output1, input_Y, trans_func_refer_inverse, CDF_output1);
	input_channels[0] = input_Y;
	// merge Y, U, V channels
	merge(input_channels, 3, output);

	// YUV -> RGB (use "CV_YUV2RGB" flag)
	cvtColor(output, result, CV_YUV2RGB);

	//equalized PDF(최종결과)
	float** equalized_PDF_result = cal_PDF_RGB(result);

	for (int i = 0; i < L; i++) {
		// color image print
		for (int k = 0; k < 3; k++) {
			fprintf(f_PDF_RGB, "%d\t%f ", i, PDF_input[i][k]);
			fprintf(f_equalized_PDF_RGB, "%d\t%f ", i, equalized_PDF_result[i][k]);
		}
		fprintf(f_PDF_RGB, "\n");
		fprintf(f_equalized_PDF_RGB, "\n");
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
	fclose(f_PDF_RGB);
	fclose(f_equalized_PDF_RGB);
	fclose(f_trans_func_input);
	fclose(f_trans_func_result);
	fclose(f_trans_func_result);

	////////////////////// Show each image ///////////////////////

	namedWindow("input_gs", WINDOW_AUTOSIZE);
	imshow("input_gs", input);

	namedWindow("intermediate_gs", WINDOW_AUTOSIZE);
	imshow("intermediate_gs", output1);

	namedWindow("refer_gs", WINDOW_AUTOSIZE);
	imshow("refer_gs", refer);

	namedWindow("Matched Image", WINDOW_AUTOSIZE);
	imshow("Matched Image", result);

	//히스토그램 출력
	Mat* hist = new Mat[3];
	hist = drawHist(input);
	
	namedWindow("histogram_input_B", WINDOW_AUTOSIZE);
	imshow("histogram_input_B", (Mat)hist[0]);
	namedWindow("histogram_input_G", WINDOW_AUTOSIZE);
	imshow("histogram_input_G", (Mat)hist[1]);
	namedWindow("histogram_input_R", WINDOW_AUTOSIZE);
	imshow("histogram_input_R", (Mat)hist[2]);

	hist = drawHist(result);
	namedWindow("histogram_result_B", WINDOW_AUTOSIZE);
	imshow("histogram_result_B", (Mat)hist[0]);
	namedWindow("histogram_result_G", WINDOW_AUTOSIZE);
	imshow("histogram_result_G", (Mat)hist[1]);
	namedWindow("histogram_input_R", WINDOW_AUTOSIZE);
	imshow("histogram_result_R", (Mat)hist[2]);

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

Mat* drawHist(Mat img) {
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