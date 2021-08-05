#include "hist_func.h"

/*
	grey scale image�� Histogram Matching �������
	�����ϴ� ����̴�.
	���� ǳ���� �Կ��� �������� �̹�����
	contranst�� ���� �Ǹ��� �̹����� �������� �ΰ�
	������ �̹������� �ִ��� �� �̹����� ����ϵ��� �����ϴ� �۾��̴�.
	�� ������ �����ϱ� ���� �̹����� transformation function�� �ΰ��� �ʿ��ϴ�.
	�ϳ��� ��ȯ�� �̹���, �׸��� �ϳ��� �� ���� �̹����� HE�� �����ϱ� ���� ���̴�.
	��ǲ �̹����� ���� T, G-1�� �ι��� �����ؼ� ����� ����Ѵ�.
	���� HE�� 2�� �����ϴ°������� �� ���������� �������� trasformation�Լ���
	���Լ��� ���ϴ� ������ ���ľ� �Ѵٴ� ���̴�.
	���Լ��� 1:1������ �ƴϸ� ���Ҽ� ���� ���� ��Ģ�ε�, �־���
	transformation function�� �׷��� ���� ��찡 �Ϲ����̴�.
	���� ���Լ��� ���� �Ŀ� �̸� 1:1�������� �ٽ� �������ִ� ������ �ʿ��ϴ�.
*/

/*
	�÷������� ������׷� ��Ī�� YUV�� ����ؾ� �Ѵ�.
*/

void hist_match(Mat& input, Mat& equalized, G* trans_func, float* CDF);
void hist_match_final(Mat& input, Mat& equalized, G* trans_func, float* CDF);

Mat* drawHist(Mat img);

int main() {

	//input, output �̹����� ������ Matrix�� �����Ѵ�.
	Mat input = imread("blurred_input.jpg", CV_LOAD_IMAGE_COLOR); //���⿡ �ٲٰ��� �ϴ� �̹����� �ִ´�.
	Mat refer = imread("input.jpg", CV_LOAD_IMAGE_COLOR); //������ �̹���

	Mat input_YUV;
	Mat refer_YUV;

	cvtColor(input, input_YUV, CV_RGB2YUV);	// RGB -> YUV
	cvtColor(refer, refer_YUV, CV_RGB2YUV);	// RGB -> YUV

	// split each channel(Y, U, V)
	Mat input_channels[3];
	split(input, input_channels); //YUV�� ���� �и��ؼ� �����Ѵ�.
	Mat refer_channels[3];
	split(refer, input_channels); //YUV�� ���� �и��ؼ� �����Ѵ�.
	// Y�� ���ؼ��� HE����
	// grey scale�� ���� �̹����� ������.
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

	float** PDF_input = cal_PDF_RGB(input); //���Ͽ� ���ֱ� ���� �÷��� ȣ��
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

	// equalized PDF(�߰����)
	float* equalized_PDF_input = cal_PDF(output1); // input 1����ȯ PDF
	float* equalized_PDF_refer = cal_PDF(output2); // reference�̹����� ���

	// Make inverse function
	G i = 0;
	for (int k = 0; k < L; k++) {
		trans_func_refer_inverse[k] = 0;
	}

	for (int k = 0; k < L; k++) {
		//index, data �ڹٲٱ�
		while (i <= trans_func_refer[k]) {
			if (trans_func_refer_inverse[i] == 0) {
				trans_func_refer_inverse[i] = k;
			}
			i++;
		}
	}

	//���������� ���Լ�, s(intermediate result)�� �̿��ؼ� result�� ����� ����
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

	//equalized PDF(�������)
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

	//������׷� ���
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

	//R,G,B���� ������׷� ���
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

	//�����Ҵ����� ������ ������ �ڿ� �ϳ��� ���� �������־�� �Ѵ�.
	Mat* hist = new Mat[3];
	hist[0] = histImgB;
	hist[1] = histImgG;
	hist[2] = histImgR;

	return hist;
}