#include "hist_func.h"

void hist_eq(Mat &input, Mat &equalized, G *trans_func, float *CDF);

int main() {

	Mat input = imread("input.jpg", CV_LOAD_IMAGE_COLOR);
	Mat equalized_YUV;
	Mat output = input.clone();

	cvtColor(input, equalized_YUV, CV_RGB2YUV);	// RGB -> YUV
	
	// split each channel(Y, U, V)
	Mat channels[3];
	split(equalized_YUV, channels);
	Mat Y = channels[0];						// U = channels[1], V = channels[2]
	Mat equalized_Y = Y.clone();
	// PDF or transfer function txt files
	FILE *f_equalized_PDF_YUV, *f_PDF_RGB;
	FILE *f_trans_func_eq_YUV;

	float **PDF_RGB = cal_PDF_RGB(input);		// PDF of Input image(RGB) : [L][3]
	float *CDF_YUV = cal_CDF(Y);				// CDF of Y channel image

	fopen_s(&f_PDF_RGB, "HEYUV_PDF_RGB.txt", "w+");
	fopen_s(&f_equalized_PDF_YUV, "HEYUV_equalized_PDF_YUV.txt", "w+");
	fopen_s(&f_trans_func_eq_YUV, "HEYUV_trans_func_eq_YUV.txt", "w+");

	G trans_func_eq_YUV[L] = { 0 };			// transfer function

	// histogram equalization on Y channel
	hist_eq(Y, equalized_Y, trans_func_eq_YUV, CDF_YUV);
	
	channels[0] = equalized_Y.clone();

	// merge Y, U, V channels
	merge(channels, 3, equalized_YUV);
	
	// YUV -> RGB (use "CV_YUV2RGB" flag)
	cvtColor(equalized_YUV, output,  CV_YUV2RGB);

	// equalized PDF (YUV)
	float* equalized_PDF_YUV = cal_PDF(equalized_Y);

	for (int i = 0; i < L; i++) {
		// write PDF
		for (int k = 0 ; k < 3 ; k++)
			fprintf(f_PDF_RGB, "%d\t%f\n", i, PDF_RGB[i][k]);
		fprintf(f_equalized_PDF_YUV, "%d\t%f\n", i, CDF_YUV[i]);
		fprintf(f_PDF_RGB, "\n");

		// write transfer functions
		fprintf(f_trans_func_eq_YUV, "%d\t%d\n", i, trans_func_eq_YUV[i]);
		
	}

	// memory release
	free(PDF_RGB);
	free(CDF_YUV);
	fclose(f_PDF_RGB);
	fclose(f_equalized_PDF_YUV);
	fclose(f_trans_func_eq_YUV);

	////////////////////// Show each image ///////////////////////

	namedWindow("RGB", WINDOW_AUTOSIZE);
	imshow("RGB", input);

	namedWindow("Equalized_YUV", WINDOW_AUTOSIZE);
	imshow("Equalized_YUV", output);

	Mat* hist = new Mat[3];
	hist = drawHist_RGB_PDF(input);

	namedWindow("histogram_input_B", WINDOW_AUTOSIZE);
	imshow("histogram_input_B", (Mat)hist[0]);
	namedWindow("histogram_input_G", WINDOW_AUTOSIZE);
	imshow("histogram_input_G", (Mat)hist[1]);
	namedWindow("histogram_input_R", WINDOW_AUTOSIZE);
	imshow("histogram_input_R", (Mat)hist[2]);

	hist = new Mat[3];
	hist = drawHist_RGB_PDF(output);

	namedWindow("histogram_output_B", WINDOW_AUTOSIZE);
	imshow("histogram_output_B", (Mat)hist[0]);
	namedWindow("histogram_output_G", WINDOW_AUTOSIZE);
	imshow("histogram_output_G", (Mat)hist[1]);
	namedWindow("histogram_output_R", WINDOW_AUTOSIZE);
	imshow("histogram_output_R", (Mat)hist[2]);

	//////////////////////////////////////////////////////////////

	waitKey(0);

	return 0;
}

// histogram equalization
void hist_eq(Mat &input, Mat &equalized, G *trans_func, float *CDF) {

	// compute transfer function
	for (int i = 0; i < L; i++)
		trans_func[i] = (G)((L - 1) * CDF[i]);

	// perform the transfer function
	for (int i = 0; i < input.rows; i++)
		for (int j = 0; j < input.cols; j++)
			equalized.at<G>(i, j) = trans_func[input.at<G>(i, j)];
}