#include "hist_func.h"

int main() {
	
	Mat input = imread("input.jpg", CV_LOAD_IMAGE_COLOR);
	Mat input_gray;

	cvtColor(input, input_gray, CV_RGB2GRAY);	// convert RGB to Grayscale
	
	// PDF, CDF txt files
	FILE *f_PDF, *f_CDF;
	
	fopen_s(&f_PDF, "PDF_CDF_PDF.txt", "w+");
	fopen_s(&f_CDF, "PDF_CDF_CDF.txt", "w+");

	// each histogram
	float *PDF = cal_PDF(input_gray);		// PDF of Input image(Grayscale) : [L]
	float *CDF = cal_CDF(input_gray);		// CDF of Input image(Grayscale) : [L]

	for (int i = 0; i < L; i++) {
		// write PDF, CDF
		fprintf(f_PDF, "%d\t%f\n", i, PDF[i]);
		fprintf(f_CDF, "%d\t%f\n", i, CDF[i]);
	}

	// memory release
	free(PDF);
	free(CDF);
	fclose(f_PDF);
	fclose(f_CDF);
	
	////////////////////// Show each image ///////////////////////
	
	namedWindow("RGB", WINDOW_AUTOSIZE);
	imshow("RGB", input);

	namedWindow("Grayscale", WINDOW_AUTOSIZE);
	imshow("Grayscale", input_gray);

	//Histogram
	//For RGB
	Mat* hist = new Mat[3];
	hist = drawHist_RGB_PDF(input);
	namedWindow("histogram_input_B_PDF", WINDOW_AUTOSIZE);
	imshow("histogram_input_B_PDF", (Mat)hist[0]);
	namedWindow("histogram_input_G_PDF", WINDOW_AUTOSIZE);
	imshow("histogram_input_G_PDF", (Mat)hist[1]);
	namedWindow("histogram_input_R_PDF", WINDOW_AUTOSIZE);
	imshow("histogram_input_R_PDF", (Mat)hist[2]);

	hist = drawHist_RGB_CDF(input);
	namedWindow("histogram_input_B_CDF", WINDOW_AUTOSIZE);
	imshow("histogram_input_B_CDF", (Mat)hist[0]);
	namedWindow("histogram_input_G_CDF", WINDOW_AUTOSIZE);
	imshow("histogram_input_G_CDF", (Mat)hist[1]);
	namedWindow("histogram_input_R_CDF", WINDOW_AUTOSIZE);
	imshow("histogram_input_R_CDF", (Mat)hist[2]);

	//For Grayscale
	namedWindow("Histogram_result_Gray_PDF", WINDOW_AUTOSIZE);
	imshow("Histogram_result_Gray_PDF", drawHist_gray_PDF(input_gray));
	namedWindow("Histogram_result_Gray_CDF", WINDOW_AUTOSIZE);
	imshow("Histogram_result_Gray_CDF", drawHist_gray_CDF(input_gray));

	//////////////////////////////////////////////////////////////

	waitKey(0);

	return 0;
}

