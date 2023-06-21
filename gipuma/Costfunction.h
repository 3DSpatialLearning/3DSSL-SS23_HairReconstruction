#pragma once
#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include <iomanip>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <cmath>
#include <numeric>

using namespace std;
using namespace cv;

#ifndef M_PI
#define M_PI 3.14159265358979323846
#define HEIGHT 1100
#define WIDTH 1604
#endif

//read the orientation and confidence map
//input: filepath
//output: a matrix(double) store the orientation/confidence values
cv::Mat importFloatImage__(const std::string& filePath) {
	// Open the file for reading as a binary file
	std::ifstream file(filePath, std::ios::binary);

	if (!file.is_open()) {
		std::cerr << "Error opening file: " << filePath << std::endl;
		return cv::Mat();
	}

	try {
		// Read the width and height from the file
		int width, height;
		file.read(reinterpret_cast<char*>(&height), sizeof(int));
		file.read(reinterpret_cast<char*>(&width), sizeof(int));
		// Create a cv::Mat object for the float image
		cv::Mat floatImage(height, width, CV_64F);
		// Read the pixel values from the file
		file.read(reinterpret_cast<char*>(floatImage.ptr()), floatImage.total() * sizeof(double));

		// Close the file
		file.close();

		//std::cout << "Float image imported successfully: " << filePath << std::endl;
		return floatImage;
	}
	catch (const std::exception& e) {
		std::cerr << "Error importing float image: " << e.what() << std::endl;
		return cv::Mat();
	}
}

//get the intensity value of the sample point's position
//input: the imagepath and sample points
//output: the insensity values at the samples position(16 views, 41 sample points per view)
vector<vector<float>> GetIntensityValue(const vector<string>& imagepath, const vector<vector<Vec2f>>& points)
{
	vector<vector<float>> intensity(16, vector<float>(41));
	for (int i = 0; i < 16; i++)
	{
		Mat image_1 = cv::imread(imagepath[i], IMREAD_GRAYSCALE);
		for (int j = 0; j < 41; j++)
		{
			intensity[i][j] = static_cast<float>(image_1.at<uchar>(int(points[i][j][1]), int(points[i][j][0])));
		}
	}
	return intensity;
}

//compute normalized cross correlation between two sets of intensity value of sample points
//input: two sets of intensity values (each contains 41 values)
//output: NCC (Normalized Cross Correlation,) of these two sets, ranges from -1 to 1
float computeNCC(const vector<float>& intensity_1, const vector<float>& intensity_2)
{
	// Check if the sets have the same size
	if (intensity_1.size() != intensity_2.size())
	{
		std::cerr << "Error: Sets must have the same size." << std::endl;
		return 0.0f;
	}

	// Compute the means of the sets
	float mean1 = accumulate(intensity_1.begin(), intensity_1.end(), 0.0f) / intensity_1.size();
	float mean2 = accumulate(intensity_2.begin(), intensity_2.end(), 0.0f) / intensity_2.size();

	// Compute the cross-correlation
	float numerator = 0.0f;
	float denominator1 = 0.0f;
	float denominator2 = 0.0f;
	for (size_t i = 0; i < intensity_1.size(); ++i)
	{
		numerator += (intensity_1[i] - mean1) * (intensity_2[i] - mean2);
		denominator1 += std::pow(intensity_1[i] - mean1, 2);
		denominator2 += std::pow(intensity_2[i] - mean2, 2);
	}

	// Compute the normalized cross-correlation
	float denominator = std::sqrt(denominator1 * denominator2);
	if (denominator == 0.0f)
	{
		std::cerr << "Error: Denominator is zero." << std::endl;
		return 0.0f;
	}

	float correlation = numerator / denominator;
	return correlation;
}

//compute the intensity cost
//input: intensity values of samples in all views
//output: the final intensity cost
float ComputeIntensityCost(vector<vector<float>>& intensity) {
	float intensity_cost = 0.0;
	for (int i = 0; i < 15; i++)
	{
		intensity_cost += computeECC(intensity[0], intensity[i]);
	}
	intensity_cost /= 15;
	return intensity_cost;
}

//get the sampled direction in order to compute the geometric cost
//input: the sample points for all views
//output: the 2d projected line direction for all views
vector<Vec2f> GetSampleDirection(vector<vector<Vec2f>>& sample_points) {
	vector<Vec2f> sample_directions(16);
	for (int i = 0; i < 16; i++) {
		sample_directions[i][0] = sample_points[i][0][0] - sample_points[i][1][0];
		sample_directions[i][1] = sample_points[i][0][1] - sample_points[i][1][1];
	}
	return sample_directions;
}

//compute the angular difference of the line direction and the orientation angle at the sampled position
//input: the 2d projected line in float vector and 2d orientation value in radian
//output: the angular difference between them in degree
float angular_Diff(const Vec2f& line_direction, const float& orientation_angle)
{

	float line_angle = atan2(line_direction[1], line_direction[0]) * 180.0 / M_PI;
	float angle_diff = abs(line_angle - orientation_angle * 180 / M_PI);
	angle_diff = min(angle_diff, (float(360.0) - angle_diff));

	return angle_diff;
}


//compute the geometric cost
//input: the 2D projected direction and orientation and confidence value, the sample points, weights for each view
//output: the geometric cost
float geometricCost(const vector<Vec2f>& sampledirection,//sampleline directions in all views,
	const vector<vector<Vec2f>>& samplepoints, //all the samples in all views
	vector<cv::Mat>& orientation_map,//orientationmaps of all the views
	vector<cv::Mat>& confidence_map,
	const float& r0,//weight for ith view
	const float& ri
) {
	float cost = 0.0;
	float r;
	float r_sum = 0.0;

	for (int i = 0; i < 2; i++)
	{
		if (i == 0)
		{
			r = r0;
		}
		else {
			r = ri;
		}
		float g_i = 0.0;
		float confidence_sum = 0.0;

		for (int j = 0; j < 41; j++) {
			g_i += confidence_map[i].at<float>(int(samplepoints[i][j][0]), int(samplepoints[i][j][1]))
				* angular_Diff(sampledirection[i], orientation_map[i].at<float>(int(samplepoints[i][j][0]), int(samplepoints[i][j][1])));
			confidence_sum += confidence_map[i].at<float>(int(samplepoints[i][j][0]), int(samplepoints[i][j][1]));
		}
		g_i /= confidence_sum;
		cost += r * g_i;
		r_sum += r;
	}
	cost = cost / r_sum;
	return cost;
}


//add the geometric and intensity cost to get the final cost
float compute_costfunction(
	const vector<string>& imagepath,
	const vector<vector<Vec2f>>& samplepoints,
	const vector<Vec2f>& sampledirection,
	vector<cv::Mat>& orientation_map,
	vector<cv::Mat>& confidence_map,
	const float& r0, //weight for the reference view: 16.0
	const float& ri //weight for other views: 1.0
)
{
	float geometric_cost = 0.0;
	float intensity_cost = 0.0;
	geometric_cost = geometricCost(sampledirection,
		samplepoints, 
		orientation_map,
		confidence_map,
		r0,
		ri
	);
	vector<vector<float>> intensity_value = GetIntensityValue(imagepath, samplepoints);
	intensity_cost = ComputeIntensityCost(intensity_value);
	float alpha = 0.1;
	float cost_all = (1 - alpha) * geometric_cost + alpha * intensity_cost;
	return cost_all;
}