//
//  Orient2D.hpp
//  HairSketch
//

// TODO: Clear up the code
#pragma once
#ifndef __ORIENT2D_HPP__
#define __ORIENT2D_HPP__

// #include "omp.h"
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <vector>
#include <string>
#include <fftw3.h>
#include "Im.hpp"
#include "OrientMap.hpp"
#include <fstream>
#include <limits>

#include <dirent.h>
#include <sys/types.h>
#include <sys/stat.h>

using namespace std;

class COrient2D
{
public:
    COrient2D() {}

    COrient2D(const std::string &filename, const std::string &outfilename)
    {
        int ndegree = 180;
        double sigma_h = 0.5;
        double sigma_l = 1;
        double sigma_y = 4;
        int npass = 1;

        // Read the input image into colorImg
        cv::Mat colorImg = cv::imread(filename, cv::IMREAD_COLOR);
        cv::Mat filterImg;
        cv::cvtColor(colorImg, filterImg, cv::COLOR_BGRA2GRAY);
        filterImg.convertTo(filterImg, CV_64FC1);
        cv::normalize(filterImg, filterImg, 0, 1, cv::NORM_MINMAX, CV_64F);
        std::cout << "filter image size: " << filterImg.size() << " " << filterImg.channels() << std::endl;
        std::cout << "color image size: " << colorImg.size() << " " << colorImg.channels() << std::endl;

        cv::Mat m_hairConf = cv::Mat::zeros(filterImg.size(), CV_64FC1);
        cv::Mat m_hairOrient = cv::Mat::zeros(filterImg.size(), CV_64FC1);
        cv::Mat m_hairVariance = cv::Mat::zeros(filterImg.size(), CV_64FC1);

        // Compute the mask using the provided sigma_l value
        cv::Mat mask = cv::Mat::zeros(filterImg.size(), CV_64FC1);
        compute_mask_cv(filterImg, sigma_l, mask);

        int npix = filterImg.cols * filterImg.rows;
        cv::Mat interFilter, interOrient, interConf, interVar;
        // Perform npass iterations of filtering and manipulation
        std::string folder_name{outfilename};
        mkdir(folder_name.data(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
        for (int i = 0; i < npass; i++)
        {
            // Apply the filter to the normalized buffer and store the result in the alternate buffer
            filter(filterImg, ndegree, sigma_h, sigma_l, sigma_y, m_hairOrient, m_hairConf, m_hairVariance, outfilename);
            // Calculate the maximum magnitude from the filtered buffer
            // double max_mag = 0.0;
            // for (int j = 0; j < npix; j++)
            // {
            //     double mag = mask.at<double>(j) != 0.0 ? max<double>(m_hairConf.at<double>(j), 0.0) : 0.0;
            //     filterImg.at<double>(j) = mag;
            //     m_hairConf.at<double>(j) = mag;
            //     max_mag = max<double>(max_mag, mag);
            // }

            // Normalize the values in the buffer by dividing by the maximum magnitude
            // filterImg /= max_mag;
            // cv::GaussianBlur(m_hairVariance, interVar, cv::Size(21, 21), 0.);
            // cv::GaussianBlur(m_hairConf, interConf, cv::Size(21, 21), 0.);

            cv::normalize(filterImg, interFilter, 0, 255, cv::NORM_MINMAX, CV_8U);
            cv::imwrite(outfilename + "/interFilter" + std::to_string(i) + ".png", interFilter); // mask((outfilename + "/mask.png")); // Write the mask image to file
            cv::normalize(m_hairConf, interConf, 0, 255, cv::NORM_MINMAX, CV_8U);
            cv::imwrite(outfilename + "/interConf" + std::to_string(i) + ".png", interConf);
            cv::normalize(m_hairOrient, interOrient, 0, 255, cv::NORM_MINMAX, CV_8U);
            cv::imwrite(outfilename + "/interOrient" + std::to_string(i) + ".png", interOrient);
            cv::normalize(m_hairVariance, interVar, 0, 255, cv::NORM_MINMAX, CV_8U);
            cv::imwrite(outfilename + "/interVar" + std::to_string(i) + ".png", interVar);
        }

        exportFloatImage(m_hairConf, outfilename + "_undiffusedConf.flo");
        exportFloatImage(m_hairOrient, outfilename + "_finalOrient.flo");

        // // Visualize the orientations in the buffer using a color scheme
        std::cout << "converting color" << std::endl;
        cv::Mat colorizedOrientImg = cv::Mat::zeros(colorImg.size(), CV_64FC3);
        cv::Mat normalizedColorizedOrientImg;
        std::cout << "converting color finished" << std::endl;

        viz_ori_2color(colorizedOrientImg, m_hairOrient, m_hairConf, mask);
        cv::normalize(colorizedOrientImg, normalizedColorizedOrientImg, 0, 255, cv::NORM_MINMAX, CV_8U);
        cv::imwrite(outfilename + "_orientColorized.png", normalizedColorizedOrientImg);

        // Diffuse confidence results
        cv::Mat filterHairConf, filterHairVar;
        cv::Mat normalizedHairConf, normalizedHairVar;
        write_pixel_response(m_hairVariance, outfilename, "variance");
        write_pixel_response(m_hairConf, outfilename, "confidence");

        cv::normalize(m_hairConf, normalizedHairConf, 0, 1, cv::NORM_MINMAX, CV_64F);
        cv::GaussianBlur(m_hairConf, filterHairConf, cv::Size(21, 21), 0.);
        cv::normalize(filterHairConf, filterHairConf, 0, 1, cv::NORM_MINMAX, CV_64F);

        cv::normalize(m_hairVariance, normalizedHairVar, 0, 1, cv::NORM_MINMAX, CV_64F);
        cv::GaussianBlur(m_hairVariance, filterHairVar, cv::Size(21, 21), 0.);
        cv::normalize(filterHairVar, filterHairVar, 0, 1, cv::NORM_MINMAX, CV_64F);

        int i = 0, j = 0;
        for (int hI = 0; hI < m_hairConf.rows; hI++)
        {
            for (int wI = 0; wI < m_hairConf.cols; wI++)
            {
                if (mask.at<double>(hI, wI) == 0.0)
                {
                    m_hairConf.at<double>(hI, wI) = 0.;
                    m_hairVariance.at<double>(hI, wI) = 0.;
                }
            }
        }

        write_pixel_response(m_hairVariance, outfilename, "variance_norm");
        write_pixel_response(m_hairConf, outfilename, "confidence_norm");

        viz_ori_2color(colorizedOrientImg, m_hairOrient, filterHairConf, mask);
        cv::normalize(colorizedOrientImg, normalizedColorizedOrientImg, 0, 255, cv::NORM_MINMAX, CV_8U);
        cv::imwrite(outfilename + "_orientColorizedNew.png", normalizedColorizedOrientImg);

        viz_ori_2color(colorizedOrientImg, m_hairOrient, filterHairVar, mask);
        cv::normalize(colorizedOrientImg, normalizedColorizedOrientImg, 0, 255, cv::NORM_MINMAX, CV_8U);
        cv::imwrite(outfilename + "_orientVarianceColorizedNew.png", normalizedColorizedOrientImg);

        cv::normalize(m_hairConf, m_hairConf, 0, 255, cv::NORM_MINMAX, CV_8U);
        cv::imwrite(outfilename + "_undiffusedConf.png", m_hairConf);

        cv::normalize(m_hairVariance, m_hairVariance, 0, 255, cv::NORM_MINMAX, CV_8U);
        cv::imwrite(outfilename + "_undiffusedVar.png", m_hairVariance);

        cv::normalize(normalizedHairConf, normalizedHairConf, 0, 255, cv::NORM_MINMAX, CV_8U);
        cv::imwrite(outfilename + "_normalizedConf.png", normalizedHairConf);

        cv::normalize(normalizedHairVar, normalizedHairVar, 0, 255, cv::NORM_MINMAX, CV_8U);
        cv::imwrite(outfilename + "_normalizedVar.png", normalizedHairVar);

        exportFloatImage(filterHairConf, outfilename + "_finalConfidence.flo");
        cv::normalize(filterHairConf, filterHairConf, 0, 255, cv::NORM_MINMAX, CV_8U);
        cv::imwrite(outfilename + "_finalizedConf.png", filterHairConf);

        exportFloatImage(filterHairVar, outfilename + "_finalVariance.flo");
        cv::normalize(filterHairVar, filterHairVar, 0, 255, cv::NORM_MINMAX, CV_8U);
        cv::imwrite(outfilename + "_finalizedVar.png", filterHairVar);
    }

    ~COrient2D() {}

    void exportFloatImage(const cv::Mat &floatImage, const std::string &filePath)
    {
        // Open the file for writing as a binary file
        std::ofstream file(filePath, std::ios::binary);

        if (!file.is_open())
        {
            std::cerr << "Error opening file: " << filePath << std::endl;
            return;
        }

        try
        {
            // Write width and height to the file
            int width = floatImage.cols;
            int height = floatImage.rows;

            file.write(reinterpret_cast<const char *>(&width), sizeof(int));
            file.write(reinterpret_cast<const char *>(&height), sizeof(int));

            // Write the pixel values
            file.write(reinterpret_cast<const char *>(floatImage.ptr()), floatImage.total() * sizeof(double));

            // Close the file
            file.close();
            std::cout << "Float image exported successfully: " << filePath << std::endl;
        }
        catch (const std::exception &e)
        {
            std::cerr << "Error exporting float image: " << e.what() << std::endl;
        }
    }

    cv::Mat importFloatImage(const std::string &filePath, cv::Mat &floatImage)
    {
        // Open the file for reading as a binary file
        std::ifstream file(filePath, std::ios::binary);

        if (!file.is_open())
        {
            std::cerr << "Error opening file: " << filePath << std::endl;
            return cv::Mat();
        }

        try
        {
            // Read the width and height from the file
            int width, height;
            file.read(reinterpret_cast<char *>(&width), sizeof(int));
            file.read(reinterpret_cast<char *>(&height), sizeof(int));
            // Create a cv::Mat object for the float image

            // Read the pixel values from the file
            file.read(reinterpret_cast<char *>(floatImage.ptr()), floatImage.total() * sizeof(double));

            // Close the file
            file.close();

            std::cout << "Float image imported successfully: " << filePath << std::endl;
            return floatImage;
        }
        catch (const std::exception &e)
        {
            std::cerr << "Error importing float image: " << e.what() << std::endl;
            return cv::Mat();
        }
    }

    void mexican_hat(
        const fftw_complex *in, int w, int h, double sigma_xh,
        double sigma_xl, double sigma_y, double theta, fftw_complex *out)
    {
        // Calculate sine and cosine of theta for coordinate rotation
        double s = sin(theta);
        double c = cos(theta);

        // Calculate multipliers for the Gaussian and Laplacian components
        double xhmult = -2.0 * sqr(M_PI * sigma_xh);
        double xlmult = -2.0 * sqr(M_PI * sigma_xl);
        double ymult = -2.0 * sqr(M_PI * sigma_y);

// Apply the Mexican Hat wavelet to each frequency component
#pragma omp parallel for
        for (int y = 0; y < h; y++)
        {
            // Normalize the y-coordinate
            double ynorm = (y >= h / 2) ? y - h : y; // [-1, 1]
            ynorm /= h;

            for (int x = 0; x <= w / 2; x++)
            {
                // Normalize the x-coordinate
                double xnorm = static_cast<double>(x) / w; // [-1, 1]

                // Calculate the squared and rotated coordinates
                double xrot2 = sqr(s * xnorm - c * ynorm);
                double yrot2 = sqr(c * xnorm + s * ynorm);

                // Compute the index for the current frequency component
                int i = x + y * (w / 2 + 1);

                // Calculate the Mexican Hat wavelet value for the current frequency
                double g = exp(xhmult * xrot2 + ymult * yrot2) -
                           exp(xlmult * xrot2 + ymult * yrot2);

                // Apply the Mexican Hat wavelet filter to the frequency component
                out[i][0] = in[i][0] * g; // Real component
                out[i][1] = in[i][1] * g; // Imaginary component
            }
        }
    }

    void gabor(
        const fftw_complex *in, int w, int h, double sigma_xh,
        double sigma_xl, double sigma_y, double theta, fftw_complex *out)
    {
        // Calculate sine and cosine of theta for coordinate rotation
        double s = std::sin(theta);
        double c = std::cos(theta);

        // Calculate multipliers for the Gaussian and Laplacian components
        double xhmult = -2.0 * sqr(M_PI * sigma_xh);
        double xlmult = -2.0 * sqr(M_PI * sigma_xl);
        double ymult = -2.0 * sqr(M_PI * sigma_y);
        double psi = 0;
        double gamma = 1;
        double sigma = 2;
        double lambda = 2;
// Apply the Mexican Hat wavelet to each frequency component
#pragma omp parallel for
        for (int y = 0; y < h; y++)
        {
            // Normalize the y-coordinate
            double ynorm = (y >= h / 2) ? y - h : y; // [-1, 1]
            ynorm /= h;

            for (int x = 0; x <= w / 2; x++)
            {
                // Normalize the x-coordinate
                double xnorm = static_cast<double>(x) / w; // [0, 1]

                // Calculate the squared and rotated coordinates
                double x_theta = xnorm * c + ynorm * s;
                double y_theta = -xnorm * s + ynorm * c;
                double xrot2 = x_theta * x_theta;
                double yrot2 = y_theta * y_theta;

                // Compute the index for the current frequency component
                int i = x + y * (w / 2 + 1);

                // Calculate the Gabor filter value at the current frequency component
                double real_part = std::exp(-(xrot2 + gamma * gamma * yrot2) / (2.0 * sigma * sigma)) *
                                   std::cos(2.0 * M_PI * x_theta / lambda + psi);
                double imag_part = std::exp(-(xrot2 + gamma * gamma * yrot2) / (2.0 * sigma * sigma)) *
                                   std::sin(2.0 * M_PI * x_theta / lambda + psi);

                // Apply the Mexican Hat wavelet filter to the frequency component
                out[i][0] = in[i][0] * real_part; // Real component
                out[i][1] = in[i][1] * imag_part; // Imaginary component
            }
        }
    }

    void gaussian(const fftw_complex *in, int w, int h, double sigma, fftw_complex *out)
    {
        // Calculate the constant multiplier for the Gaussian function
        double mult = -2.0 * M_PI * M_PI * sigma * sigma;

// Apply the Gaussian filter to each frequency component
#pragma omp parallel for
        for (int y = 0; y < h; y++)
        {
            // Normalize the y-coordinate
            double ynorm = (y >= h / 2) ? y - h : y;
            ynorm /= h;

            for (int x = 0; x <= w / 2; x++)
            {
                // Normalize the x-coordinate
                double xnorm = static_cast<double>(x) / w;

                // Compute the index for the current frequency component
                int i = x + y * (w / 2 + 1);

                // Compute the value of the Gaussian function for the current frequency
                double g = exp(mult * (xnorm * xnorm + ynorm * ynorm));

                // Apply the Gaussian filter to the frequency component
                out[i][0] = in[i][0] * g; // Real component
                out[i][1] = in[i][1] * g; // Imaginary component
            }
        }
    }

    void write_pixel_response(const cv::Mat &vector, const std::string &folder_name, const std::string &name)
    {
        // Open the file for writing
        std::ofstream output_file(folder_name + "/" + name + ".txt");
        // Check if the file opened successfully
        if (!output_file.is_open())
        {
            std::cout << "Failed to open the file." << std::endl;
            return;
        }

        // Write the vector elements to the file
        for (size_t i = 0; i < vector.rows; i++)
        {
            for (size_t j = 0; j < vector.cols; j++)
            {
                output_file << vector.at<double>(i, j) << " ";
            }
        }

        output_file << std::endl;
        // Close the file
        output_file.close();

        std::cout << "Vector exported successfully." << std::endl;
    }
    // Build a single level in the pyramid
    void filter_dense(const fftw_complex *imfft, int w, int h,
                      double sigma_h, double sigma_l, double sigma_y,
                      int orientations, cv::Mat &hair_orient, cv::Mat &hair_conf, cv::Mat &hair_variance, const std::string &folder_name)
    {
        // Clear and resize the output image
        hair_conf = cv::Mat::zeros(cv::Size(w, h), CV_64FC1);
        hair_orient = cv::Mat::zeros(cv::Size(w, h), CV_64FC1);
        hair_variance = cv::Mat::zeros(cv::Size(w, h), CV_64FC1);
        int npix = w * h;

        // Allocate memory for intermediate storage
        fftw_complex *filtfft = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * (w / 2 + 1) * h);
        double *filtered = (double *)fftw_malloc(sizeof(double) * w * h);

        // Create a plan for inverse FFT
        fftw_plan idft = fftw_plan_dft_c2r_2d(h, w, filtfft, filtered, FFTW_ESTIMATE);
        std::vector<cv::Mat> resp_array(orientations);
        for (int i_orient = 0; i_orient < orientations; i_orient++)
        {
            // Calculate the angle for the current orientation
            double angle = M_PI * i_orient / orientations;

            // Apply the Mexican Hat wavelet filter to the input image
            mexican_hat(imfft, w, h, sigma_h, sigma_l, sigma_y, angle, filtfft);
            // gabor(imfft, w, h, sigma_h, sigma_l, sigma_y, angle, filtfft);

            // Perform inverse FFT
            fftw_execute(idft);
            resp_array.at(i_orient) = cv::Mat::zeros(cv::Size(w, h), CV_64FC1);
            for (int i = 0; i < npix; i++)
            {
                double res = filtered[i];
                resp_array.at(i_orient).at<double>(i) = res;
            }

            // Update maximum response in the output image without confidence computation
            // for (int j = 0; j < npix; j++)
            // {
            //     double res = filtered[j];

            //     if (std::abs(hair_conf.at<double>(j)) < std::abs(res))
            //     {
            //         hair_orient.at<double>(j) = angle;
            //         hair_conf.at<double>(j) = res;
            //     }
            // }
        }
        std::vector<double> responses;
        for (int i = 0; i < npix; i++)
        {
            // Find max response at the pixel
            double max_resp = 0.;
            double best_orient = 0.;
            responses.clear();
            for (int i_orient = 0; i_orient < orientations; i_orient++)
            {
                double &resp = resp_array.at(i_orient).at<double>(i);
                if (resp < 0.)
                    resp = 0.;
                else if (resp > max_resp)
                {
                    max_resp = resp;
                    best_orient = M_PI * i_orient / orientations;
                }
                responses.push_back(resp);
            }
            // Calculate variance
            double variance = 0.;
            for (int i_orient = 0; i_orient < orientations; i_orient++)
            {
                double orient = M_PI * i_orient / orientations;
                double orient_diff = MIN(abs(orient - best_orient), MIN(abs(orient - best_orient - M_PI), abs(orient - best_orient + M_PI)));
                double resp_diff = max_resp - resp_array.at(i_orient).at<double>(i); // TODO: calculate mean response.
                variance += orient_diff * pow(resp_diff, 2);                         // resp_diff over 2
            }
            // Standard variance
            variance = sqrt(variance);
            // Update overall variance/orientation if necessary
            if (variance > hair_variance.at<double>(i))
            {
                hair_variance.at<double>(i) = variance;
                hair_orient.at<double>(i) = best_orient;
                hair_conf.at<double>(i) = max_resp;
            }
        }

        // Normalize variance and max response
        double max_all_resp = 0.;
        double max_all_var = 0.;
        for (int i = 0; i < npix; i++)
        {
            if (hair_variance.at<double>(i) > max_all_var)
                max_all_var = hair_variance.at<double>(i);
            if (hair_conf.at<double>(i) > max_all_resp)
                max_all_resp = hair_conf.at<double>(i);
        }
        std::cout << "MAX VARIANCE: " << max_all_var << std::endl;
        std::cout << "MAX ALL RESP: " << max_all_resp << std::endl;
        hair_conf /= max_all_resp;
        max_all_var /= max_all_var;

        // Clean up resources
        fftw_destroy_plan(idft);
        fftw_free(filtered);
        fftw_free(filtfft);
    }

    void filter(
        const cv::Mat &im, int orientations,
        double sigma_h, double sigma_l, double sigma_y, cv::Mat &hair_orient, cv::Mat &hair_conf, cv::Mat &hair_variance, const std::string &folder_name)
    {
        int w = im.cols, h = im.rows, npix = w * h;
        double *ffttmp = (double *)fftw_malloc(sizeof(double) * w * h);
        fftw_complex *imfft = (fftw_complex *)
            fftw_malloc(sizeof(fftw_complex) * (w / 2 + 1) * h);

        cv::Mat curr_im{im}; // Create a copy of the input image

        fftw_plan imdft = fftw_plan_dft_r2c_2d(h, w, ffttmp, imfft, FFTW_ESTIMATE);
        double fftscale = 1.0 / npix;

        // Perform forward Fourier transform
        for (int j = 0; j < npix; j++)
            ffttmp[j] = curr_im.at<double>(j); // Scale pixel values --> curr_im GRAY value

        fftw_execute(imdft);

        // Apply dense pyramid filter to transformed data
        filter_dense(imfft, w, h, sigma_h, sigma_l, sigma_y, orientations, hair_orient, hair_conf, hair_variance, folder_name);

        fftw_destroy_plan(imdft);

        fftw_free(imfft);
        fftw_free(ffttmp);
    }
    // Computes a binary mask based on the input image using gaussian filter in Fourier domain
    void compute_mask_cv(const cv::Mat &im, double sigma, cv::Mat &out)
    {
        const float fg_impact_thresh = 0.9999f;
        int w = im.cols, h = im.rows, npix = w * h;

        // Allocate memory for intermediate storage
        double *ffttmp = (double *)fftw_malloc(sizeof(double) * w * h);
        fftw_complex *imfft = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * (w / 2 + 1) * h);

        // Create plans for forward and inverse FFT
        fftw_plan imdft = fftw_plan_dft_r2c_2d(h, w, ffttmp, imfft, FFTW_ESTIMATE);
        fftw_plan imidft = fftw_plan_dft_c2r_2d(h, w, imfft, ffttmp, FFTW_ESTIMATE);

        double fftscale = 1.0 / npix;

        // Compute the Fourier transform of the input image
        // for (int j = 0; j < npix; j++)
        // {
        // ffttmp[j] = im.at(j) ? fftscale : 0.0;
        // std::cout << im.at<int>(j) << std::endl;
        // }
        std::cout << "computing mask" << std::endl;
        for (int j = 0; j < npix; j++)
        {
            ffttmp[j] = im.at<double>(j) ? fftscale : 0.0;
        }

        std::cout << "computing mask finished" << std::endl;

        fftw_execute(imdft);

        // Apply a Gaussian filter in the Fourier domain
        gaussian(imfft, w, h, sigma, imfft);

        fftw_execute(imidft);

        // Resize the output mask
        out.resize(w, h);

        // Generate a binary mask based on the threshold
        for (int j = 0; j < npix; j++)
        {
            out.at<double>(j) = ffttmp[j] >= fg_impact_thresh ? 1.0 : 0.0;
        }

        // Clean up resources
        fftw_destroy_plan(imidft);
        fftw_destroy_plan(imdft);

        fftw_free(imfft);
        fftw_free(ffttmp);
    }

    void viz_ori_2color(cv::Mat &im, const cv::Mat &hair_orient, const cv::Mat &hair_conf, const cv::Mat &mask)
    {
        int npix = im.cols * im.rows;
        double max_mag = 0.0;
        std::cout << "find max" << std::endl;

        // Find the maximum magnitude among orientations
        for (int i = 0; i < npix; i++)
        {
            double mag = mask.at<double>(i) != 0.0 ? max<double>(hair_conf.at<double>(i), 0.0) : 0.0;
            max_mag = max(max_mag, mag);
        }
        std::cout << "find max finished" << std::endl;

        cv::Vec3d black(0, 0, 0);
        cv::Vec3d gray(0.5, 0.5, 0.5);
        std::cout << "assign colors" << std::endl;
        // Assign colors to orientations based on their values
        std::cout << im.size() << " " << hair_conf.size() << " " << hair_orient.size() << " " << im.channels() << " " << hair_conf.channels() << " " << std::endl;
        for (int i = 0; i < npix; i++)
        {
            if (std::abs(hair_conf.at<double>(i)) < 0.08)
                im.at<cv::Vec3d>(i) = black;
            else if (hair_orient.at<double>(i) <= 0.0)
                im.at<cv::Vec3d>(i) = gray;
            else
            {
                im.at<cv::Vec3d>(i) = cv::Vec3d{
                    -cos(hair_orient.at<double>(i)) * 0.5f + 0.5f, // [0,1]
                    sin(hair_orient.at<double>(i)) * 0.5f + 0.5f,  // [-1,1]
                    1.0};
            }
        }
        std::cout << "assign colors finished" << std::endl;
    }

    // Function to calculate the median of a vector of data
    double calculateMedian(const cv::Mat &hair_mat)
    {
        std::vector<double> data(hair_mat.total());
        for (size_t i = 0; i < hair_mat.total(); i++)
        {
            data.push_back(hair_mat.at<double>(i));
        }
        
        std::sort(data.begin(), data.end());
        size_t n = data.size();
        if (n % 2 == 0)
            return (data[n / 2 - 1] + data[n / 2]) / 2.0;
        else
            return data[n / 2];
    }

    // Function to replace outliers with the median
    void replaceOutliersWithMedian(std::vector<double> &data, double median, double threshold)
    {
        for (double &value : data)
        {
            double diff = std::abs(value - median);
            if (diff > threshold)
            {
                value = median;
            }
        }
    }

private:
    OrientMap m_orientMap;
};

#endif
