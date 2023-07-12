//
//  Orient2D.hpp
//  HairSketch
//
//  Created by Liwen on 12/20/14.
//  Copyright (c) 2014 Liwen Hu. All rights reserved.
//

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
        int npass = 3;

        // Read the input image into colorImg
        cv::Mat colorImg = cv::imread(filename, cv::IMREAD_COLOR);
        cv::Mat filterImg;
        cv::cvtColor(colorImg, filterImg, cv::COLOR_BGRA2GRAY);
        filterImg.convertTo(filterImg, CV_64FC1);
        filterImg /= 255.;
        std::cout << "filter image size: " << filterImg.size() << " " << filterImg.channels() << std::endl;
        std::cout << "color image size: " << colorImg.size() << " " << colorImg.channels() << std::endl;

        cv::Mat m_hairConf = cv::Mat::zeros(filterImg.size(), CV_64FC1);
        cv::Mat m_hairOrient = cv::Mat::zeros(filterImg.size(), CV_64FC1);

        // Compute the mask using the provided sigma_l value
        cv::Mat mask = cv::Mat::zeros(filterImg.size(), CV_64FC1);
        compute_mask_cv(filterImg, sigma_l, mask);

        int npix = filterImg.cols * filterImg.rows;
        cv::Mat interFilter, interOrient, interConf;
        // Perform npass iterations of filtering and manipulation
        for (int i = 0; i < npass; i++)
        {
            cv::normalize(filterImg, interFilter, 0, 255, cv::NORM_MINMAX, CV_8U);
            cv::imwrite(outfilename + "_interFilter" + std::to_string(i) + ".png", interFilter); // mask((outfilename + "_mask.png")); // Write the mask image to file
            cv::normalize(m_hairConf, interConf, 0, 255, cv::NORM_MINMAX, CV_8U);
            cv::imwrite(outfilename + "_interConf" + std::to_string(i) + ".png", interConf);
            cv::normalize(m_hairOrient, interOrient, 0, 255, cv::NORM_MINMAX, CV_8U);
            cv::imwrite(outfilename + "_interOrient" + std::to_string(i) + ".png", interOrient);
            // Apply the filter to the normalized buffer and store the result in the alternate buffer
            filter(filterImg, ndegree, sigma_h, sigma_l, sigma_y, m_hairOrient, m_hairConf);
            // Calculate the maximum magnitude from the filtered buffer
            double max_mag = 0.0;
            for (int j = 0; j < npix; j++)
            {
                double mag = mask.at<double>(j) != 0.0 ? max<double>(m_hairConf.at<double>(j), 0.0) : 0.0;
                filterImg.at<double>(j) = mag;
                m_hairConf.at<double>(j) = mag;
                max_mag = max<double>(max_mag, mag);
            }

            // Normalize the values in the buffer by dividing by the maximum magnitude
            filterImg /= max_mag;

        }
        exportFloatImage(m_hairConf, outfilename + "_finalConf.flo");
        exportFloatImage(m_hairOrient, outfilename + "_finalOrient.flo");

        // // Visualize the orientations in the buffer using a color scheme
        std::cout << "converting color" << std::endl;
        cv::Mat colorizedOrientImg = cv::Mat::zeros(colorImg.size(), CV_64FC3);
        cv::Mat dur;
        std::cout << "converting color finished" << std::endl;

        viz_ori_2color(colorizedOrientImg, m_hairOrient, m_hairConf, mask);
        cv::normalize(colorizedOrientImg, dur, 0, 255, cv::NORM_MINMAX, CV_8U);
        cv::imwrite(outfilename + "_orientColorized.png", dur);


        // Diffuse confidence results
        cv::Mat filterHairConf;
        cv::GaussianBlur(m_hairConf, filterHairConf, cv::Size(21, 21), 0.);
        double maxConf = 0.;
        for (int hI = 0; hI < m_hairConf.rows; hI++)
        {
            for (int wI = 0; wI < m_hairConf.cols; wI++)
            {
                if (mask.at<double>(hI, wI) != 0.0)
                    maxConf = MAX(maxConf, filterHairConf.at<float>(hI, wI));
            }
        }
        for (int hI = 0; hI < m_hairConf.rows; hI++)
        {
            for (int wI = 0; wI < m_hairConf.cols; wI++)
            {
                m_hairConf.at<float>(hI, wI) /= maxConf;
                if (mask.at<double>(hI, wI) == 0.0)
                    m_hairConf.at<float>(hI, wI) = 0.;
            }
        }

        viz_ori_2color(colorizedOrientImg, m_hairOrient, filterHairConf, mask);
        cv::normalize(colorizedOrientImg, dur, 0, 255, cv::NORM_MINMAX, CV_8U);
        cv::imwrite(outfilename + "_orientColorizedNew.png", dur);
        cv::normalize(m_hairConf, m_hairConf, 0, 255, cv::NORM_MINMAX, CV_8U);
        cv::imwrite(outfilename + "_finalizedConf.png", m_hairConf);
        std::cout << m_hairOrient.at<double>(0) << std::endl;
        
        exportFloatImage(filterHairConf, outfilename + "_finalConfidence.flo");
        cv::normalize(filterHairConf, filterHairConf, 0, 255, cv::NORM_MINMAX, CV_8U);
        cv::imwrite(outfilename + "_finalizedConf.png", filterHairConf);
    }

    ~COrient2D() {}

    void exportFloatImage(const cv::Mat& floatImage, const std::string& filePath) {
        // Open the file for writing as a binary file
        std::ofstream file(filePath, std::ios::binary);

        if (!file.is_open()) {
            std::cerr << "Error opening file: " << filePath << std::endl;
            return;
        }

        try {
            // Write width and height to the file
            int width = floatImage.cols;
            int height = floatImage.rows;

            file.write(reinterpret_cast<const char*>(&width), sizeof(int));
            file.write(reinterpret_cast<const char*>(&height), sizeof(int));

            // Write the pixel values
            file.write(reinterpret_cast<const char*>(floatImage.ptr()), floatImage.total() * sizeof(double));

            // Close the file
            file.close();
            std::cout << "Float image exported successfully: " << filePath << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Error exporting float image: " << e.what() << std::endl;
        }
    }

    cv::Mat importFloatImage(const std::string& filePath, cv::Mat& floatImage) {
        // Open the file for reading as a binary file
        std::ifstream file(filePath, std::ios::binary);

        if (!file.is_open()) {
            std::cerr << "Error opening file: " << filePath << std::endl;
            return cv::Mat();
        }

        try {
            // Read the width and height from the file
            int width, height;
            file.read(reinterpret_cast<char*>(&width), sizeof(int));
            file.read(reinterpret_cast<char*>(&height), sizeof(int));
            // Create a cv::Mat object for the float image

            // Read the pixel values from the file
            file.read(reinterpret_cast<char*>(floatImage.ptr()), floatImage.total() * sizeof(double));

            // Close the file
            file.close();

            std::cout << "Float image imported successfully: " << filePath << std::endl;
            return floatImage;
        } catch (const std::exception& e) {
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

    // Build a single level in the pyramid
    void filter_dense(const fftw_complex *imfft, int w, int h,
                      double sigma_h, double sigma_l, double sigma_y,
                      int orientations, cv::Mat &hair_orient, cv::Mat &hair_conf)
    {
        // Clear and resize the output image
        hair_conf = cv::Mat::zeros(cv::Size(w, h), CV_64FC1);
		hair_orient = cv::Mat::zeros(cv::Size(w, h), CV_64FC1);

        int npix = w * h;

        // Allocate memory for intermediate storage
        fftw_complex *filtfft = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * (w / 2 + 1) * h);
        double *filtered = (double *)fftw_malloc(sizeof(double) * w * h);

        // Create a plan for inverse FFT
        fftw_plan idft = fftw_plan_dft_c2r_2d(h, w, filtfft, filtered, FFTW_ESTIMATE);

        for (int i = 0; i < orientations; i++)
        {
            // Calculate the angle for the current orientation
            double angle = M_PI * i / orientations;

            // Apply the Mexican Hat wavelet filter to the input image
            mexican_hat(imfft, w, h, sigma_h, sigma_l, sigma_y, angle, filtfft);

            // Perform inverse FFT
            fftw_execute(idft);

            // Update maximum response in the output image without confidence computation
            for (int j = 0; j < npix; j++)
            {
                double res = filtered[j];

                if (std::abs(hair_conf.at<double>(j)) < std::abs(res))
                {
                    hair_orient.at<double>(j) = angle;
                    hair_conf.at<double>(j) = res;
                }
            }
        }

        // Clean up resources
        fftw_destroy_plan(idft);
        fftw_free(filtered);
        fftw_free(filtfft);
    }

    void filter(
        const cv::Mat &im, int orientations,
        double sigma_h, double sigma_l, double sigma_y, cv::Mat &hair_orient, cv::Mat &hair_conf)
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
            ffttmp[j] = fftscale * curr_im.at<double>(j / w, j % w); // Scale pixel values --> curr_im GRAY value

        fftw_execute(imdft);

        // Apply dense pyramid filter to transformed data
        filter_dense(imfft, w, h, sigma_h, sigma_l, sigma_y, orientations, hair_orient, hair_conf);

        fftw_destroy_plan(imdft);

        fftw_free(imfft);
        fftw_free(ffttmp);
    }
    // Computes a binary mask based on the input image using gaussian filter in Fourier domain
    void compute_mask(const Im &im, double sigma, Im &out)
    {
        const float fg_impact_thresh = 0.9999f;
        int w = im.w, h = im.h, npix = w * h;

        // Allocate memory for intermediate storage
        double *ffttmp = (double *)fftw_malloc(sizeof(double) * w * h);
        fftw_complex *imfft = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * (w / 2 + 1) * h);

        // Create plans for forward and inverse FFT
        fftw_plan imdft = fftw_plan_dft_r2c_2d(h, w, ffttmp, imfft, FFTW_ESTIMATE);
        fftw_plan imidft = fftw_plan_dft_c2r_2d(h, w, imfft, ffttmp, FFTW_ESTIMATE);

        double fftscale = 1.0 / (w * h);

        // Compute the Fourier transform of the input image
        for (int j = 0; j < npix; j++)
            ffttmp[j] = im[j].sum() ? fftscale : 0.0;

        fftw_execute(imdft);

        // Apply a Gaussian filter in the Fourier domain
        gaussian(imfft, w, h, sigma, imfft);

        fftw_execute(imidft);

        // Resize the output mask
        out.resize(w, h);

        // Generate a binary mask based on the threshold
        for (int j = 0; j < npix; j++)
        {
            out[j] = ffttmp[j] >= fg_impact_thresh ? Color(1.0) : Color(0.0);
        }

        // Clean up resources
        fftw_destroy_plan(imidft);
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
            if (hair_conf.at<double>(i) == 0.0f)
            {
                im.at<cv::Vec3d>(i) = black;
            }
            else if (hair_orient.at<double>(i) <= 0.0f)
            {
                im.at<cv::Vec3d>(i) = gray;
            }
            else
            {
                im.at<cv::Vec3d>(i) = cv::Vec3d{
                    -cos(hair_orient.at<double>(i)) * 0.5f + 0.5f, // [0,1]
                    sin(hair_orient.at<double>(i)),                // [-1,1]
                    1.0};                    // magnitude of response
                // im[i] = Color(
                //         cos(2.0f * im[i][0]) * im[i][1] * scale * 0.5f + 0.5f,
                //         sin(2.0f * im[i][0]) * im[i][1] * scale * 0.5f + 0.5f,
                //         0.5f);
            }
        }
        std::cout << "assign colors finished" << std::endl;
    }

private:
    OrientMap m_orientMap;
};

#endif
