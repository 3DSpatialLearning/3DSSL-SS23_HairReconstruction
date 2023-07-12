//
//  main.cpp
//
//  Created by Usame on 16/06/23.
//  Copyright (c) 2017 Usame@TUM. All rights reserved.
//

#include <iostream>
#include <stdarg.h>

#include "Orient2D.hpp"

#include <GLFW/glfw3.h>
#include <glm/gtc/quaternion.hpp>
#include <thread>

#include <dirent.h>
#include <sys/types.h>
#include <sys/stat.h>

#include <fstream>
#include <iostream>
#include <sys/stat.h>

using namespace std;

void save_orient_with_body_image(string orient_image_fn, string body_image_fn, string seg_image_fn, string out_orient_with_body_image_fn)
{
    cout << "Reading: " << orient_image_fn << endl;
    cv::Mat orient_img = cv::imread(orient_image_fn, cv::IMREAD_UNCHANGED);
    cout << "Reading: " << body_image_fn << endl;
    cv::Mat body_img = cv::imread(body_image_fn, cv::IMREAD_GRAYSCALE);
    cout << "Reading: " << seg_image_fn << endl;
    cv::Mat seg_img = cv::imread(seg_image_fn, cv::IMREAD_GRAYSCALE);
    cv::Mat orient_with_body_png(orient_img.rows, orient_img.cols, CV_8UC3);
    cv::Mat orient_with_body_exr(orient_img.rows, orient_img.cols, CV_32FC3);

    for (int i = 0; i < orient_img.rows; i++)
    {
        for (int j = 0; j < orient_img.cols; j++)
        {
            int is_body = body_img.at<uchar>(i, j);
            int is_hair = seg_img.at<uchar>(i, j);
            orient_with_body_exr.at<cv::Vec3f>(i, j)[0] = orient_img.at<cv::Vec3f>(i, j)[0];
            orient_with_body_exr.at<cv::Vec3f>(i, j)[1] = orient_img.at<cv::Vec3f>(i, j)[1];
            orient_with_body_exr.at<cv::Vec3f>(i, j)[2] = 1.0;

            if (is_hair > 122)
                orient_with_body_exr.at<cv::Vec3f>(i, j)[2] = 1.0;
            else if (is_body > 122)
                orient_with_body_exr.at<cv::Vec3f>(i, j) = cv::Vec3f(0, 0, 0.5);
            else
                orient_with_body_exr.at<cv::Vec3f>(i, j) = cv::Vec3f(0, 0, 0);
        }
    }

    cv::imwrite(out_orient_with_body_image_fn + ".exr", orient_with_body_exr);

    orient_with_body_exr *= 255;
    orient_with_body_exr.convertTo(orient_with_body_png, CV_8UC3);
    cv::imwrite(out_orient_with_body_image_fn + ".png", orient_with_body_png);
}

void save_orient_with_body_image_from_folder(
    string body_img_folder,
    string input_img_folder, string output_orient_img_folder, string hair_seg_folder, string output_orient_img_with_body_folder)
{
    DIR *dirp;
    struct dirent *directory;

    dirp = opendir(input_img_folder.data());
    if (!dirp)
    {
        cout << "no this input_img folder.\n";
        return;
    }

    while ((directory = readdir(dirp)) != NULL)
    {
        string d_name = directory->d_name;

        if (d_name.length() < 5)
            continue;
        if ((d_name.substr(d_name.size() - 4, d_name.size()) != ".png") && (d_name.substr(d_name.size() - 4, d_name.size()) != ".jpg"))
            continue;

        string input_img_fn = input_img_folder + directory->d_name;
        string hair_name = directory->d_name;
        string hair_stem = hair_name.substr(0, hair_name.size() - 4);
        cout << "---------------------------------------" << endl;
        cout << "Process: " << input_img_fn << endl;
        cout << "Hair Name: " << hair_name << endl;
        cout << "Hair Stem: " << hair_stem << endl;

        string body_img_fn = body_img_folder + hair_stem + ".png";

        string orient_img_fn = output_orient_img_folder + hair_stem;
        cout << "Body Image FN: " << body_img_fn << endl;
        cout << "Orient Image FN: " << orient_img_fn << endl;

        std::ifstream infile(orient_img_fn);
        if (!infile.good())
            COrient2D orient2D(input_img_fn.data(), orient_img_fn);

    }
}


int main(int argc, char **argv)
{

    if (argc < 2)
    {
        cout << "Need two argv. string hair_folder;";
        return 1;
    }

    string hair_folder = argv[1];

    string img_folder = hair_folder + "img/";
    string hair_seg_folder = hair_folder + "seg/";

    string out_body_img_folder = hair_folder + "body_img/";

    string out_orient_img_folder = hair_folder + "orient_img/";
    mkdir(out_orient_img_folder.data(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    string out_orient_img_with_body_folder = hair_folder + "orient_img_with_body/";
    mkdir(out_orient_img_with_body_folder.data(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);

    cout << "compute orientation map in original size.\n";
    save_orient_with_body_image_from_folder(out_body_img_folder, img_folder, out_orient_img_folder, hair_seg_folder,
                                            out_orient_img_with_body_folder);

    return 0;
}
