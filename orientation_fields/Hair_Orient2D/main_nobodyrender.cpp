//
//  main.cpp
//
//  Created by Yi on 06/28/17.
//  Copyright (c) 2017 Yi@USC. All rights reserved.
//

#include <iostream>
#include <stdarg.h>

#include "Orient2D.hpp"

#include <thread>

#include <dirent.h>
#include <sys/types.h>
#include <sys/stat.h>

#include <fstream>
#include <iostream>
#include <sys/stat.h>

using namespace std;

string intToStrLen5(int i)
{
    string s;
    char t[256];
    snprintf(t, 10, "%05d", i);
    // sprintf_s(t, "%05d", i);
    s = t;
    return s;
}

void save_orient_with_body_image(string orient_image_fn, string body_image_fn, string seg_image_fn, string out_orient_with_body_image_fn)
{
    cv::Mat orient_img = cv::imread(orient_image_fn, cv::IMREAD_UNCHANGED);
    cv::Mat body_img = cv::imread(body_image_fn, cv::IMREAD_GRAYSCALE);
    cv::Mat seg_img = cv::imread(seg_image_fn, cv::IMREAD_GRAYSCALE);
    cv::Mat orient_with_body_png(orient_img.rows, orient_img.cols, CV_8UC3);
    cv::Mat orient_with_body_exr(orient_img.rows, orient_img.cols, CV_32FC3);

    for (int i = 0; i < orient_img.cols; i++)
        for (int j = 0; j < orient_img.rows; j++)
        {
            int is_body = body_img.at<uchar>(i, j);
            int is_hair = seg_img.at<uchar>(i, j);
            orient_with_body_exr.at<cv::Vec3f>(i, j)[0] = orient_img.at<cv::Vec3f>(i, j)[0];
            orient_with_body_exr.at<cv::Vec3f>(i, j)[1] = orient_img.at<cv::Vec3f>(i, j)[1];
            orient_with_body_exr.at<cv::Vec3f>(i, j)[2] = 1.0;

            if (is_hair > 122)
            {
                orient_with_body_exr.at<cv::Vec3f>(i, j)[2] = 1.0;
            }
            if ((is_body > 122) && (is_hair < 122))
            {
                orient_with_body_exr.at<cv::Vec3f>(i, j) = cv::Vec3f(0, 0, 0.5);
            }
            if ((is_body < 122) && (is_hair < 122))
                orient_with_body_exr.at<cv::Vec3f>(i, j) = cv::Vec3f(0, 0, 0);
        }

    cv::imwrite(out_orient_with_body_image_fn + ".exr", orient_with_body_exr);

    cv::Mat orient_with_body_exr2(orient_img.rows, orient_img.cols, CV_32FC3);
    orient_with_body_exr2 = orient_with_body_exr * 255;
    orient_with_body_exr2.convertTo(orient_with_body_png, CV_8UC3);
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
        string hair_name0 = directory->d_name;
        string hair_name = hair_name0.substr(0, hair_name0.size() - 4);
        cout << "Process " << input_img_fn << "\n";

        string body_img_fn = body_img_folder + hair_name + ".png";

        string orient_img_fn = output_orient_img_folder + hair_name + ".exr";

        std::ifstream infile(orient_img_fn);
        if (!infile.good())
            COrient2D orient2D(input_img_fn, orient_img_fn);

        string hairseg_fn = hair_seg_folder + hair_name + ".png";
        string orient_img_with_body_fn = output_orient_img_with_body_folder + hair_name;

        cout << "save orient with body image.\n";
        cout << hairseg_fn << "\n";
        cout << orient_img_with_body_fn << "\n";
        cout << body_img_fn << "\n";
        cout << orient_img_fn << "\n";
        save_orient_with_body_image(orient_img_fn, body_img_fn, hairseg_fn, orient_img_with_body_fn);
    }
}

cv::Vec3f get_sample(std::vector<cv::Vec3f> samples, std::vector<float> weights)
{
    int body_num = 0;
    int hair_num = 0;
    int background_num = 0;

    for (int i = 0; i < samples.size(); i++)
    {
        if (samples[i][2] == 1.0)
            hair_num++;
        if (samples[i][2] == 0.5)
            body_num++;
        if (samples[i][2] == 0)
            background_num++;
    }

    if ((background_num >= body_num) && (background_num >= hair_num))
        return cv::Vec3f(0, 0, 0);
    if ((body_num >= background_num) && (body_num >= hair_num))
        return cv::Vec3f(0, 0, 0.5);

    // calculate avg
    cv::Vec2f avg_o(0, 0);
    int weights_sum = 0;
    for (int i = 0; i < samples.size(); i++)
    {
        if (samples[i][2] == 1.0)
        {
            if ((samples[i][0] > 1.0) || (samples[i][0] < 0) || (samples[i][1] > 1.0) || (samples[i][1] < 0))
                cout << "wrong orient: " << samples[i][0] << " " << samples[i][1] << "\n";

            float x = 2.0 * (samples[i][0] - 0.5);
            float y = samples[i][1];

            float norm = sqrt(x * x + y * y);
            if (norm == 0)
                continue;
            x = x / norm;
            y = y / norm;

            // cout<<"gt: "<<x<<" "<<y<<"\n";
            avg_o[0] = avg_o[0] + (x * x - y * y) * weights[i];
            avg_o[1] = avg_o[1] + (2 * x * y) * weights[i];
            weights_sum = weights[i] + weights_sum;
        }
    }

    if (weights_sum == 0)
        return cv::Vec3f(0, 0, 0.5);

    avg_o[0] = avg_o[0] / weights_sum;
    avg_o[1] = avg_o[1] / weights_sum;

    float norm = sqrt(avg_o[0] * avg_o[0] + avg_o[1] * avg_o[1]);
    if (norm == 0)
        return cv::Vec3f(0, 0, 0.5);
    avg_o[0] = avg_o[0] / norm;
    avg_o[1] = avg_o[1] / norm;

    // cout<<"af: "<<avg_o[0]<<" "<<avg_o[1]<<"\n";
    avg_o = avg_o + cv::Vec2f(1, 0);

    norm = sqrt(avg_o[0] * avg_o[0] + avg_o[1] * avg_o[1]);
    if (norm == 0)
        return cv::Vec3f(0, 0, 0.5);
    avg_o[0] = avg_o[0] / norm;
    avg_o[1] = avg_o[1] / norm;

    if (avg_o[1] < 0)
    {
        avg_o[1] = avg_o[1] * -1.0;
        avg_o[0] = avg_o[0] * -1.0;
    }

    // cout<<"af: "<<avg_o[0]<<" "<<avg_o[1]<<"\n";
    return cv::Vec3f(avg_o[0] * 0.5 + 0.5, avg_o[1], 1.0);

    float min_dis = 12345;
    cv::Vec2f min_o(0, 0);
    for (int i = 0; i < samples.size(); i++)
    {
        if (samples[i][2] == 1.0)
        {
            cv::Vec2f dis_v = cv::Vec2f(samples[i][0], samples[i][1]) - avg_o;
            double dis = (dis_v[0] * dis_v[0] * dis_v[1] * dis_v[1]);
            if (dis < min_dis)
            {
                min_dis = dis;
                min_o = cv::Vec2f(samples[i][0], samples[i][1]);
            }
        }
    }
    return cv::Vec3f(avg_o[0], avg_o[1], 1.0);
}

int main(int argc, char **argv)
{

    if (argc < 3)
    {
        cout << "Need three argv. bool has_body_imgs; string hair_folder;";
        return 1;
    }

    string hair_folder = argv[2];

    string img_folder = hair_folder + "img/";
    string hair_seg_folder = hair_folder + "seg/";
    string out_body_img_folder = hair_folder + "body_img/";
    string out_orient_img_folder = hair_folder + "orient_img/";

    mkdir(out_orient_img_folder.data(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    string out_orient_img_with_body_folder = hair_folder + "orient_img_with_body/";
    mkdir(out_orient_img_with_body_folder.data(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);

    save_orient_with_body_image_from_folder(out_body_img_folder, img_folder, out_orient_img_folder, hair_seg_folder,
                                            out_orient_img_with_body_folder);
    return 0;
}
