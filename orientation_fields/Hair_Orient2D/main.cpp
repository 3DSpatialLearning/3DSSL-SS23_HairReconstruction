//
//  main.cpp
//
//  Created by Usame on 16/06/23.
//  Copyright (c) 2017 Usame@TUM. All rights reserved.
//

#include <iostream>
#include <stdarg.h>

#include "Orient2D.hpp"
#include <opencv2/opencv.hpp>
#include <dirent.h>
#include <sys/types.h>
#include <sys/stat.h>

#include <fstream>
#include <iostream>
#include <sys/stat.h>

using namespace std;


void save_orient_with_body_image_from_folder(string input_img_folder, string output_orient_img_folder)
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
        cout << "Processing: " << input_img_fn << endl;

        string orient_img_fn = output_orient_img_folder + hair_stem;

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

    string out_orient_img_folder = hair_folder + "orient_img/";
    mkdir(out_orient_img_folder.data(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);

    cout << "compute orientation map in original size.\n";
    save_orient_with_body_image_from_folder(img_folder, out_orient_img_folder);

    return 0;
}
