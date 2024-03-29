# 3DSSL-SS23_HairReconstruction

This repository contains the implementation of the paper: Giljoo Nam, Chenglei Wu, Min Kim, and Yaser Sheikh.
[Strand-accurate multi-view hair capture](https://openaccess.thecvf.com/content_CVPR_2019/papers/Nam_Strand-Accurate_Multi-View_Hair_Capture_CVPR_2019_paper.pdf). pages 155–164, 06, 2019.

Note that the last part of the pipeline suggested in the paper - Hair Growing, is not implemented.

This implementation was based on the repositories:
- [gipuma](https://github.com/kysucix/gipuma)
- [fuisible](https://github.com/kysucix/fusibile)

The code in the folders `/gipuma` and `/fuisible` was modified to perform Line-based PatchMach Multi-view Stereo as proposed by the Strand-accurate multi-view hair capture paper.

## Generate Orientation Fields and Confidence Values
### Setup
- Ensure you have OpenCV installed on your system.
- Navigate to the Hair_Orient2D directory.
- Create a build directory and compile the program using CMake:

```sh
mkdir build
cd build
cmake ..
make
```
### Usage
Run the program by providing the path to the input image folder:

```sh
./Orient2D <input_image_folder>
```
Replace `<input_image_folder>` with the path to the folder containing the images you want to process.
## Running Line-based PatchMatch MVS
### Some important files:
-   `gipuma/gipuma.cu`: File containing cuda code for running LPMVS - Estimates 3D line map for one view
-   `fuisible/fuisible.cu`: File containing cuda code for merging all 3D line per-view maps into one. 
-   `gipuma/scripts/97_frame_00005.sh`: Shell script file for executing gipuma for each view and merge the results by running fuisible

### Setup
Use cmake to compile `fusible` and `gipuma` projects:

1. ```cd fuisible```
2. `cmake . -D CMAKE_C_COMPILER=/usr/bin/gcc-7`
3. `make`
4. `cd ../gipuma`
5. `cmake . -D CMAKE_C_COMPILER=/usr/bin/gcc-7`
6. `make`

### Data
1. Create a new folder `data/my_dataset` inside the `gipuma` folder
2. Add a subfolder `images` containing the multiview images and the camera parameters following the KRT format
3. [Optional] Add another subfolder for `masks` contining the hair segmentation masks. `masks/cam_img1.png` should correspond to `images/cam_img1.png`, ...
4. [Optional] Add another subfolder for `depth` contining the initial depth values. `depth/cam_img1.dat` should correspond to `images/cam_img1.png`, ...
5. Add another subfolder for `orientation_fields` contining the generated orientation fields. `orientation_fields/cam_img_1_finalOrient.flo` should correspond to `images/cam_img1.png`, ...
6. Add another subfolder for `confidence_values` contining the generated orientation fields. `confidence_values/cam_img_1_finalVariance.flo` should correspond to `images/cam_img1.png`, ...
7. Add a script `my_dataset.sh` in the `/scripts` folder to run the Line-based PatchMatch MVS. For an example see: `scrips/97_frame_00005.sh`


The final file structure should look like:
```
├── gipuma
|   ├── ...
|   ├── data
|   |   ├── my_dataset
|   |   |   ├── images
|   |   |   |   ├── cam_img_1.png
|   |   |   |   ├── cam_img_2.png
|   |   |   |   ├── ...
|   |   |   |   ├── cam_img_N.png
|   |   |   |   ├── cam_par.txt
|   |   |   ├── depth                             # Optional folder
|   |   |   |   ├── cam_img_1.png
|   |   |   |   ├── cam_img_2.png
|   |   |   |   ├── ...
|   |   |   |   ├── cam_img_N.png
|   |   |   ├── masks                             # Optional folder
|   |   |   |   ├── cam_img_1.png
|   |   |   |   ├── cam_img_2.png
|   |   |   |   ├── ...
|   |   |   |   ├── cam_img_N.png
|   |   |   ├── orientation_fields
|   |   |   |   ├── cam_img_1_finalOrient.flo
|   |   |   |   ├── cam_img_2_finalOrient.flo
|   |   |   |   ├── ...
|   |   |   |   ├── cam_img_N_finalOrient.flo
|   |   |   ├── confidence_values
|   |   |   |   ├── cam_img_1_finalVariance.flo
|   |   |   |   ├── cam_img_2_finalVariance.flo
|   |   |   |   ├── ...
|   |   |   |   ├── cam_img_N_finalVariance.flo
|   ├── ...
|   ├── scripts
|   |   ├── 97_frame_00005.sh
|   |   ├── my_dataset.sh
|   |   ├── ...
|   ├── ...
├── ...
```

Finally, run the following command:
`./my_dataset.sh`

The final result will be stored in `gipuma/results/{output_dir_basename}/consistencyCheck-.../line_cloud.dat`

## Running Line Fusion

-   `CMakeLists.txt`: The CMake build configuration file for compiling the program.
-   `lib/LineFusion.hpp`: Header file containing the `LineFusion` class for performing line fusion on the noisy point cloud.
-   `pointclouds/`: Directory containing example point cloud files.
-   `src/LineFusion.cpp`: Source code file implementing the `LineFusion` class.
-   `src/main.cpp`: Main source code file implementing the program's main functionality.

### Usage

1.  Ensure that you have the Point Cloud Library (PCL) and Eigen3 libraries installed on your system.
        
2.  Navigate to the repository's root directory.
    
3.  Create a build directory and compile the program using CMake:
    ```bash: 
    mkdir build
    cd build
    cmake ..
    make 
    ```
    
4.  Run the program by providing the input and output filenames as command-line arguments:
    ```sh 
    ./line_fusion <lpmvs_result> <output_pointcloud_file>
    ``` 
    
    Replace `<lpmvs_result>` with the path to the binary file containing the LPMVS data, and `<output_pointcloud_file>` with the desired path for the fused point cloud output.

## Running Short Hair Generation

-   `Hair_Generation.cpp`: Source code for the hair strands generation.
-   `all_output_strand.ply`: The generation result.

### Usage

1.  Ensure that you have the Point Cloud Library (PCL), Eigen3 and OpenCV installed on your system.
   
2.  Run the program in Visual Studio 2022
    
