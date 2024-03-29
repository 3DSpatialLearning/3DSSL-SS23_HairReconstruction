/*
 * utility functions for visualization of results (disparity in color, warped output, ...)
 */

#pragma once
#include <sstream>
#include <fstream>

#if (CV_MAJOR_VERSION ==2)
#include <opencv2/contrib/contrib.hpp> // needed for applyColorMap!
#endif

#include "point_cloud.h"
#include "point_cloud_list.h"

/* compute gamma correction (just for display purposes to see more details in farther away areas of disparity image)
 * Input: img   - image
 *        gamma - gamma value
 * Output: gamma corrected image
 */
Mat correctGamma( Mat& img, double gamma ) {
 double inverse_gamma = 1.0 / gamma;

 Mat lut_matrix(1, 256, CV_8UC1 );
 uchar * ptr = lut_matrix.ptr();
 for( int i = 0; i < 256; i++ )
   ptr[i] = (int)( pow( (double) i / 255.0, inverse_gamma ) * 255.0 );

 Mat result;
 LUT( img, lut_matrix, result );

 return result;
}


static void getDisparityForDisplay(const Mat_<float> &disp, Mat &dispGray, Mat &dispColor, float numDisparities, float minDisp = 0.0f){
	float gamma = 2.0f; // to get higher contrast for lower disparity range (just for color visualization)
	disp.convertTo(dispGray,CV_16U,65535.f/(numDisparities-minDisp),-minDisp*65535.f/(numDisparities-minDisp));
	Mat disp8;
	disp.convertTo(disp8,CV_8U,255.f/(numDisparities-minDisp),-minDisp*255.f/(numDisparities-minDisp));
	if(minDisp == 0.0f)
		disp8 = correctGamma(disp8,gamma);
	applyColorMap(disp8, dispColor, COLORMAP_JET);
	for(int y = 0; y < dispColor.rows; y++){
		for(int x = 0; x < dispColor.cols; x++){
			if(disp(y,x) <= 0.0f)
				dispColor.at<Vec3b>(y,x) = Vec3b(0,0,0);
		}
	}
}

static void convertDisparityDepthImage(const Mat_<float> &dispL, Mat_<float> &d, float f, float baseline){
	d = Mat::zeros(dispL.rows, dispL.cols, CV_32F);
	for(int y = 0; y < dispL.rows; y++){
		for(int x = 0; x < dispL.cols; x++){
			d(y,x) = disparityDepthConversion(f,baseline,dispL(y,x));
		}
	}
}

static string getColorString(uint8_t color){
	stringstream ss;
	ss << (int)color << " " << (int)color << " " << (int)color;
	return ss.str();
}


static string getColorString(Vec3b color){
	stringstream ss;
	ss << (int)color(2) << " " << (int)color(1) << " " << (int)color(0);
	return ss.str();
}

static string getColorString(Vec3i color){
	stringstream ss;
	ss << (int)((float)color(2)/256.f) << " " << (int)((float)color(1)/256.f) << " " << (int)((float)color(0)/256.f);
	return ss.str();
}
static void storePlyFileBinaryPointCloud (char* plyFilePath, PointCloudList &pc, Mat_<float> &distImg) {
    cout << "store 3D points to ply file" << endl;

    FILE *outputPly;
    outputPly=fopen(plyFilePath,"wb");

    /*write header*/
    fprintf(outputPly, "ply\n");
    fprintf(outputPly, "format binary_little_endian 1.0\n");
    fprintf(outputPly, "element vertex %d\n",pc.size);
    fprintf(outputPly, "property float x\n");
    fprintf(outputPly, "property float y\n");
    fprintf(outputPly, "property float z\n");
    fprintf(outputPly, "property float nx\n");
    fprintf(outputPly, "property float ny\n");
    fprintf(outputPly, "property float nz\n");
    fprintf(outputPly, "property uchar red\n");
    fprintf(outputPly, "property uchar green\n");
    fprintf(outputPly, "property uchar blue\n");
    fprintf(outputPly, "end_header\n");

    distImg = Mat::zeros(pc.rows,pc.cols,CV_32F);

    //write data
#pragma omp parallel for
    for(long int i = 0; i < pc.size; i++) {
        const Point_li &p = pc.points[i];
        const float4 normal = p.normal;
        float4 X = p.coord;
        const char color = (int)p.texture;
        /*const int color = 127.0f;*/
        /*printf("Writing point %f %f %f\n", X.x, X.y, X.z);*/

        if(!(X.x < FLT_MAX && X.x > -FLT_MAX) || !(X.y < FLT_MAX && X.y > -FLT_MAX) || !(X.z < FLT_MAX && X.z >= -FLT_MAX)){
            X.x = 0.0f;
            X.y = 0.0f;
            X.z = 0.0f;
        }
#pragma omp critical
        {
            /*myfile << X.x << " " << X.y << " " << X.z << " " << normal.x << " " << normal.y << " " << normal.z << " " << color << " " << color << " " << color << endl;*/
            fwrite(&X.x,      sizeof(X.x), 1, outputPly);
            fwrite(&X.y,      sizeof(X.y), 1, outputPly);
            fwrite(&X.z,      sizeof(X.z), 1, outputPly);
            fwrite(&normal.x, sizeof(normal.x), 1, outputPly);
            fwrite(&normal.y, sizeof(normal.y), 1, outputPly);
            fwrite(&normal.z, sizeof(normal.z), 1, outputPly);
            fwrite(&color,  sizeof(char), 1, outputPly);
            fwrite(&color,  sizeof(char), 1, outputPly);
            fwrite(&color,  sizeof(char), 1, outputPly);
        }

    }
    fclose(outputPly);
}

static void storePointCloudBinaryWithDirections(char* filePath, PointCloudList &pc) {
    cout << "store 3D points and directions to file" << endl;

    FILE *outputFile;
    outputFile=fopen(filePath,"wb");

    const long int pointsCount = pc.size;

    fwrite(&pointsCount, sizeof(pointsCount), 1, outputFile);

    for(long int i = 0; i < pc.size; i++) {
        const Point_li &p = pc.points[i];
        const float4 direction = p.normal;
        float4 position = p.coord;
        fwrite(&position.x, sizeof(position.x), 1, outputFile);
        fwrite(&position.y, sizeof(position.y), 1, outputFile);
        fwrite(&position.z, sizeof(position.z), 1, outputFile);

        fwrite(&direction.x, sizeof(direction.x), 1, outputFile);
        fwrite(&direction.y, sizeof(direction.y), 1, outputFile);
        fwrite(&direction.z, sizeof(direction.z), 1, outputFile);
    }

    fclose(outputFile);
}


static void readPointCloudBinaryWithDirections(char* filePath) {
    cout << "read 3D points and directions to file" << endl;
    std::ifstream file(filePath, std::ios::binary);

    long int pointsCount;
    file.read(reinterpret_cast<char *>(&pointsCount), sizeof(long int));

    cout << "points count: " << pointsCount << endl;

    std::vector<float> data(6 * pointsCount);
    file.read(reinterpret_cast<char*>(&data[0]), 6 * pointsCount*sizeof(float));
    for(size_t i = 0; i < data.size(); i++) {
        std::cout << data[i] << "\n";
    }

    file.close();
}

static void storePlyFileBinaryPointCloudWithDirections(char* plyFilePath,char* plyDirectionFilePath, PointCloudList &pc) {
    cout << "store 3D points to ply file" << endl;

    FILE *outputPly;
    outputPly=fopen(plyFilePath,"wb");

    /*write header*/
    fprintf(outputPly, "ply\n");
    fprintf(outputPly, "format binary_little_endian 1.0\n");
    fprintf(outputPly, "element vertex %d\n",pc.size);
    fprintf(outputPly, "property float x\n");
    fprintf(outputPly, "property float y\n");
    fprintf(outputPly, "property float z\n");
    fprintf(outputPly, "property float nx\n");
    fprintf(outputPly, "property float ny\n");
    fprintf(outputPly, "property float nz\n");
    fprintf(outputPly, "property uchar red\n");
    fprintf(outputPly, "property uchar green\n");
    fprintf(outputPly, "property uchar blue\n");
    fprintf(outputPly, "end_header\n");

    //write data
#pragma omp parallel for
    for(long int i = 0; i < pc.size; i++) {
        const Point_li &p = pc.points[i];
        const float4 normal = p.normal;
        float4 X = p.coord;
        const char color = (int)p.texture;
        /*const int color = 127.0f;*/
        /*printf("Writing point %f %f %f\n", X.x, X.y, X.z);*/

        if(!(X.x < FLT_MAX && X.x > -FLT_MAX) || !(X.y < FLT_MAX && X.y > -FLT_MAX) || !(X.z < FLT_MAX && X.z >= -FLT_MAX)){
            X.x = 0.0f;
            X.y = 0.0f;
            X.z = 0.0f;
        }
#pragma omp critical
        {
            /*myfile << X.x << " " << X.y << " " << X.z << " " << normal.x << " " << normal.y << " " << normal.z << " " << color << " " << color << " " << color << endl;*/
            fwrite(&X.x,      sizeof(X.x), 1, outputPly);
            fwrite(&X.y,      sizeof(X.y), 1, outputPly);
            fwrite(&X.z,      sizeof(X.z), 1, outputPly);
            fwrite(&normal.x, sizeof(normal.x), 1, outputPly);
            fwrite(&normal.y, sizeof(normal.y), 1, outputPly);
            fwrite(&normal.z, sizeof(normal.z), 1, outputPly);
            fwrite(&color,  sizeof(char), 1, outputPly);
            fwrite(&color,  sizeof(char), 1, outputPly);
            fwrite(&color,  sizeof(char), 1, outputPly);
        }

    }
    fclose(outputPly);



    FILE *outputDirectionsPly;
    outputDirectionsPly = fopen(plyDirectionFilePath, "wb");
    int samplesAlongDirection = 4;

    // write header
    fprintf(outputDirectionsPly, "ply\n");
    fprintf(outputDirectionsPly, "format binary_little_endian 1.0\n");
    fprintf(outputPly, "element vertex %d\n",pc.size * samplesAlongDirection);
    fprintf(outputDirectionsPly, "property float x\n");
    fprintf(outputDirectionsPly, "property float y\n");
    fprintf(outputDirectionsPly, "property float z\n");
    fprintf(outputDirectionsPly, "property uchar red\n");
    fprintf(outputDirectionsPly, "property uchar green\n");
    fprintf(outputDirectionsPly, "property uchar blue\n");
    fprintf(outputDirectionsPly, "end_header\n");

    char redC = 255;
    char greenC = 0;
    char blueC = 0;


#pragma omp parallel for
    for(long int i = 0; i < pc.size; i++) {
        const Point_li &p = pc.points[i];
        const float4 direction = p.normal;
        float4 X = p.coord;


        if(!(X.x < FLT_MAX && X.x > -FLT_MAX) || !(X.y < FLT_MAX && X.y > -FLT_MAX) || !(X.z < FLT_MAX && X.z >= -FLT_MAX)){
            X.x = 0.0f;
            X.y = 0.0f;
            X.z = 0.0f;
        }
            
            for(int sampleIdx = 1; sampleIdx <= samplesAlongDirection; sampleIdx++) {
                float3 ptX2;
                ptX2.x = X.x + + std::pow(-1, sampleIdx) * (0.00001f) * sampleIdx * direction.x;
                ptX2.y = X.y + + std::pow(-1, sampleIdx) * (0.00001f) * sampleIdx * direction.y;
                ptX2.z = X.z + + std::pow(-1, sampleIdx) * (0.00001f) * sampleIdx * direction.z;
#pragma omp critical
            {
                fwrite(&ptX2.x,      sizeof(ptX2.x), 1, outputPly);
                fwrite(&ptX2.y,      sizeof(ptX2.y), 1, outputPly);
                fwrite(&ptX2.z,      sizeof(ptX2.z), 1, outputPly);
                fwrite(&redC, sizeof(redC), 1, outputDirectionsPly);
                fwrite(&greenC, sizeof(greenC), 1, outputDirectionsPly);
                fwrite(&blueC, sizeof(blueC), 1, outputDirectionsPly);
            }
        }
        
    }

    fclose(outputDirectionsPly);
}

static void storeXYZPointCloud (char* plyFilePath, PointCloudList &pc) {
    cout << "store 3D points to ply file" << endl;

	ofstream myfile;
	myfile.open (plyFilePath, ios::out);

    //write data
#pragma omp parallel for
    for(long i = 0; i < pc.size; i++) {
        const Point_li &p = pc.points[i];
        const float4 normal = p.normal;
        float4 X = p.coord;

        if(!(X.x < FLT_MAX && X.x > -FLT_MAX) || !(X.y < FLT_MAX && X.y > -FLT_MAX) || !(X.z < FLT_MAX && X.z >= -FLT_MAX)){
            X.x = 0.0f;
            X.y = 0.0f;
            X.z = 0.0f;
        }
#pragma omp critical
        {
            myfile << X.x << " " << X.y << " " << X.z << " " << normal.x << " " << normal.y << " " << normal.z << endl;
        }

    }
	myfile.close();
}
static void storePlyFileAsciiPointCloud (char* plyFilePath, PointCloudList &pc, Mat_<float> &distImg) {
	cout << "store 3D points to ply file" << endl;

    /*FILE *outputPly;*/
    /*outputPly=fopen(plyFilePath,"wb");*/

    /*write header*/
    /*fprintf(outputPly, "ply\n");*/
    /*fprintf(outputPly, "format binary_little_endian 1.0\n");*/
    /*fprintf(outputPly, "element vertex %d\n",count);*/
    /*fprintf(outputPly, "property float x\n");*/
    /*fprintf(outputPly, "property float y\n");*/
    /*fprintf(outputPly, "property float z\n");*/
    /*fprintf(outputPly, "property float nx\n");*/
    /*fprintf(outputPly, "property float ny\n");*/
    /*fprintf(outputPly, "property float nz\n");*/
    /*fprintf(outputPly, "property uchar red\n");*/
    /*fprintf(outputPly, "property uchar green\n");*/
    /*fprintf(outputPly, "property uchar blue\n");*/
    /*fprintf(outputPly, "end_header\n");*/

	ofstream myfile;
	myfile.open (plyFilePath, ios::out);

	//write header
	myfile << "ply" << endl;
	myfile << "format ascii 1.0" << endl;
	myfile << "element vertex " << pc.size << endl;
	myfile << "property float x" << endl;
	myfile << "property float y" << endl;
	myfile << "property float z" << endl;
	myfile << "property float nx" << endl;
	myfile << "property float ny" << endl;
	myfile << "property float nz" << endl;
	myfile << "property uchar red" << endl;
	myfile << "property uchar green" << endl;
	myfile << "property uchar blue" << endl;
	myfile << "end_header" << endl;


    distImg = Mat::zeros(pc.rows,pc.cols,CV_32F);

	//write data
    #pragma omp parallel for
    for(long i = 0; i < pc.size; i++) {
        const Point_li &p = pc.points[i];
        const float4 normal = p.normal;
        float4 X = p.coord;
        const int color = (int)p.texture;
        /*const int color = 127.0f;*/
        /*printf("Writing point %f %f %f\n", X.x, X.y, X.z);*/

        if(!(X.x < FLT_MAX && X.x > -FLT_MAX) || !(X.y < FLT_MAX && X.y > -FLT_MAX) || !(X.z < FLT_MAX && X.z >= -FLT_MAX)){
            X.x = 0.0f;
            X.y = 0.0f;
            X.z = 0.0f;
        }
#pragma omp critical
        {
            myfile << X.x << " " << X.y << " " << X.z << " " << normal.x << " " << normal.y << " " << normal.z << " " << color << " " << color << " " << color << endl;
            /*fwrite(&X.x,      sizeof(float), 1, outputPly);*/
            /*fwrite(&X.y,      sizeof(float), 1, outputPly);*/
            /*fwrite(&X.z,      sizeof(float), 1, outputPly);*/
            /*fwrite(&normal.x, sizeof(float), 1, outputPly);*/
            /*fwrite(&normal.y, sizeof(float), 1, outputPly);*/
            /*fwrite(&normal.z, sizeof(float), 1, outputPly);*/
            /*fwrite(&color,  sizeof(float), 1, outputPly);*/
            /*fwrite(&color,  sizeof(float), 1, outputPly);*/
            /*fwrite(&color,  sizeof(float), 1, outputPly);*/
        }

    }
	myfile.close();
/*fclose(outputPly);*/
}
//template <typename ImgType>
static void storePlyFileBinaryPointCloud(char* plyFilePath, PointCloud &pc, Mat_<float> &distImg) {
    cout << "store 3D points to ply file" << endl;

    FILE *outputPly;
    outputPly=fopen(plyFilePath,"wb");

    /*write header*/
    fprintf(outputPly, "ply\n");
    fprintf(outputPly, "format binary_little_endian 1.0\n");
    fprintf(outputPly, "element vertex %d\n",pc.size);
    fprintf(outputPly, "property float x\n");
    fprintf(outputPly, "property float y\n");
    fprintf(outputPly, "property float z\n");
    /*fprintf(outputPly, "property float nx\n");*/
    /*fprintf(outputPly, "property float ny\n");*/
    /*fprintf(outputPly, "property float nz\n");*/
    /*fprintf(outputPly, "property uchar red\n");*/
    /*fprintf(outputPly, "property uchar green\n");*/
    /*fprintf(outputPly, "property uchar blue\n");*/
    fprintf(outputPly, "end_header\n");


    distImg = Mat::zeros(pc.rows,pc.cols,CV_32F);

    //write data
/*#pragma omp parallel for*/
    for(int i = 0; i < pc.size; i++){
        const Point_cu &p = pc.points[i];
        //const float4 normal = p.normal;
        float4 X = p.coord;
        /*printf("Writing point %f %f %f\n", X.x, X.y, X.z);*/
        /*float color = p.texture;*/
        //const char color = 127.0f;

        if(!(X.x < FLT_MAX && X.x > -FLT_MAX) || !(X.y < FLT_MAX && X.y > -FLT_MAX) || !(X.z < FLT_MAX && X.z >= -FLT_MAX)){
            X.x = 0.0f;
            X.y = 0.0f;
            X.z = 0.0f;
        }
/*#pragma omp critical*/
        {
            /*myfile << X.x << " " << X.y << " " << X.z << " " << normal.x << " " << normal.y << " " << normal.z << " " << color << " " << color << " " << color << endl;*/
            fwrite(&(X.x),      sizeof(float), 1, outputPly);
            fwrite(&(X.y),      sizeof(float), 1, outputPly);
            fwrite(&(X.z),      sizeof(float), 1, outputPly);
            /*fwrite(&(normal.x), sizeof(float), 1, outputPly);*/
            /*fwrite(&(normal.y), sizeof(float), 1, outputPly);*/
            /*fwrite(&(normal.z), sizeof(float), 1, outputPly);*/
            /*fwrite(&color,  sizeof(char), 1, outputPly);*/
            /*fwrite(&color,  sizeof(char), 1, outputPly);*/
            /*fwrite(&color,  sizeof(char), 1, outputPly);*/
        }

        /*distImg(y,x) = sqrt(pow(X.x-cam.C(0),2)+pow(X.y-cam.C(1),2)+pow(X.z-cam.C(2),2));*/
    }

    /*myfile.close();*/
    fclose(outputPly);
}
template <typename ImgType>
static void storePlyFileBinary(char* plyFilePath, const Mat_<float> &depthImg, const Mat_<Vec3f> &normals, const Mat_<ImgType> img, Camera cam, Mat_<float> &distImg){
	cout << "store 3D points to ply file" << endl;

    FILE *outputPly;
    outputPly=fopen(plyFilePath,"wb");

	//write header
    fprintf(outputPly, "ply\n");
    fprintf(outputPly, "format binary_little_endian 1.0\n");
    fprintf(outputPly, "element vertex %d\n",depthImg.rows * depthImg.cols);
    fprintf(outputPly, "property float x\n");
    fprintf(outputPly, "property float y\n");
    fprintf(outputPly, "property float z\n");
    fprintf(outputPly, "property float nx\n");
    fprintf(outputPly, "property float ny\n");
    fprintf(outputPly, "property float nz\n");
    fprintf(outputPly, "property uchar red\n");
    fprintf(outputPly, "property uchar green\n");
    fprintf(outputPly, "property uchar blue\n");
    fprintf(outputPly, "end_header\n");

	distImg = Mat::zeros(depthImg.rows,depthImg.cols,CV_32F);

	//write data
    #pragma omp parallel for
	for(int x = 0; x < depthImg.cols; x++){
		for(int y = 0; y < depthImg.rows; y++){
			Vec3f n = normals(y,x);
			ImgType color = img(y,x);

			Vec3f ptX = get3Dpoint(cam,x,y,depthImg(y,x));

			if(!(ptX(0) < FLT_MAX && ptX(0) > -FLT_MAX) || !(ptX(1) < FLT_MAX && ptX(12) > -FLT_MAX) || !(ptX(2) < FLT_MAX && ptX(2) >= -FLT_MAX)){
				ptX(0) = 0.0f;
				ptX(1) = 0.0f;
				ptX(2) = 0.0f;
			}
            #pragma omp critical
            {
                //myfile << ptX(0) << " " << ptX(1) << " " << ptX(2) << " " << n(0) << " " << n(1) << " " << n(2) << " " << getColorString(color) << endl;
                fwrite(&(ptX(0)), sizeof(float), 3, outputPly);
                fwrite(&(n(0))  , sizeof(float), 3, outputPly);
                fwrite(&color, sizeof(color) , 1,  outputPly);
                fwrite(&color, sizeof(color) , 1,  outputPly);
                fwrite(&color, sizeof(color) , 1,  outputPly);
            }

			distImg(y,x) = sqrt(pow(ptX(0)-cam.C(0),2)+pow(ptX(1)-cam.C(1),2)+pow(ptX(2)-cam.C(2),2));

			//}else{
			//	cout << ptX(0) << " " << ptX(1) << " " << ptX(2) << endl;
			//	cout << depthImg(y,x) << endl;
			//}


			//P *
			//cout << xValue << " " << yValue << " " << zValue << " / " <<
		}
	}

fclose(outputPly);
}

template <typename ImgType>
static void storePlyFile(char* plyFilePath, const Mat_<float> &depthImg, const Mat_<Vec3f> &normals, const Mat_<ImgType> img, Camera cam, Mat_<float> &distImg){
	cout << "store 3D points to ply file" << endl;
	ofstream myfile;
	myfile.open (plyFilePath, ios::out);

	//write header
	myfile << "ply" << endl;
	myfile << "format ascii 1.0" << endl;
	myfile << "element vertex " << depthImg.rows * depthImg.cols << endl;
	myfile << "property float x" << endl;
	myfile << "property float y" << endl;
	myfile << "property float z" << endl;
	myfile << "property float nx" << endl;
	myfile << "property float ny" << endl;
	myfile << "property float nz" << endl;
	myfile << "property uchar red" << endl;
	myfile << "property uchar green" << endl;
	myfile << "property uchar blue" << endl;
	myfile << "end_header" << endl;

	distImg = Mat::zeros(depthImg.rows,depthImg.cols,CV_32F);

	//write data
    //#pragma omp parallel for
	for(int x = 0; x < depthImg.cols; x++){
		for(int y = 0; y < depthImg.rows; y++){
			/*
			float zValue = depthImg(x,y);
			float xValue = ((float)x-cx)*zValue/camParams.f;
			float yValue = ((float)y-cy)*zValue/camParams.f;
			myfile << xValue << " " << yValue << " " << zValue << endl;
			*/

			//Mat_<float> pt = Mat::ones(3,1,CV_32F);
			//pt(0,0) = (float)x;
			//pt(1,0) = (float)y;

			Vec3f n = normals(y,x);
			ImgType color = img(y,x);

			//Mat_<float> ptX = P1_inv * depthImg(y,x)*pt;

			//if(depthImg(y,x) <= 0.0001f)
			//	continue;


			Vec3f ptX = get3Dpoint(cam,x,y,depthImg(y,x));

			//Vec3f ptX_v1 = get3dPointFromPlane(cam.P_inv,cam.C,n,planes.d(y,x),x,y);
			//cout << ptX_v1 << " / " << ptX << endl;

			if(!(ptX(0) < FLT_MAX && ptX(0) > -FLT_MAX) || !(ptX(1) < FLT_MAX && ptX(12) > -FLT_MAX) || !(ptX(2) < FLT_MAX && ptX(2) >= -FLT_MAX)){
				ptX(0) = 0.0f;
				ptX(1) = 0.0f;
				ptX(2) = 0.0f;
			}
            //#pragma omp critical
            {
			myfile << ptX(0) << " " << ptX(1) << " " << ptX(2) << " " << n(0) << " " << n(1) << " " << n(2) << " " << getColorString(color) << endl;
            }

			distImg(y,x) = sqrt(pow(ptX(0)-cam.C(0),2)+pow(ptX(1)-cam.C(1),2)+pow(ptX(2)-cam.C(2),2));

			//}else{
			//	cout << ptX(0) << " " << ptX(1) << " " << ptX(2) << endl;
			//	cout << depthImg(y,x) << endl;
			//}


			//P *
			//cout << xValue << " " << yValue << " " << zValue << " / " <<
		}
	}

	myfile.close();
}



static void getNormalsForDisplay(const Mat &normals, Mat &normals_display, int rtype = CV_16U){
	if(rtype == CV_8U)
		normals.convertTo(normals_display,CV_8U,128,128);
	else
		normals.convertTo(normals_display,CV_16U,32767,32767);
	cvtColor(normals_display,normals_display,COLOR_RGB2BGR);
}
