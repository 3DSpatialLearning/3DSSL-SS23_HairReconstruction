#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <cmath>

using namespace std;
using namespace cv;


#define M_PI 3.14159265358979323846
#define NUM_VIEW 16
#define HEIGHT 1604
#define WIDTH 1100
#define WINDOW_HEIGHT 3
#define WINDOW_WIDTH 10
#define STEP_SIZE 0.01f

struct Points {
    cv::Vec3f position;
    cv::Vec3f direction;
    uint8_t r;
    uint8_t g;
    uint8_t b;

    bool operator==(const Points& other) const {
        // Compare the position and direction of two Points objects
        return position == other.position && direction == other.direction;
    }
};

typedef vector<Points> strand;

struct StrandComparator {
    bool operator()(const strand& a, const strand& b) const {
        if (a.size() != b.size()) {
            return a.size() > b.size();
        }
        // Compare the individual points within the strands
        for (size_t i = 0; i < a.size(); i++) {
            if (a[i].position != b[i].position || a[i].direction != b[i].direction) {
                return a[i].position[0] < b[i].position[0];
            }
        }
        // If all points are equal, consider the strands as equal
        return false;
    }
};

typedef set<strand, StrandComparator> strand_set;

cv::Mat ReadIntrinsic(const std::string& filepath) {
    cv::Mat intrinsic(3, 3, CV_32F);
    vector<cv::Mat_<float>> R(NUM_VIEW);
    vector<cv::Mat_<float>> t(NUM_VIEW);
    vector<cv::Mat_<float>> P(NUM_VIEW); //P is the extrinsic matrix
    for (int i = 0; i < NUM_VIEW; i++) {
        R[i] = cv::Mat_<float>(3, 3);
        t[i] = cv::Mat_<float>(3, 1);
        P[i] = cv::Mat_<float>(3, 4);
    }
    string cam_num;
    ifstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Unable to open the file.");
    }
    int num;
    file >> num;
    for (int i = 0; i < 16; i++) {
        file >> cam_num;
        //read the intrinsic matrix
        for (int j = 0; j < 3; j++) {
            for (int k = 0; k < 3; k++) {
                file >> intrinsic.at<float>(j, k);
            }
        }

        //read the rotation matrix in extrinsic
        for (int m = 0; m < 3; m++) {
            for (int n = 0; n < 3; n++) {
                file >> R[i].at<float>(m, n);
            }
        }

        //read the translation matrix
        for (int a = 0; a < 3; a++) {
            file >> t[i].at<float>(a, 0);
        }
        cv::hconcat(R[i], t[i], P[i]);
    }
    return intrinsic / 2;
}

vector<cv::Mat_<float>> ReadExtrinsic(const std::string& filepath) {
    cv::Mat intrinsic(3, 3, CV_32F);
    vector<cv::Mat_<float>> R(NUM_VIEW);
    vector<cv::Mat_<float>> t(NUM_VIEW);
    vector<cv::Mat_<float>> P(NUM_VIEW); //P is the extrinsic matrix
    for (int i = 0; i < NUM_VIEW; i++) {
        R[i] = cv::Mat_<float>(3, 3);
        t[i] = cv::Mat_<float>(3, 1);
        P[i] = cv::Mat_<float>(3, 4);
    }
    string cam_num;
    ifstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Unable to open the file.");
    }
    int num;
    file >> num;
    for (int i = 0; i < 16; i++) {
        file >> cam_num;
        //read the intrinsic matrix
        for (int j = 0; j < 3; j++) {
            for (int k = 0; k < 3; k++) {
                file >> intrinsic.at<float>(j, k);
            }
        }

        //read the rotation matrix in extrinsic
        for (int m = 0; m < 3; m++) {
            for (int n = 0; n < 3; n++) {
                file >> R[i].at<float>(m, n);
            }
        }

        //read the translation matrix
        for (int a = 0; a < 3; a++) {
            file >> t[i].at<float>(a, 0);
        }
        cv::hconcat(R[i], t[i], P[i]);
    }
    return P;
}

//read the orientation map
cv::Mat ReadOrientationMap(const std::string& filePath) {
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
        //cout << "The width is " << width << "\n";
        file.read(reinterpret_cast<char*>(&height), sizeof(int));
        //cout << "the height is " << height << "\n";

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

//Get the projection of the strands' endpoint (position) onto each view
cv::Vec2f ProjectPointontoView(cv::Vec3f start_point, cv::Mat intrinsic, cv::Mat extrinsic) {
    cv::Vec4f start_point_homog(start_point[0], start_point[1], start_point[2], 1);
    cv::Mat start_point_cam = extrinsic * start_point_homog;
    cv::Mat start_point_img = intrinsic * start_point_cam;
    cv::Vec2f point_2d(start_point_img.at<float>(0, 0) / start_point_img.at<float>(2, 0), start_point_img.at<float>(1, 0) / start_point_img.at<float>(2, 0));
    return point_2d;
}

//Get the projection of the strands' direction onto each view
cv::Vec2f ProjectStrandontoView(cv::Vec3f start_point, cv::Vec3f end_point, cv::Mat intrinsic, cv::Mat extrinsic) {
    cv::Vec4f start_point_homog(start_point[0], start_point[1], start_point[2], 1);
    cv::Vec4f end_point_homog(end_point[0], end_point[1], end_point[2], 1);

    cv::Mat start_point_cam = extrinsic * start_point_homog;
    cv::Mat end_point_cam = extrinsic * end_point_homog;

    cv::Mat start_point_img = intrinsic * start_point_cam;
    cv::Mat end_point_img = intrinsic * end_point_cam;
   
    cv::Mat direction = start_point_img - end_point_img;
    cv::Vec2f direction_2d = { direction.at<float>(0, 0) / direction.at<float>(2, 0), direction.at<float>(1, 0) / direction.at<float>(2, 0) };

    cv::normalize(direction_2d, direction_2d);
    return direction_2d;
}


vector<cv::Vec2f> Create2DConeDirections(const cv::Vec2f& center, const cv::Vec2f& orig_direction, float opening_angle_degrees, float angular_resolution_degrees) {

    // Calculate the number of samples based on angular resolution
    int numSamples = static_cast<int>(opening_angle_degrees / angular_resolution_degrees) + 1;
    
    //calculate the angle of the original direction (center)
    float orig_direction_degree = (std::atan2(orig_direction[1], orig_direction[0])) * 180.0 / M_PI;

    //all sampled directions
    std::vector<cv::Vec2f> sampled_directions(numSamples);
    sampled_directions[0] = orig_direction;

    //Iterate over each sample direction
    for (int i = 1; i < numSamples / 2; ++i) {
        // Calculate the angle for the current sample
        float direction_positive = orig_direction_degree + i * angular_resolution_degrees;
        float direction_negative = orig_direction_degree - i * angular_resolution_degrees;

        // Calculate the direction vector for the current angle
        cv::Vec2f direction_positive_vector(std::cos(direction_positive), std::sin(direction_positive));
        cv::Vec2f direction_negative_vector(std::cos(direction_negative), std::cos(direction_negative));

        // Add the direction vector to the list
        sampled_directions.push_back(direction_positive_vector);
        sampled_directions.push_back(direction_negative_vector);
    }

    return sampled_directions;
}

float Angular_Difference_Radiens(cv::Vec2f& direction, float& orientation) {
    float direction_rad = std::atan2(direction[1], direction[0]);
    float difference_rad = abs(direction_rad - orientation);
    return difference_rad;
}

float Angular_Difference_Degree(cv::Vec2f& direction, float& orientation) {
    float direction_degree = std::atan2(direction[1], direction[0]) * 180.0 / M_PI;
    float orientation_degree = orientation * 180 / M_PI;
    float difference_degree = abs(direction_degree - orientation_degree);
    return difference_degree;
}

//from the center of the cone we can create a window
vector<float> CreateWindowGetScore(cv::Vec2f& position, vector<cv::Vec2f>& direction, cv::Mat& orientation_value, cv::Mat& confidence_value, const int width, const int height) {
    int num_samples = direction.size();
    cv::Mat window(height, width, CV_32F);
    vector<float> total_difference(num_samples);

    //calculate the average of the differences
    for (int n = 0; n < num_samples; n++) {
        int num_valid_pixels = 0;
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                float difference = 0.0;
                int x = int(position[0] + i); //current position
                int y = int(position[1] + j);
                if (x >= HEIGHT || y >= WIDTH) {
                    difference = 0.0;
                }
                else {
                    difference = Angular_Difference_Radiens(direction[n], orientation_value.at<float>(x, y));
                    num_valid_pixels += 1;
                }
                total_difference[n] += difference;
            }
        }
        total_difference[n] /= num_valid_pixels;
    }


    return total_difference;
}

//input: all the samples directions  and the scores for all samples
//output: the optimal 2d direction with lowest score
cv::Vec2f Get2dDirectionwithLowestScore(vector<cv::Vec2f> directions, vector<float> scores) {
    int index = 0;
    for (int i = 0; i < scores.size(); i++) {
        if (scores[i] < scores[index]) {
            index = i;
        }
    }
    return directions[index];
}

//get the normal vector of plane defined by the the camera center and the optimal d2d direction
//input: the end point of the projected strand and the optimal 2d direction
//output: the cross product of the 2d direction and a line defined by the end point and the camera center
cv::Mat GetNormalVector(cv::Vec2f position, cv::Vec2f direction, cv::Mat& intrinsic) {
    cv::Vec3f position_homog(position[0], position[1], 1.0);
    cv::Vec3f direction_homog(direction[0], direction[1], 1.0);
    cv::Mat intrinsic_inv(3, 3, CV_32F);
    cv::invert(intrinsic, intrinsic_inv);
    cv::Mat position_cam = intrinsic_inv * position_homog;
    cv::Mat direction_cam = intrinsic_inv * direction_homog;
    cv::Mat normal = position_cam.cross(direction_cam);
    cv::normalize(normal, normal);
    return normal;
}


vector<cv::Mat> ProjectNormalintoWorld(vector<cv::Mat>& normal, vector<cv::Mat_<float>> extrinsic) {
    int num = normal.size();
    vector<cv::Mat_<float>> extrinsic_all(NUM_VIEW, cv::Mat::eye(4, 4, CV_32F));
    vector<cv::Mat_<float>> extrinsic_inv(NUM_VIEW, cv::Mat(4, 4, CV_32F));

    for (int m = 0; m < NUM_VIEW; m++) {
        extrinsic[m].copyTo(extrinsic_all[m](cv::Rect(0, 0, extrinsic[m].cols, extrinsic[m].rows)));
        cv::invert(extrinsic_all[m], extrinsic_inv[m]);
    }

    vector<cv::Mat> normal_world(num);
    for (int i = 0; i < num; i++) {
        normal_world[i] = cv::Mat(3, 1, CV_32F);
        cv::Vec4f normal_camera_homog(normal[i].at<float>(0, 0), normal[i].at<float>(1, 0), normal[i].at<float>(2, 0), 1.0);
        cv::Mat normal_world_matrix = extrinsic_inv[i] * normal_camera_homog;
       
        normal_world[i].at<float>(0, 0) = normal_world_matrix.at<float>(0, 0);
        normal_world[i].at<float>(1, 0) = normal_world_matrix.at<float>(1, 0);
        normal_world[i].at<float>(2, 0) = normal_world_matrix.at<float>(2, 0);

    }
    return normal_world;
}

cv::Mat GetGrowingDirection(std::vector<cv::Mat> normals) {
    float residual = 1.0;
    cv::Mat H = normals[0];
    for (int i = 1; i < NUM_VIEW; i++) {
        cv::vconcat(H, normals[i], H);
    }
    cout << "H is " << H << "\n";
    cv::Mat g;
    for (int iteration = 0; iteration < 2; iteration++) {
        H *= residual;
        cv::SVD svd(H);
        cv::Mat singularValues = svd.w;
        cv::Mat leftSingularVectors = svd.u;
        cv::Mat rightSingularVectors = svd.vt.t();

        cv::Mat sortedIndices;
        cv::sortIdx(singularValues, sortedIndices, cv::SORT_EVERY_COLUMN + cv::SORT_ASCENDING);
        int smallestIndex = sortedIndices.at<int>(0);

        g = rightSingularVectors.col(smallestIndex);
        cv::Mat residual_mat = H * g;

        residual = 1 / (cv::norm(residual_mat) * cv::norm(residual_mat));
    }

    return g;
}

strand HairElongation(const strand& short_strand, const cv::Mat& growing_direction, const float& step_size, const cv::Mat& intrinsic, const vector<cv::Mat_<float>> extrinsic) {
    strand long_strand = short_strand;
    int strand_size = short_strand.size();
    int n = 0;
    int nonvalid_view = 0;

    //stop elongation if the newly added points lay outside every view
    while (nonvalid_view < NUM_VIEW)
    {
        //add the point at the end of short hair strand along the 3d growing direction
        Points new_point;
        new_point.position[0] = short_strand[1].position[0] + n * step_size * growing_direction.at<float>(0, 0);
        new_point.position[1] = short_strand[1].position[1] + n * step_size * growing_direction.at<float>(1, 0);
        new_point.position[2] = short_strand[1].position[2] + n * step_size * growing_direction.at<float>(2, 0);
        nonvalid_view = 0;

        //project the newly added 3d point onto each view
        for (int i = 0; i < NUM_VIEW; i++) {
            cv::Vec2f projected_point = ProjectPointontoView(new_point.position, intrinsic, extrinsic[i]);
            //if the projected point lay outside of the view -> nonvalid view
            //if the number of nonvalid views = 16, stop elongation
            if (abs(projected_point[0]) >= HEIGHT || abs(projected_point[1]) >= WIDTH) {
                nonvalid_view++;
            }
        }
        if (nonvalid_view < NUM_VIEW) {
            long_strand.push_back(new_point);
        }

        n++;
    }
    cout << "finish\n";

    return long_strand;
}

void WriteStrandPLYFile(const std::string& filename, const strand_set& strandSet) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return;
    }

    int vertexSize = 0;
    for (const auto& strand : strandSet) {
        vertexSize += strand.size();
    }

    // Write the PLY header
    file << "ply\n";
    file << "format binary_little_endian 1.0\n";
    file << "element vertex " << vertexSize << "\n";
    file << "property float x\n";
    file << "property float y\n";
    file << "property float z\n";
    file << "property uchar red\n";
    file << "property uchar green\n";
    file << "property uchar blue\n";
    file << "end_header\n";

    // Write the vertex data
    for (const auto& strand : strandSet) {
        for (const auto& point : strand) {
            file.write(reinterpret_cast<const char*>(&point.position[0]), sizeof(point.position[0]));
            file.write(reinterpret_cast<const char*>(&point.position[1]), sizeof(point.position[1]));
            file.write(reinterpret_cast<const char*>(&point.position[2]), sizeof(point.position[2]));
            file.write(reinterpret_cast<const char*>(&point.r), sizeof(point.r));
            file.write(reinterpret_cast<const char*>(&point.g), sizeof(point.g));
            file.write(reinterpret_cast<const char*>(&point.b), sizeof(point.b));
        }
    }



    file.close();

    std::cout << "PLY file written: " << filename << std::endl;
}

int main() {

    //get the cam parameters
    cv::Mat intrinsic = ReadIntrinsic("cam_par.txt"); 
    vector<cv::Mat_<float>> P = ReadExtrinsic("cam_par.txt");
   
    //get the orientation map
    //read all the orientation paths
    vector<string> orientation_paths(NUM_VIEW);
    vector<string> confidence_paths(NUM_VIEW);
    for (int f = 0; f < NUM_VIEW; f++) {
        orientation_paths[f] = "image" + to_string(f + 1) + "_finalOrient.flo";
        confidence_paths[f] = "image" + to_string(f + 1) + "_finalConfidence.flo";
    }

    //read all the orientation and confidence maps in double
    vector<cv::Mat> orientation_all(NUM_VIEW, cv::Mat(HEIGHT, WIDTH, CV_64F));
    vector<cv::Mat> confidence_all(NUM_VIEW, cv::Mat(HEIGHT, WIDTH, CV_64F));

    //the orientation and confidence maps in float
    vector<cv::Mat> floatorientation_all(NUM_VIEW, cv::Mat(HEIGHT, WIDTH, CV_32F));
    vector<cv::Mat> floatconfidence_all(NUM_VIEW, cv::Mat(HEIGHT, WIDTH, CV_32F));

    //get all the orientation and confidence maps in float
    for (int g = 0; g < NUM_VIEW; g++) {
        orientation_all[g] = ReadOrientationMap(orientation_paths[g]);
        confidence_all[g] = ReadOrientationMap(confidence_paths[g]);
        //convert the matrices from double to float
        orientation_all[g].convertTo(floatorientation_all[g], CV_32F);
        confidence_all[g].convertTo(floatconfidence_all[g], CV_32F);
    }
    cout << "Read the orientation maps successfully\n";

    //the 2d projection of the strand on each view
    vector<cv::Vec2f> strand_projections(NUM_VIEW);

    //the sample 2d projections
    vector<vector<cv::Vec2f>> strand_projections_samples(16, vector<cv::Vec2f>(7));

    //create a window of each 2d direction sample and compute the scores of each sample (window)
    vector<vector<float>> scores(NUM_VIEW, vector<float>(16));

    //the best 2d direction of each view
    vector<cv::Vec2f> final_2d_directions(NUM_VIEW);

    //thr normal vectors of the plane defined by each camera's center and optimal 2d direction
    vector<cv::Mat> normals(NUM_VIEW);

    //test sample (short strand contains 2 points), can be changed to any vector
    cv::Vec3f start_point(0.9, 9.9, 49.9);
    cv::Vec3f end_point(1.0, 10.0, 50.0);
   
    for (int i = 0; i < NUM_VIEW; i++) {

        //Get the projection of the strands' endpoint (position) onto each view
        cv::Vec2f sample_center = ProjectPointontoView(end_point, intrinsic, P[i]);
        cout << "The end point is " << sample_center << "\n";
       
        //Get the projection of the strands' direction onto each view
        strand_projections[i] = ProjectStrandontoView(start_point, end_point, intrinsic, P[i]);
        cout << "The strand projection is " << strand_projections[i] << "\n";

        //Get the 2d direction samples centered at the projection of the end point
        strand_projections_samples[i] = Create2DConeDirections(sample_center, strand_projections[i], 6.0, 1.0);
        int num_samples = strand_projections_samples[i].size();

        //compute the score of each window (samples direction) and select the direction with the lowest score as the final 2d direction
        vector<float> sample_scores = CreateWindowGetScore(sample_center, strand_projections_samples[i], floatorientation_all[i], floatconfidence_all[i], WINDOW_WIDTH, WINDOW_HEIGHT);
        final_2d_directions[i] = Get2dDirectionwithLowestScore(strand_projections_samples[i], sample_scores);
        cout << "The optimal 2D direction at " << i << "th view is " << final_2d_directions[i] << "\n";

        //Get the normal vector of the plane defined by the 2d direction and camera center
        normals[i] = GetNormalVector(sample_center, final_2d_directions[i], intrinsic);
        cout << "The normal vector at " << i << "th plane is " << normals[i] << "\n";
    }
    
    //the previous normal vectors are in camera center, but the 3d growing direction is in world space, 
    //so we transform the normal into world space
    vector<cv::Mat> normals_world = ProjectNormalintoWorld(normals, P);
    //get the transpose to construct the H matrix
    vector<cv::Mat> normals_trans(NUM_VIEW);
    for (int h = 0; h < NUM_VIEW; h++) {
        //normals_trans[h] = cv::Mat(1, 3, CV_32F);
        cv::transpose(normals_world[h], normals_trans[h]);
        //cout << "The inverse of normal is " << normals_trans[h] << "\n";
        cout << "The world normal in " << h << "th view is " << normals_trans[h] << "\n";
    }

    //Get the growing direction
    cv::Mat growing_direction = GetGrowingDirection(normals_trans);
    cout << "The final growing direction is " << growing_direction << endl;
    
    //test on the short_strand set
    strand short_strand;
    strand long_strand;
    strand_set short_strands;
    strand_set long_strands;

    //define one test strand
    Points p1, p2;
    p1.position = (0.9, 9.9, 49.9);
    short_strand.push_back(p1);
    p2.position = (1, 10, 50);
    short_strand.push_back(p2);
    short_strands.insert(short_strand);


    //elongate strand
    for (const auto& short_strand : short_strands) {
        // Iterate over the elements in the vector
        strand long_strand = HairElongation(short_strand, growing_direction, STEP_SIZE, intrinsic, P);
        long_strands.insert(long_strand);
    }

    WriteStrandPLYFile("short_hair_strand.ply", short_strands);
    WriteStrandPLYFile("long_hair_strands.ply", long_strands);

    return 0;
}



