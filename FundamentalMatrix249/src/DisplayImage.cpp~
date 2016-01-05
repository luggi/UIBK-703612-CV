#include "opencv2/opencv_modules.hpp"
#include <stdio.h>
#include <iostream>

#ifndef HAVE_OPENCV_NONFREE

int main(int, char**)
{
    printf("The sample requires nonfree module that is not available in your OpenCV distribution.\n");
    return -1;
}

#else

# include "opencv2/core/core.hpp"
# include "opencv2/features2d/features2d.hpp"
# include "opencv2/highgui/highgui.hpp"
# include "opencv2/nonfree/features2d.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/calib3d/calib3d.hpp"

using namespace cv;
using namespace std;

void readme();

/**
 * @function main
 * @brief Main function
 */

class Settings
{
public:
    void write(FileStorage& fs) const                        //Write serialization for this class
    {
        fs << "{" << "p11"  << p11
        		  << "p12"  << p12
                  << "p13" << p13
                  << "p21"         << p21
                  << "p22" << p22
                  << "p23" << p23
                  << "p31" << p31
                  << "p32" << p32
                  << "p33" << p33
                  << "Write_outputFileName"  << outputFileName
           << "}";
    }
    void read(const FileNode& node)                          //Read serialization for this class
    {
    	node["p11"] >> p11;
        node["p12" ] >> p12;
        node["p13"] >> p13;
        node["p21"] >> p21;
        node["p22"]  >> p22;
        node["p23"] >> p23;
        node["p31"] >> p31;
        node["p32"] >> p32;
        node["p33"] >> p33;
        node["Write_outputFileName"] >> outputFileName;

    }

public:
    float p11;
    float p12;            // The size of the board -> Number of items by width and height
    float p13;// One of the Chessboard, circles, or asymmetric circle pattern
    float p21;          // The size of a square in your defined unit (point, millimeter,etc).
    float p22;              // The number of frames to use from the input for calibration
    float p23;         // The aspect ratio
    float p31;                 // In case of a video input
    float p32;         //  Write detected feature points
    float p33;     // Write extrinsic parameters
    string outputFileName;      // The name of the file where to write
};

static void read(const FileNode& node, Settings& x, const Settings& default_value = Settings())
{
    if(node.empty())
        x = default_value;
    else
        x.read(node);
}


int main( int argc, char** argv )
{
  if( argc != 3 )
  { readme(); return -1; }

  Mat img_1 = imread( argv[1], CV_LOAD_IMAGE_COLOR );
  Mat img_2 = imread( argv[2], CV_LOAD_IMAGE_COLOR );

  if( !img_1.data || !img_2.data )
  { printf(" --(!) Error reading images \n"); return -1; }

  //-- Step 1: Detect the keypoints using SURF Detector
  int minHessian = 400;

  SurfFeatureDetector detector( minHessian );

  std::vector<KeyPoint> keypoints_1, keypoints_2;

  detector.detect( img_1, keypoints_1 );
  detector.detect( img_2, keypoints_2 );

  //-- Step 2: Calculate descriptors (feature vectors)
  SurfDescriptorExtractor extractor;

  Mat descriptors_1, descriptors_2;

  extractor.compute( img_1, keypoints_1, descriptors_1 );
  extractor.compute( img_2, keypoints_2, descriptors_2 );

  //-- Step 3: Matching descriptor vectors using FLANN matcher
  FlannBasedMatcher matcher;
  std::vector< DMatch > matches;
  matcher.match( descriptors_1, descriptors_2, matches );


  // CALCULATE FUNDAMENTAL MATRIX////////////
  vector<Point2f>imgpts1,imgpts2;
  for( unsigned int i = 0; i<matches.size(); i++ )
  {
      // queryIdx is the "left" image
      imgpts1.push_back(keypoints_1[matches[i].queryIdx].pt);
      // trainIdx is the "right" image
      imgpts2.push_back(keypoints_2[matches[i].trainIdx].pt);
  }



  Settings s;
  const string inputSettingsFile = "cameraIntrinsic.xml";
  FileStorage fs(inputSettingsFile, FileStorage::READ); // Read the settings
      if (!fs.isOpened())
      {
          cout << "Could not open the configuration file: \"" << inputSettingsFile << "\"" << endl;
          return -1;
      }
      fs["Settings"] >> s;
      fs.release();                                         // close Settings file

   Mat f_mask;
   Mat F =  findFundamentalMat (imgpts1, imgpts2, FM_RANSAC, 0.5, 0.99, f_mask);
   F.convertTo(F, CV_32F);
   float data[9] = {s.p11,s.p12,s.p13,s.p21,s.p22,s.p23,s.p31,s.p32,s.p33};
   Mat K = Mat(3,3, CV_32F,data);
   Mat E= (K.t()*F)*K;
   //Mat E=K.t().mul(F).mul(K);

   cout << "E= "<< endl << " " << E << endl << endl;

  double max_dist = 0; double min_dist = 100;

  //-- Quick calculation of max and min distances between keypoints
  for( int i = 0; i < descriptors_1.rows; i++ )
  { double dist = matches[i].distance;
    if( dist < min_dist ) min_dist = dist;
    if( dist > max_dist ) max_dist = dist;
  }

  //-- Draw only "good" matches (i.e. whose distance is less than 2*min_dist,
  //-- or a small arbitary value ( 0.02 ) in the event that min_dist is very
  //-- small)
  //-- PS.- radiusMatch can also be used here.
  std::vector< DMatch > good_matches;

  for( int i = 0; i < descriptors_1.rows; i++ )
  { if( matches[i].distance <= max(4*min_dist, 0.02) )
    { good_matches.push_back( matches[i]); }
  }

  //-- Draw only "good" matches
  Mat img_matches;
  drawMatches( img_1, keypoints_1, img_2, keypoints_2,
               good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
               vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

  //-- Show detected matches
  imshow( "Good Matches", img_matches );

  for( int i = 0; i < (int)good_matches.size(); i++ )
  { printf( "-- Good Match [%d] Keypoint 1: %d  -- Keypoint 2: %d  \n", i, good_matches[i].queryIdx, good_matches[i].trainIdx ); }

  waitKey(0);



  return 0;
}

/**
 * @function readme
 */
void readme()
{ printf(" Usage: ./SURF_FlannMatcher <img1> <img2>\n"); }

#endif
