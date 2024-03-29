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
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace std;

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

void readme();
vector<Mat> fourfold_Ambiguity(Mat F, Settings s,vector<Point2f>goodPoints1,vector<Point2f>goodPoints2);
void drawCube(Mat& image, int x, int y,int z ,int length, Mat cameraMatrix,vector<Mat> RotationTranslation);
void drawCube2(Mat& image, int x, int y,int z ,int length, Mat cameraMatrix,vector<Mat> RotationTranslation);

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

  /////////////////////////////////// Read the settings/////////////////////
  Settings s;
  const string inputSettingsFile = "cameraIntrinsic.xml";
  FileStorage fs(inputSettingsFile, FileStorage::READ);
      if (!fs.isOpened())
      {
          cout << "Could not open the configuration file: \"" << inputSettingsFile << "\"" << endl;
          return -1;
      }
      fs["Settings"] >> s;
      fs.release();                                         // close Settings file




  /////////////////////////////////-- Step 0: Get images///////////////////////////////////////////////////
  Mat img_1 = imread( argv[1], CV_LOAD_IMAGE_COLOR );
  Mat img_2 = imread( argv[2], CV_LOAD_IMAGE_COLOR );

  if( !img_1.data || !img_2.data )
  { printf(" --(!) Error reading images \n"); return -1; }

  ///////////////////////////////-- Step 1: Detect the keypoints using SURF Detector////////////////////////////
  int minHessian = 800;

  SurfFeatureDetector detector(minHessian);

  std::vector<KeyPoint> keypoints_1, keypoints_2;

  detector.detect( img_1, keypoints_1 );
  detector.detect( img_2, keypoints_2 );

  ///////////////////////////////-- Step 2: Calculate descriptors (feature vectors)/////////////////////////////
  SurfDescriptorExtractor extractor;

  Mat descriptors_1, descriptors_2;

  extractor.compute( img_1, keypoints_1, descriptors_1 );
  extractor.compute( img_2, keypoints_2, descriptors_2 );

  ///////////////////////////////-- Step 3: Matching descriptor vectors using FLANN matcher////////////////////
  FlannBasedMatcher matcher;
  std::vector< DMatch > matches;
  matcher.match( descriptors_1, descriptors_2, matches );


  ///////////////////////////////-- Step 4: Get coordinates of matches in both images//////////////////////////
  vector<Point2f>imgpts1,imgpts2;
  for( unsigned int i = 0; i<matches.size(); i++ )
  {
      // queryIdx is the "left" image
      imgpts1.push_back(keypoints_1[matches[i].queryIdx].pt);
      // trainIdx is the "right" image
      imgpts2.push_back(keypoints_2[matches[i].trainIdx].pt);
  }


  ///////////////////////////////-- Step 5: Quick calculation max and min distances between keypoints//////////
  double max_dist = 0; double min_dist = 100;
  for( int i = 0; i < descriptors_1.rows; i++ )
        { double dist = matches[i].distance;
          if( dist < min_dist ) min_dist = dist;
          if( dist > max_dist ) max_dist = dist;
  }

  ///////////////////////////////-- Step 6: Filter only good matches according to the distance/////////////////
    std::vector< DMatch > good_matches,good_matches_filtered;
      for( int i = 0; i < descriptors_1.rows; i++ )
      { if( matches[i].distance <= max(3*min_dist, 0.02) )
        {
    	  good_matches.push_back( matches[i]);
        }
      }


  ///////////////////////////////-- Step 7: Cast good_matches to new image points/////////////////////////////
  vector<Point2f>imgpts3,imgpts4;
        for( unsigned int i = 0; i<good_matches.size(); i++ )
        {
            // queryIdx is the "left" image
            imgpts3.push_back(keypoints_1[good_matches[i].queryIdx].pt);
            // trainIdx is the "right" image
            imgpts4.push_back(keypoints_2[good_matches[i].trainIdx].pt);
        }

   ////////////////////////////-- Step 8: Feed good image points to Ransac/////////////////////////////////////
   Mat f_mask;
   Mat F =  findFundamentalMat (imgpts3, imgpts4, FM_RANSAC, 0.5, 0.99, f_mask);


   ////////////////////////////-- Step 9: Filter good matches again - now according to RANSAC /////////////////

   for( int i = 0; i < (int)good_matches.size(); i++ )
         {
	   	   if( f_mask.at<int>(i,0) == 1 )
           {
	   		   good_matches_filtered.push_back( good_matches[i]);
           }
         }
   cout << good_matches_filtered.size() << endl;

  ////////////////////////////-- Step 9: Draw only "best" matches /////////////////////////////////////////////
  Mat img_matches;
  drawMatches( img_1, keypoints_1, img_2, keypoints_2,
               good_matches_filtered, img_matches, Scalar::all(-1), Scalar::all(-1),
               vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

  //-- Show detected matches
  Mat img_matches_resized;
  resize(img_matches,img_matches_resized,Size(0,0),0.2,0.2,CV_INTER_AREA);

  std::vector<Point2f> goodPoints1,goodPoints2;
  for( int i = 0; i < (int)good_matches_filtered.size(); i++ )
  {
	  printf( "-- Good Match Filtered [%d] Keypoint 1: %d  -- Keypoint 2: %d  \n", i, good_matches_filtered[i].queryIdx, good_matches_filtered[i].trainIdx );
	  goodPoints1.push_back( keypoints_1[good_matches_filtered[i].queryIdx].pt);
	  goodPoints2.push_back( keypoints_2[good_matches_filtered[i].trainIdx].pt);
  }


  vector<Mat> temp = fourfold_Ambiguity(F,s,goodPoints1,goodPoints2);
  Mat projectionMatrix1 = temp[1];
  Mat projectionMatrix2 = temp[2];
  Mat testPoint = temp[0].col(5);

  cout << testPoint << endl;
  Mat projectedPoint1 = Mat_<float>(3,1);
  projectedPoint1 = projectionMatrix1*testPoint;
  Mat projectedPoint2 = Mat_<float>(3,1);
  projectedPoint2 = projectionMatrix2*testPoint;

  Point point1;
  point1.x = projectedPoint1.at<float>(0,0);
  point1.y = projectedPoint1.at<float>(1,0);
  circle(img_1,point1,30, Scalar(0,0,255),CV_FILLED, 8,0);
  Mat img_1_resized;
  resize(img_1,img_1_resized,Size(0,0),0.2,0.2,CV_INTER_AREA);
  imshow( "Img1", img_1_resized);

  Point point2;
  point2.x = projectedPoint2.at<float>(0,0);
  point2.y = projectedPoint2.at<float>(1,0);
  circle(img_2,point2,30, Scalar(0,0,255),CV_FILLED, 8,0);
  Mat img_2_resized;
  resize(img_2,img_2_resized,Size(0,0),0.2,0.2,CV_INTER_AREA);
  imshow( "Img2", img_2_resized);



  float data[9] = {s.p11,s.p12,s.p13,s.p21,s.p22,s.p23,s.p31,s.p32,s.p33};
  Mat K = Mat(3,3, CV_32F,data);
  //drawCube(img_matches_resized,0,0,0,100,K,temp);
  imshow( "Good Matches", img_matches_resized);

  waitKey(0);
  return 0;
}


vector<Mat> fourfold_Ambiguity(Mat F,Settings s,vector<Point2f> goodPoints1,vector<Point2f>goodPoints2)
{
	   F.convertTo(F, CV_32F);
	   float data[9] = {s.p11,s.p12,s.p13,s.p21,s.p22,s.p23,s.p31,s.p32,s.p33};
	   Mat K = Mat(3,3, CV_32F,data);
	   Mat E= (K.t()*F)*K;

	   float data2[12] = {1,0,0,0,0,1,0,0,0,0,1,0};
	   Mat Identity= Mat(3,4, CV_32F,data2);
	   Mat P0=K*Identity;

	   SVD decomp = SVD(E);
	   //Mat U,Vt,W;
	   //decomp.compute(E,W,U,Vt);
	   Mat U = decomp.u;
	   Mat Vt = decomp.vt;
	   Mat W(3, 3, CV_32F, Scalar(0));
	   //cout << "W= "<< endl << " " << W << endl << endl;
	   W.at<float>(0, 1) = -1.0;
	   W.at<float>(1, 0) = 1.0;
	   W.at<float>(2, 2) = 1.0;
	   //cout << "W= "<< endl << " " << W << endl << endl;
	   //cout << "W.t()= "<< endl << " " << W.t() << endl << endl;

	   Mat R0= U * W * Vt;
	   //cout << "R0= "<< endl << " " << R0 << endl << endl;
	   Mat R1= U * W.t() * Vt;
	   //cout << "R1= "<< endl << " " << R1 << endl << endl;
	   Mat t0=U.col(2);
	   //cout << "t0= "<< endl << " " << t0 << endl << endl;
	   Mat t1=-1*U.col(2);
	   //cout << "t1= "<< endl << " " << t1 << endl << endl;

	   Mat P11,P12,P13,P14;
	   hconcat(R0, t0, P11);
	   P11=K*P11;
	   hconcat(R1, t0, P12);
	   P12=K*P12;
	   hconcat(R0, t1, P13);
	   P13=K*P13;
	   hconcat(R1, t1, P14);
	   P14=K*P14;

	   Mat pnts3D;
	   triangulatePoints(P0,P11,goodPoints1,goodPoints2,pnts3D);
	 //cout << "goodPoints2= "<< endl << " " << goodPoints2 << endl << endl;

	 pnts3D=P11*pnts3D;
	 //cout << "P11= "<< endl << " " << P11 << endl << endl;
	 //cout << "pnts3D= "<< endl << " " << pnts3D << endl << endl;
	 triangulatePoints(P0,P12,goodPoints1,goodPoints2,pnts3D);
	 pnts3D=P12*pnts3D;
	 //cout << "P12= "<< endl << " " << P12 << endl << endl;
	 //cout << "pnts3D= "<< endl << " " << pnts3D << endl << endl;
	 triangulatePoints(P0,P13,goodPoints1,goodPoints2,pnts3D);
	 pnts3D=P13*pnts3D;
	 //cout << "P13= "<< endl << " " << P13 << endl << endl;
	 //cout << "pnts3D= "<< endl << " " << pnts3D << endl << endl;
	 triangulatePoints(P0,P14,goodPoints1,goodPoints2,pnts3D);
	 vector<Mat> toReturn;
	 toReturn.push_back(pnts3D);
	 pnts3D=P14*pnts3D;
	 //cout << "P14= "<< endl << " " << P14 << endl << endl;
	 //cout << "pnts3D= "<< endl << " " << pnts3D << endl << endl;

	 //toReturn.push_back(R1);
	 //toReturn.push_back(t1);
	 toReturn.push_back(P0);
	 toReturn.push_back(P14);
	 return toReturn;
}

void drawCube(Mat& image, int x, int y, int z, int length, Mat cameraMatrix, vector<Mat> RotationTranslation){

    vector<Point2d> outputPoints;
    vector<Point2d> distCoeffs;
	vector<Point3d> points;
	points.push_back(Point3d(x,y,z));
	points.push_back(Point3d(x+length,y,z));
	points.push_back(Point3d(x,y,z+length));
	points.push_back(Point3d(x+length,y,z+length));
	points.push_back(Point3d(x,y+length,z));
	points.push_back(Point3d(x+length,y+length,z));
	points.push_back(Point3d(x,y+length,z+length));
	points.push_back(Point3d(x+length,y+length,z+length));

	Mat onePoint = Mat_<float>(4,1);

	onePoint.at<float>(0,0)=x;
	onePoint.at<float>(1,0)=y;
	onePoint.at<float>(2,0)=z;
	onePoint.at<float>(3,0)=1;
	cout << onePoint << endl;
	cout << RotationTranslation[0] << endl;
	Mat projectedPoint = Mat_<float>(3,1);

	projectedPoint = RotationTranslation[0]*onePoint;
	cout << projectedPoint << endl;

	//projectPoints(points,rvec,tvec,cameraMatrix,distCoeffs,outputPoints);
}

/*void drawCube2(Mat& image, int x, int y, int z, int length, Mat cameraMatrix, vector<Mat> RotationTranslation){
    Mat distPar = Mat_<double>(1,5);
	vector<Point3d> points;
	points.push_back(Point3d(x,y,z));
	points.push_back(Point3d(x+length,y,z));
	points.push_back(Point3d(x,y,z+length));
	points.push_back(Point3d(x+length,y,z+length));
	points.push_back(Point3d(x,y+length,z));
	points.push_back(Point3d(x+length,y+length,z));
	points.push_back(Point3d(x,y+length,z+length));
	points.push_back(Point3d(x+length,y+length,z+length));

	Mat matPoints = Mat_<Point3d>(1,points.size()) << points[0],points[1],points[2],points[3],points[4],points[5],points[6],points[7];
	Mat outputPoints = Mat_<Point3d>(1,points.size());

	projectPoints(matPoints,RotationTranslation[0],RotationTranslation[1],cameraMatrix,distPar,outpuPoints);
	//cout << "aa" << endl;
	//cout << outputPoints << endl;
}

void drawCube(Mat& image, int x, int y, int z, int length, Mat cameraMatrix, vector<Mat> RotationTranslation){
    vector<Point2f> outputPoints;
    vector<Point2f> distPar;
	vector<Point3f> points;
	points.push_back(Point3f(x,y,z));
	points.push_back(Point3f(x+length,y,z));
	points.push_back(Point3f(x,y,z+length));
	points.push_back(Point3f(x+length,y,z+length));
	points.push_back(Point3f(x,y+length,z));
	points.push_back(Point3f(x+length,y+length,z));
	points.push_back(Point3f(x,y+length,z+length));
	points.push_back(Point3f(x+length,y+length,z+length));

	/*vector<float> rvec,tvec;
	for(int i=0; i<RotationTranslation[1].rows;i++){
		tvec.push_back(RotationTranslation[1].at<float>(i,0));
	}
	Rodrigues(RotationTranslation[0],rvec);*/

    // Decompose the projection matrix into:
	/*Mat K(3,3,cv::DataType<float>::type); // intrinsic parameter matrix
    Mat rvec(3,3,cv::DataType<float>::type); // rotation matrix
    Mat T(4,1,cv::DataType<float>::type); // translation vector

    decomposeProjectionMatrix(RotationTranslation[0], K, rvec, T);

    // Create zero distortion
    Mat distCoeffs(4,1,cv::DataType<float>::type);
    distCoeffs.at<float>(0) = 0;
    distCoeffs.at<float>(1) = 0;
    distCoeffs.at<float>(2) = 0;
    distCoeffs.at<float>(3) = 0;

	///decomposeProjectionMatrix(RotationTranslation[0], cameraMatrix, rvec, tvec);
	projectPoints(points.front(),rvec,T,K,distCoeffs,outputPoints);
}
*/


void readme()
{ printf(" Usage: ./SURF_FlannMatcher <img1> <img2>\n"); }

#endif






