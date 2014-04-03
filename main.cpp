#include <iostream>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iomanip>
using namespace std;
using namespace cv;



/* canny */

void follow(Mat nonmax, Mat &result, int x, int y, int thresMin, int thresMax)
{
    result.at<uchar>(y,x) = 255;

    if( result.at<uchar>(y + 1,x + 1) == 0 &&  nonmax.at<uchar>(y + 1,x + 1) > thresMin && nonmax.at<uchar>(y + 1,x + 1) <= thresMax){
        follow(nonmax, result, x + 1, y + 1, thresMin, thresMax);
    }
    if( result.at<uchar>(y - 1,x - 1) == 0 &&  nonmax.at<uchar>(y - 1,x - 1) > thresMin && nonmax.at<uchar>(y + 1,x + 1) <= thresMax){
        follow(nonmax, result, x - 1, y - 1, thresMin, thresMax);
    }
    if( result.at<uchar>(y, x - 1) == 0 &&  nonmax.at<uchar>(y,x - 1) > thresMin && nonmax.at<uchar>(y + 1,x + 1) <= thresMax){
        follow(nonmax, result, x - 1, y, thresMin, thresMax);
    }
    if( result.at<uchar>(y, x + 1) == 0 &&  nonmax.at<uchar>(y,x + 1) > thresMin && nonmax.at<uchar>(y + 1,x + 1) <= thresMax){
        follow(nonmax, result, x + 1, y, thresMin, thresMax);
    }
    if( result.at<uchar>(y - 1, x) == 0 &&  nonmax.at<uchar>(y - 1,x) > thresMin && nonmax.at<uchar>(y + 1,x + 1) <= thresMax){
        follow(nonmax, result, x, y - 1, thresMin, thresMax);
    }
    if( result.at<uchar>(y + 1, x) == 0 &&  nonmax.at<uchar>(y + 1,x) > thresMin && nonmax.at<uchar>(y + 1,x + 1) <= thresMax){
        follow(nonmax, result, x, y + 1, thresMin, thresMax);
    }
    if( result.at<uchar>(y + 1, x - 1) == 0 &&  nonmax.at<uchar>(y + 1, x - 1) > thresMin && nonmax.at<uchar>(y + 1,x + 1) <= thresMax){
        follow(nonmax, result, x - 1, y + 1, thresMin, thresMax);
    }
    if( result.at<uchar>(y - 1, x + 1) == 0 &&  nonmax.at<uchar>(y - 1, x + 1) > thresMin && nonmax.at<uchar>(y + 1,x + 1) <= thresMax){
        follow(nonmax, result, x + 1, y - 1, thresMin, thresMax);
    }
}

void canny(const Mat src, Mat& result, Mat &direction,  int thresMax, int thresMin, Mat kernelX, Mat kernelY)
{
    int centerKernel = kernelX.cols / 2;
    int cols = src.cols;
    int rows = src.rows;

    Mat mmagnitude =  (Mat(src.rows, src.cols,CV_64FC1, Scalar(0)));

    direction =  (Mat(src.rows, src.cols,CV_8UC1));


    Mat nonmax  =  (Mat(src.rows, src.cols,CV_8UC1, Scalar(0)));

    result = Mat(src.rows, src.cols,CV_8UC1, Scalar(0));

    double sX, sY;

    int ii, jj;
    double dir;

    //compute derivative of filter image

    for(int i = 0; i < cols; ++i){
        for(int j = 0; j < rows; ++j){
            sX = 0;
            sY = 0;

            for(int ik = -centerKernel; ik <= centerKernel; ++ik ){
                ii = i + ik;
                for(int jk = -centerKernel; jk <= centerKernel; ++jk ){
                    jj = j + jk;

                    if(ii >= 0 && ii < cols && jj >= 0 && jj < rows){
                        sX += src.at<uchar>(jj, ii) * kernelX.at<double>(centerKernel + jk, centerKernel + ik);
                        sY += src.at<uchar>(jj, ii) * kernelY.at<double>(centerKernel + jk, centerKernel + ik);
                    }
                }
            }
            dir = (atan2(sX, sY)/M_PI) * 180.0;
            if ( ( (dir < 22.5) && (dir > -22.5) ) || (dir > 157.5) || (dir < -157.5) ){
                dir = 0;
            }
            if ( ( (dir > 22.5) && (dir < 67.5) ) || ( (dir < -112.5) && (dir > -157.5) ) ){
                dir = 45;
            }
            if ( ( (dir > 67.5) && (dir < 112.5) ) || ( (dir < -67.5) && (dir > -112.5) ) ){
                dir = 90;
            }
            if ( ( (dir > 112.5) && (dir < 157.5) ) || ( (dir < -22.5) && (dir > -67.5) ) ){
                dir = 135;
            }
            direction.at<uchar>(j, i) = dir;
            mmagnitude.at<double>(j, i) = sqrt(pow(sX, 2) + pow(sY, 2));
        }
    }

    for(int y = 1; y < rows - 1; ++y){
        for(int x = 1; x < cols - 1; ++x){

            //non-maximal suppression
            switch(direction.at<uchar>(y, x)){
                case 0 :
                    if(mmagnitude.at<double>(y,x) > mmagnitude.at<double>(y + 1,x) && mmagnitude.at<double>(y,x) > mmagnitude.at<double>(y - 1, x)){
                        if(mmagnitude.at<double>(y,x) > thresMax){
                            result.at<uchar>(y,x) = 255;
                        }

                        else if(mmagnitude.at<double>(y,x) <= thresMax && mmagnitude.at<double>(y,x) > thresMin){
                            nonmax.at<uchar>(y,x) = mmagnitude.at<double>(y,x);
                        }
                    }
                    break;
                case 45 :
                    if(mmagnitude.at<double>(y,x) > mmagnitude.at<double>(y + 1,x + 1) && mmagnitude.at<double>(y,x) > mmagnitude.at<double>(y - 1,x  - 1)){
                        if(mmagnitude.at<double>(y,x) > thresMax){
                            result.at<uchar>(y,x) = 255;
                        }

                        else if(mmagnitude.at<double>(y,x) <= thresMax && mmagnitude.at<double>(y,x) > thresMin){
                            nonmax.at<uchar>(y,x) = mmagnitude.at<double>(y,x);
                        }
                    }
                    break;
                case 90 :
                    if(mmagnitude.at<double>(y,x) > mmagnitude.at<double>(y,x - 1) && mmagnitude.at<double>(y,x) > mmagnitude.at<double>(y,x  + 1)){
                        if(mmagnitude.at<double>(y,x) > thresMax){
                            result.at<uchar>(y,x) = 255;
                        }

                        else if(mmagnitude.at<double>(y,x) <= thresMax && mmagnitude.at<double>(y,x) > thresMin){
                            nonmax.at<uchar>(y,x) = mmagnitude.at<double>(y,x);
                        }
                    }
                    break;
                case 135 :
                    if(mmagnitude.at<double>(y,x) > mmagnitude.at<double>(y + 1,x - 1) && mmagnitude.at<double>(y,x) > mmagnitude.at<double>(y - 1,x  + 1)){

                        if(mmagnitude.at<double>(y,x) > thresMax){
                            result.at<uchar>(y,x) = 255;
                        }

                        else if(mmagnitude.at<double>(y,x) <= thresMax && mmagnitude.at<double>(y,x) > thresMin){
                            nonmax.at<uchar>(y,x) = mmagnitude.at<double>(y,x);
                        }
                    }
                    break;
            }

        }
    }

    //hysteria threshold
    for(int y = 1; y < rows - 1; ++y){
        for(int x = 1; x < cols - 1; ++x){
            if(result.at<uchar>(y,x) == 255){
                follow(nonmax, result, x, y, thresMin, thresMax);
            }
        }
    }
}




/*harris*/

void circleMidpoint(Mat &img, int x0, int y0, int radius, int val)
{
    int x = radius, y = 0;
    int radiusError = 1-x;

    while(x > y)
    {
        img.at<uchar>(y+ y0,x + x0) = val;
        img.at<uchar>(x+ y0,y + x0) = val;
        img.at<uchar>(-y+ y0,x + x0) = val;
        img.at<uchar>(-x+ y0,y + x0) = val;
        img.at<uchar>(-y+ y0,-x + x0) = val;
        img.at<uchar>(-x+ y0,-y + x0) = val;
        img.at<uchar>(y+ y0,-x + x0) = val;
        img.at<uchar>(x+ y0,-y + x0) = val;
        y++;
        if(radiusError < 0){
            radiusError += 2 * y + 1;
        }else{
            x--;
            radiusError += 2 * ( y - x + 1);
        }
    }
}

void harris(const Mat src, vector<Point> &result, Mat &borderCorner, Mat Gx, Mat Gy, Mat Gxy, int thres, double k)
{

    int centerKernelGyGx = Gy.cols / 2;
    int centerKernelGxy = Gxy.cols /2;

    Mat Ix2 = (Mat_<double>(src.rows, src.cols));
    Mat Iy2 =  (Mat_<double>(src.rows, src.cols));
    Mat Ixy =  (Mat_<double>(src.rows, src.cols));

    Mat IR =  (Mat_<double>(src.rows, src.cols));

    borderCorner = Mat(src.rows, src.cols,CV_8UC1, Scalar(0));

    double sX;
    double sY;
    int ii,jj;
    int obj = 1;


    for(int i = 0; i < src.cols; ++i){
        for(int j = 0; j < src.rows; ++j){
            sX = 0;
            sY = 0;

            for(int ik = -centerKernelGyGx; ik <= centerKernelGyGx; ++ik ){
                ii = i + ik;
                for(int jk = -centerKernelGyGx; jk <= centerKernelGyGx; ++jk ){
                    jj = j + jk;

                    if(ii >= 0 && ii < src.cols && jj >= 0 && jj < src.rows){
                        sX += src.at<uchar>(jj, ii) * Gx.at<double>(centerKernelGyGx + jk, centerKernelGyGx + ik);
                        sY += src.at<uchar>(jj, ii) * Gy.at<double>(centerKernelGyGx + jk, centerKernelGyGx + ik);
                    }
                }
            }

            Ix2.at<double>(j, i) = sX * sX;
            Iy2.at<double>(j, i) = sY * sY;
            Ixy.at<double>(j, i) = sX * sY;
        }
    }



    double sX2;
    double sY2;
    double sXY;
    double R;

    for(int i = 0; i < src.cols; ++i){
        for(int j = 0; j < src.rows; ++j){
            sX2 = 0;
            sY2 = 0;
            sXY = 0;

            for(int ik = -centerKernelGxy; ik <= centerKernelGxy; ++ik ){
                ii = i + ik;
                for(int jk = -centerKernelGxy; jk <= centerKernelGxy; ++jk ){
                    jj = j + jk;

                    if(ii >= 0 && ii < src.cols && jj >= 0 && jj < src.rows){
                        sX2 += Ix2.at<double>(jj, ii) * Gxy.at<double>(centerKernelGxy + jk, centerKernelGxy + ik);
                        sY2 += Iy2.at<double>(jj, ii) * Gxy.at<double>(centerKernelGxy + jk, centerKernelGxy + ik);
                        sXY += Ixy.at<double>(jj, ii) * Gxy.at<double>(centerKernelGxy + jk, centerKernelGxy + ik);
                    }
                }
            }

            //H = [Sx2(x, y) Sxy(x, y); Sxy(x, y) Sy2(x, y)];
            //R = det(H) - k * (trace(H) ^ 2);

            R =

                    ((sX2 * sY2) - (sXY * sXY)) //det(H)
                    -
                    pow( (sX2 + sY2),2) //(trace(H) ^ 2)
                    *
                    k
                    ;

            if(R > thres){
                IR.at<double>(j, i) = R;
            }else{
                IR.at<double>(j, i) = 0;
            }
        }
    }

    for(int y = 1; y < IR.rows - 1; ++y){
        for(int x = 6; x < IR.cols - 6; ++x){

            //non-maximal suppression
            if(
                    IR.at<double>(y,x) > IR.at<double>(y + 1,x) &&
                    IR.at<double>(y,x) > IR.at<double>(y - 1,x) &&
                    IR.at<double>(y,x) > IR.at<double>(y, x + 1) &&
                    IR.at<double>(y,x) > IR.at<double>(y, x - 1) &&
                    IR.at<double>(y,x) > IR.at<double>(y + 1,x + 1) &&
                    IR.at<double>(y,x) > IR.at<double>(y + 1,x - 1) &&
                    IR.at<double>(y,x) > IR.at<double>(y - 1, x + 1) &&
                    IR.at<double>(y,x) > IR.at<double>(y - 1, x - 1)
                    )
            {

                 result.push_back(Point(x,y));
                 borderCorner.at<uchar>(y,x) = obj;
                 //circleMidpoint(borderCorner, x, y, 4, obj);
                 int jt = 2;
                 borderCorner.at<uchar>(y,x) = obj;

                 // (-jt, -jt) -> (-jt,jt)
                 if(x - jt <= borderCorner.cols && x - jt > 0){
                     for(int i = -jt; i <= jt; ++i){
                        if(y + i <= borderCorner.rows && y + i > 0)
                        {
                            borderCorner.at<uchar>(y+i,x-jt) = obj;
                        }
                     }
                 }

                 // (-jt, jt) -> (jt,jt)
                 if(y + jt <= borderCorner.rows && y + jt > 0){
                     for(int i = -jt; i <= jt; ++i){
                        if(x + i <= borderCorner.cols && x + i > 0)
                        {
                            borderCorner.at<uchar>(y+jt,x+i) = obj;
                        }
                     }
                 }

                 // (jt, -jt) -> (jt,jt)
                 if(x + jt <= borderCorner.cols && x + jt > 0){
                     for(int i = -jt; i <= jt; ++i){
                        if(y + i <= borderCorner.rows && y + i > 0)
                        {
                            borderCorner.at<uchar>(y+i,x+jt) = obj;
                        }
                     }
                 }

                 // (-jt, -jt) -> (jt, -jt)
                 if(y - jt <= borderCorner.rows && y - jt > 0){
                     for(int i = -jt; i <= jt; ++i){
                        if(x + i <= borderCorner.cols && x + i > 0)
                        {
                            borderCorner.at<uchar>(y-jt,x+i) = obj;
                        }
                     }
                 }

                 obj++;


            }
        }
    }


}




//Gaussian
void gaussianKernelGenerator(Mat &result, int besarKernel, double delta)
{
    int kernelRadius = besarKernel / 2;
    result  = Mat_<double>(besarKernel, besarKernel);

    double pengali = 1 / ( 2 * (22 / 7)  * delta * delta) ;

    for(int filterX = - kernelRadius; filterX <= kernelRadius; filterX++){
        for(int filterY = - kernelRadius; filterY <= kernelRadius; filterY++){
            result.at<double>(filterY + kernelRadius, filterX + kernelRadius) =
                    exp(-( sqrt( pow(filterY, 2) + pow(filterX, 2)  ) / ( pow(delta, 2) * 2) ))
                    * pengali;
        }

    }
}

//Gaussian
void gaussianKernelDerivativeGenerator(Mat &resultX, Mat &resultY, int besarKernel, double delta)
{
    int kernelRadius = besarKernel / 2;
    resultX = Mat_<double>(besarKernel, besarKernel);
    resultY = Mat_<double>(besarKernel, besarKernel);

    double pengali = -1 / ( 2 * (22 / 7) * pow(delta, 4) ) ;

    for(int filterX = - kernelRadius; filterX <= kernelRadius; filterX++){
        for(int filterY = - kernelRadius; filterY <= kernelRadius; filterY++){

            resultX.at<double>(filterY + kernelRadius, filterX + kernelRadius) =
                    exp(-( ( pow(filterX, 2)  ) / ( pow(delta, 2) * 2) ))
                    * pengali * filterX;

            resultY.at<double>(filterY + kernelRadius, filterX + kernelRadius) =
                    exp(-( ( pow(filterY, 2)  ) / ( pow(delta, 2) * 2) ))
                    * pengali * filterY;

        }

    }

    //cout<< result << endl;
    //cout<< resultY << endl;
}


void rgb2gray(const Mat src, Mat &result)
{
    CV_Assert(src.depth() != sizeof(uchar)); //harus 8 bit

    result = Mat::zeros(src.rows, src.cols, CV_8UC1); //buat matrik 1 chanel
    uchar data;

    if(src.channels() == 3){

        for( int i = 0; i < src.rows; ++i)
            for( int j = 0; j < src.cols; ++j )
            {
                data = (uchar)(((Mat_<Vec3b>) src)(i,j)[0] * 0.0722 + ((Mat_<Vec3b>) src)(i,j)[1] * 0.7152 + ((Mat_<Vec3b>) src)(i,j)[2] * 0.2126);

                result.at<uchar>(i,j) = data;
            }


    }else{

        result = src;
    }

}

void followEdgeCorner(const Mat edge, const Mat borderCorner, const vector<Point> corner, Mat &result,vector<int> &PointObj, int x, int y, int &obj, const int cornerval);
void followEdge(const Mat edge, const Mat borderCorner, const vector<Point> corner, Mat &result,vector<int> &PointObj, int x, int y, int &obj, const int cornerval)
{
    vector<Point> edgeMulai;

    //kiri atas
    if(edge.at<uchar>(y-1,x-1) == 255 && result.at<uchar>(y-1,x-1) != obj && borderCorner.at<uchar>(y-1,x-1) != PointObj.back() + 1){
        result.at<uchar>(y-1,x-1) = obj;
        edgeMulai.push_back(Point(x-1,y-1));
    }
    //atas
    if(edge.at<uchar>(y-1,x) == 255 && result.at<uchar>(y-1,x) != obj && borderCorner.at<uchar>(y-1,x) != PointObj.back() + 1){
        result.at<uchar>(y-1,x) = obj;
        edgeMulai.push_back(Point(x,y-1));
    }
    //kanan atas
    if(edge.at<uchar>(y-1,x+1) == 255 && result.at<uchar>(y-1,x+1) != obj && borderCorner.at<uchar>(y-1,x+1) != PointObj.back() + 1){
        result.at<uchar>(y-1,x+1) = obj;
        edgeMulai.push_back(Point(x+1,y-1));
    }
    //kanan
    if(edge.at<uchar>(y,x+1) == 255 && result.at<uchar>(y,x+1) != obj && borderCorner.at<uchar>(y,x+1) != PointObj.back() + 1){
        result.at<uchar>(y,x+1) = obj;
        edgeMulai.push_back(Point(x+1,y));
    }
    //kanan bawah
    if(edge.at<uchar>(y+1,x+1) == 255 && result.at<uchar>(y+1,x+1) != obj && borderCorner.at<uchar>(y+1,x+1) != PointObj.back() + 1){
        result.at<uchar>(y+1,x+1) = obj;
        edgeMulai.push_back(Point(x+1,y+1));
    }
    //bawah
    if(edge.at<uchar>(y+1,x) == 255 && result.at<uchar>(y+1,x) != obj && borderCorner.at<uchar>(y+1,x) != PointObj.back() + 1){
        result.at<uchar>(y+1,x) = obj;
        edgeMulai.push_back(Point(x,y+1));
    }
    //kiri bawah
    if(edge.at<uchar>(y+1,x-1) == 255 && result.at<uchar>(y+1,x-1) != obj && borderCorner.at<uchar>(y+1,x-1) != PointObj.back() + 1){
        result.at<uchar>(y+1,x-1) = obj;
        edgeMulai.push_back(Point(x-1,y+1));
    }
    //kiri
    if(edge.at<uchar>(y,x-1) == 255 && result.at<uchar>(y,x-1) != obj && borderCorner.at<uchar>(y,x-1) != PointObj.back() + 1){
        result.at<uchar>(y,x-1) = obj;
        edgeMulai.push_back(Point(x-1,y));
    }

    unsigned int i=0;
    bool ketemuCorner = false;

    while(i < edgeMulai.size() && !ketemuCorner){
        if(borderCorner.at<uchar>(edgeMulai[i].y, edgeMulai[i].x) > 0 && borderCorner.at<uchar>(edgeMulai[i].y, edgeMulai[i].x) != cornerval){
            int newcornerval = borderCorner.at<uchar>(edgeMulai[i].y, edgeMulai[i].x);
            PointObj.push_back(newcornerval - 1);
            followEdgeCorner(edge, borderCorner, corner, result, PointObj, corner[PointObj.back()].x, corner[PointObj.back()].y, obj, newcornerval);
            ketemuCorner = true;
        }else{
            followEdge(edge, borderCorner, corner, result, PointObj, edgeMulai[i].x, edgeMulai[i].y, obj, cornerval);
        }
        i++;
    }
}

void followEdgeCorner(const Mat edge, const Mat borderCorner, const vector<Point> corner, Mat &result,vector<int> &PointObj, int x, int y, int &obj, const int cornerval)
{
    int jt = 3;
    result.at<uchar>(y,x) = obj;

    vector<Point> edgeMulai;

    // (-jt, -jt) -> (-jt,jt)
    if(x - jt <= result.cols && x - jt > 0){
        for(int i = -jt; i <= jt; ++i){
           if(y + i <= result.rows && y + i > 0 && edge.at<uchar>(y+i,x-jt) == 255)
           {
               result.at<uchar>(y+i,x-jt) = obj;
               edgeMulai.push_back(Point(x-jt, y+i));
           }
        }
    }

    // (-jt, jt) -> (jt,jt)
    if(y + jt <= result.rows && y + jt > 0){
        for(int i = -jt; i <= jt; ++i){
           if(x + i <= result.cols && x + i > 0 && edge.at<uchar>(y+jt,x+i) == 255)
           {
               result.at<uchar>(y+jt,x+i) = obj;
               edgeMulai.push_back(Point(x+i,y+jt));
           }
        }
    }

    // (jt, -jt) -> (jt,jt)
    if(x + jt <= result.cols && x + jt > 0){
        for(int i = -jt; i <= jt; ++i){
           if(y + i <= result.rows && y + i > 0 && edge.at<uchar>(y+i,x+jt) == 255)
           {
               result.at<uchar>(y+i,x+jt) = obj;
               edgeMulai.push_back(Point(x+jt, y+i));
           }
        }
    }

    // (-jt, -jt) -> (jt, -jt)
    if(y - jt <= result.rows && y - jt > 0){
        for(int i = -jt; i <= jt; ++i){
           if(x + i <= result.cols && x + i > 0 && edge.at<uchar>(y-jt,x+i) == 255)
           {
               result.at<uchar>(y-jt,x+i) = obj;
               edgeMulai.push_back(Point(x+i, y-jt));
           }
        }
    }

    for(unsigned int i=0; i< edgeMulai.size(); ++i){
        followEdge(edge, borderCorner, corner, result, PointObj, edgeMulai[i].x, edgeMulai[i].y, obj, cornerval);
    }
}


void squaresDetection(const Mat edge, const vector<Point> corner, const Mat borderCorner, vector< vector<int> > &squares, Mat &result)
{
    result = Mat(edge.rows, edge.cols,CV_8UC1, Scalar(255));
    int obj = 0;

    int x,y;

    for(unsigned int i = 0; i < corner.size(); ++i){
        x = corner[i].x;
        y = corner[i].y;
        vector<int> PointObj;
        PointObj.push_back(i);

        if(result.at<uchar>(y, x) == 255){
            followEdgeCorner(edge, borderCorner, corner, result, PointObj, x, y, obj, i+1);
        }


        if( PointObj.size() == 4 ){
            cout<<"objek ketemu (x,y) : "<<endl;
            for(unsigned int i= 0; i < PointObj.size(); ++i){
                cout<<"("<<corner[PointObj[i]].x<<","<<corner[PointObj[i]].y <<"), ";
            }
            cout<<endl;

            squares.push_back(PointObj);
        }
    }



}


int main(int /*argc*/, char /**argv[]*/)
{
    Mat gray,src = imread("D:\\Project\\C++\\CitraDigital\\shapes.png");

    namedWindow("asli");
    imshow("asli", src);

    rgb2gray(src, gray);

    Mat dGx, dGy, Gxy;
    Mat borderCorner, edge,direction, objek;
    vector<Point> corner;

    gaussianKernelGenerator(Gxy, 7, 3.9);
    cout<<"Gxy :"<<endl<<Gxy<<endl<<endl;

    gaussianKernelDerivativeGenerator(dGx, dGy, 7, 1.3);
    cout<<"dGx :"<<endl<<dGx<<endl<<endl;
    cout<<"dGy :"<<endl<<dGy<<endl<<endl;


    canny(gray, edge, direction, 15,2, dGx, dGy);
    harris(gray, corner, borderCorner, dGx, dGy, Gxy, 50000, 0.04);
    cout<<corner.size()<<endl;

    namedWindow("harris");
    imshow("harris", borderCorner);

    namedWindow("edge");
    imshow("edge", edge);

    vector< vector<int> > squares;
    squaresDetection(edge, corner, borderCorner, squares, objek);

    src.copyTo(objek);
    for(unsigned int i = 0; i < squares.size(); i++){
        line(objek, corner[squares[i][0]],corner[squares[i][1]], Scalar(0,255,0), 3, 8, 0);
        line(objek, corner[squares[i][1]],corner[squares[i][2]], Scalar(0,255,0), 3, 8, 0);
        line(objek, corner[squares[i][2]],corner[squares[i][3]], Scalar(0,255,0), 3, 8, 0);
        line(objek, corner[squares[i][0]],corner[squares[i][3]], Scalar(0,255,0), 3, 8, 0);
    }

    namedWindow("objek");
    imshow("objek", objek);

    waitKey(0);
}
