#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
using namespace cv;
using namespace std;

//高斯平滑
//未使用sigma，边缘无处理
void GaussianTemplateSmooth1(const Mat &src, Mat &dst, double sigma)
{
    sigma = sigma > 0 ? sigma : 0;
    int ksize = cvRound(sigma * 3) * 2 + 1;
    if(ksize == 1)
    {
        src.copyTo(dst);
        return;
    }
    dst.create(src.size(), src.type());
    double *kernel = new double[ksize*ksize];
    double scale = -0.5/(sigma*sigma);
    const double PI = 3.1415926;
    double cons = -scale / PI;
    double sum = 0;
    for(int i = 0; i < ksize; i++) {
        for (int j = 0; j < ksize; j++) {
            int x = i - (ksize - 1) / 2;
            int y = j - (ksize - 1) / 2;
            kernel[i * ksize + j] = cons * exp(scale * (x * x + y * y));
            sum += kernel[i * ksize + j];
        }
    }
    for(int i = ksize*ksize; i >=0; i--)
    {
        *(kernel+i) /= sum;
    }
    uchar* srcData = src.data;
    uchar* dstData = dst.data;
    int center = (ksize-1)/2;
    for(int j = 0; j < src.cols; j++)
    {
        for(int i = 0; i < src.rows; i++)
        {
            double acc = 0;
            double accb = 0, accg = 0, accr = 0;
            for(int m = -center, c = 0; m <= center, c < ksize; m++, c++) {
                for (int n = -center, r = 0; n <= center , r < ksize; n++, r++) {
                    if (m + j >= 0 && m + j < src.cols && n + i >= 0 && n + i < src.rows) {

                        if (src.channels() == 1)
                            acc += *(srcData + src.step * (i + n) + src.channels() * (j + m)) * (*(kernel+r*ksize+c));
                        else {
                            accb += *(srcData + src.step * (i + n) + src.channels() * (j + m) + 0) * (*(kernel+r*ksize+c));
                            accg += *(srcData + src.step * (i + n) + src.channels() * (j + m) + 1) * (*(kernel+r*ksize+c));
                            accr += *(srcData + src.step * (i + n) + src.channels() * (j + m) + 2) * (*(kernel+r*ksize+c));
                        }
                    }
                }
            }

            if (src.channels() == 1)
                *(dstData + dst.step * (i) + dst.channels() * (j)) = (int) acc;
            else {
                *(dstData + dst.step * (i) + dst.channels() * (j) + 0) = (int) accb;
                *(dstData + dst.step * (i) + dst.channels() * (j) + 1) = (int) accg;
                *(dstData + dst.step * (i) + dst.channels() * (j) + 2) = (int) accr;
                }
            }
        }
}

void GaussianSmooth(const Mat &src, Mat &dst, double sigma)
{
    sigma = sigma > 0 ? sigma : -sigma;
    int ksize = cvRound(sigma*3)*2 + 1;
    if(ksize == 1)
    {
        src.copyTo(dst);
        return;
    }
    double *kernel = new double[ksize];

    double scale = -0.5/(sigma*sigma);
    const double PI = 3.1415926;
    double cons = 1/sqrt(-scale/PI);

    double sum = 0;
    int kcenter = ksize / 2;
    for(int i = 0; i < ksize; i++)
    {
        int x = i - kcenter;
        *(kernel+i) = cons*exp(x*x*scale);
        sum += *(kernel+i);
    }
    //归一化
    for(int i = 0; i< ksize; i++)
    {
        *(kernel+i) /= sum;
    }
    dst.create(src.size(),src.type());
    Mat temp;
    temp.create(src.size(),src.type());

    uchar *srcData = src.data;
    uchar *dstData = dst.data;
    uchar *tempData = temp.data;

    //x 方向
    for(int y = 0; y < src.rows; y++)
    {
        for(int x = 0; x < src.cols; x++)
        {
            double mul = 0;
            sum = 0;
            double bmul = 0,gmul=0,rmul=0;
            for(int i = -kcenter; i <= kcenter; i++) {
                if ((x + i) >= 0 && (x + i) < src.cols) {
                    if (src.channels() == 1)
                        mul += *(srcData + src.step * (y) + src.channels() * (x + i)) * (*(kernel + kcenter + i));
                    else {
                        bmul += *(srcData + src.step * (y) + src.channels() * (x + i)) * (*(kernel + kcenter + i));
                        gmul += *(srcData + src.step * (y) + src.channels() * (x + i) + 1) * (*(kernel + kcenter + i));
                        rmul += *(srcData + src.step * (y) + src.channels() * (x + i) + 2) * (*(kernel + kcenter + i));

                    }
                }
            }
            if(src.channels() == 1)
            {
                *(tempData + temp.step*y + x) = mul;
            }
            else
            {
                *(tempData + temp.step*y + temp.channels()*x +0) = bmul;
                *(tempData + temp.step*y + temp.channels()*x +1) = gmul;
                *(tempData + temp.step*y + temp.channels()*x +2) = rmul;
            }
        }
    }
    //y方向
    for(int x = 0; x < temp.cols; x++)
    {
        for(int y = 0; y < temp.rows; y++)
        {
            double mul = 0;
            sum = 0;
            double bmul = 0,gmul=0,rmul=0;
            for(int i = -kcenter; i <= kcenter; i++) {
                if ((y + i) >= 0 && (y + i) < temp.rows) {
                    if (temp.channels() == 1)
                        mul += *(tempData + temp.step * (y+i) + temp.channels() * x) * (*(kernel + kcenter + i));
                    else {
                        bmul += *(tempData + temp.step * (y+i) + temp.channels() * x) * (*(kernel + kcenter + i));
                        gmul += *(tempData + src.step * (y+i) + temp.channels() * x + 1) * (*(kernel + kcenter + i));
                        rmul += *(tempData + src.step * (y+i) + temp.channels() * x + 2) * (*(kernel + kcenter + i));

                    }
                }
            }
            if(temp.channels() == 1)
            {
                *(dstData + dst.step*y + x) = mul;
            }
            else
            {
                *(dstData + dst.step*y + dst.channels()*x +0) = bmul;
                *(dstData + dst.step*y + dst.channels()*x +1) = gmul;
                *(dstData + dst.step*y + dst.channels()*x +2) = rmul;
            }
        }
    }
    delete []kernel;
}



int main()
{
    Mat img = imread("/home/jdi/test1.jpg");
    Mat dst;
    Mat dst1;
    double time = (double)getTickCount();
    GaussianSmooth(img,dst1,3);

    double time1 = ((double)getTickCount() - time)/getTickFrequency();
    cout << "first kind of GaussianSmooth consume time is " << time1 << endl;
    GaussianTemplateSmooth1(img,dst,3);
    double time2 = ((double)getTickCount() - time)/getTickFrequency();
    cout << "second kind of GaussianSmooth consume time is " << time2 << endl;
    char *window1 = "orignal";
    char *window2 = "GaussianSmooth";
    char *window3 = "GaussianSmoothAnotherWay";
    namedWindow(window1,WINDOW_AUTOSIZE);
    namedWindow(window2,WINDOW_AUTOSIZE);
    namedWindow(window3,WINDOW_AUTOSIZE);
    imshow(window1,img);
    imshow(window2,dst);
    imshow(window3,dst1);

    waitKey(0);
    return 0;













}