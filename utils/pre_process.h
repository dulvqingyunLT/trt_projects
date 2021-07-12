#ifndef IMAGE_PROCESS_H
#define IMAGE_PROCESS_H

#include <dirent.h>
#include "opencv2/opencv.hpp"


namespace LGT{
    namespace spanner{

        class PreProcess
        {
        private:
            /* data */
        public:
            PreProcess(/* args */);
            ~PreProcess();

            static inline void image_normalize(cv::Mat& input_image, const float* img_mean, const float* img_std)
            {
                if (input_image.type() != CV_32FC3)
                {
                    input_image.convertTo(input_image,CV_32FC3);
                }

                cv::Mat_<cv::Vec3f>::iterator it = input_image.begin<cv::Vec3f>();

                cv::Mat_<cv::Vec3f>::iterator itend = input_image.end<cv::Vec3f>();

                for ( ; it < itend; it++) {
                    (*it)[0] = ((*it)[0]-img_mean[0])/img_std[0];   
                    (*it)[1] = ((*it)[1]-img_mean[1])/img_std[1];  
                    (*it)[2] = ((*it)[2]-img_mean[2])/img_std[2];                               
                }
                // input_image = input_image/255.0;
                float* data = (float*)input_image.data;
                return ;

                
                // std::vector<cv::Mat> bgrChannels(3);
                // cv::split(input_image, bgrChannels);
                // for (auto i = 0; i < bgrChannels.size(); i++)
                // {
                //     bgrChannels[i].convertTo(bgrChannels[i], CV_32FC1, 1.0 / img_std[i], (0.0 - img_mean[i]) / img_std[i]);
                // }

                // cv::merge(bgrChannels, input_image);
                
                // input_image - input_image/255.0;
            }
            static inline cv::Mat resize_to_bgrmat_with_ratio(cv::Mat& input_image, int toH, int toW, bool keep_ratio=true)
            {
                // if (input_image.type() != CV_32FC3)
                // {
                //     input_image.convertTo(input_image,CV_32FC3);
                // }

                uchar* data_origin = (uchar*)input_image.data;

                cv::Mat mRet = cv::Mat(toH, toW, input_image.type());  //uint8

                cv::Size sz = input_image.size();

                if(keep_ratio)
                {
                    float ratio = std::min(( float )toH / ( float )sz.width, ( float )toW / ( float )sz.height);
                
                    int w_result = sz.width * ratio;
                    int h_result = sz.height * ratio;
                    cv::resize(input_image, mRet(cv::Rect(0, 0, w_result, h_result)), cv::Size(w_result, h_result));
                }
                else
                {
                    // float ratio_h = static_cast<float>(toH) / sz.width;
                    // float ratio_w = static_cast<float>(toW) / sz.height;
                    cv::resize(input_image, mRet, mRet.size());


                }
                


                


                // uchar* data = (uchar*)mRet.data;

                return mRet;                

            }

            static inline cv::Mat preprocessImg(cv::Mat& img, int input_w, int input_h) {
                int w, h, x, y;
                float r_w = input_w / (img.cols*1.0);
                float r_h = input_h / (img.rows*1.0);
                if (r_h > r_w) {
                    w = input_w;
                    h = r_w * img.rows;
                    x = 0;
                    y = (input_h - h) / 2;
                } else {
                    w = r_h * img.cols;
                    h = input_h;
                    x = (input_w - w) / 2;
                    y = 0;
                }
                cv::Mat re(h, w, CV_8UC3);
                cv::resize(img, re, re.size(), 0, 0, cv::INTER_LINEAR);
                cv::Mat out(input_h, input_w, CV_8UC3, cv::Scalar(128, 128, 128));
                re.copyTo(out(cv::Rect(x, y, re.cols, re.rows)));
                return out;
            }

        };    


       
    }
}

#endif //#ifndef IMAGE_PROCESS_H
