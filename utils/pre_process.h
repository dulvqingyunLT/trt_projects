#ifndef IMAGE_PROCESS_H
#define IMAGE_PROCESS_H

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

            static void image_normalize(cv::Mat& input_image, const float* img_mean, const float* img_std)
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
            static cv::Mat resize_to_bgrmat_with_ratio(cv::Mat& input_image, int toH, int toW)
            {
                // if (input_image.type() != CV_32FC3)
                // {
                //     input_image.convertTo(input_image,CV_32FC3);
                // }

                uchar* data_origin = (uchar*)input_image.data;

                cv::Mat mRet = cv::Mat(toH, toW, input_image.type());  //uint8

                cv::Size sz = input_image.size();
                
                float ratio = std::min(( float )toH / ( float )sz.width, ( float )toW / ( float )sz.height);
                
                int w_result = sz.width * ratio;
                int h_result = sz.height * ratio;

                cv::resize(input_image, mRet(cv::Rect(0, 0, w_result, h_result)), cv::Size(w_result, h_result));


                uchar* data = (uchar*)mRet.data;

                return mRet;                

            }

        };    

        //  cv::Mat spanner::resize_to_bgrmat_with_ratio(cv::Mat& input_image, int toH, int toW)
        // {

        // }
        //  void spanner:: image_normalize(cv::Mat& input_image, const float* img_mean, const float* img_std)
        // {


        // }
       
        

    }
}

#endif //#ifndef IMAGE_PROCESS_H
