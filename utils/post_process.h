#ifndef IMAGE_POST_PROCESS_H
#define IMAGE_POST_PROCESS_H

// #include "opencv2/opencv.hpp"
#include <algorithm>
#include <vector>

namespace LGT{
    namespace spanner{

        struct BBox{
            float x1;
            float y1;
            float x2;
            float y2;
            float score;
            BBox():x1(0),y1(0),x2(0),y2(0),score(0){}
            BBox(float x1_, float y1_, float x2_, float y2_, float score_):x1(x1_),y1(y1_),x2(x2_),y2(y2_),score(score_){}
        } ;

        float get_area(BBox box){
            return (box.x2-box.x1+1)*(box.y2-box.y1+1);
        }
        // intra class nms
        void class_nms(std::vector<BBox>& bboxes, float iou_threshold = 0.5, float score_threshold = 0.05)
        {
            if(! bboxes.size())
                return;
            
            float xx1, yy1, xx2, yy2;
            float area1, area2;
            bboxes.erase(std::remove_if(bboxes.begin(), bboxes.end(), [&score_threshold](BBox b){return b.score<score_threshold;}), bboxes.end());
            std::sort(bboxes.begin(), bboxes.end(), [](BBox b1, BBox b2){return b1.score>b2.score;});
            std::vector<BBox>::iterator ite1,ite2;
            for (ite1 = bboxes.begin(); ite1 != bboxes.end(); ite1 ++)
            {
                area1 = get_area(*ite1);
                for(ite2 = ite1 + 1; ite2!=bboxes.end();)
                {
                    area2 = get_area(*ite2);
                    xx1 = std::max(ite1->x1, ite2->x1);
                    yy1 = std::max(ite1->y1, ite2->y1);
                    xx2 = std::min(ite1->x2, ite2->x2);
                    yy2 = std::min(ite1->y2, ite2->y2);

                    float w = std::max(0.0f, xx2 - xx1 + 1);
                    float h = std::max(0.0f, yy2 - yy1 + 1);
                    float inter = w * h;
                    float overlap = inter / (area1 + area2 - inter);

                    if(overlap > iou_threshold)
                    {
                        ite2 = bboxes.erase(ite2);
                    }
                    else
                    {
                        ite2 ++;
                    }
                   
                }
            }

        }

        // add cx, cy, w, h offset to anchor 
        void delta2bbox(int batchSize, int anchorCount, int inputH, int inputW, int num_cls, const float* anchors_in, 
             const float * score_in, const float* delta_in,  BBox* bbox_out)
        {

            float max_ratio = abs(log2(16/1000.0));
            float anchor_y1, anchor_x1, anchor_y2,anchor_x2;
            float anchor_cy, anchor_cx, anchor_h, anchor_w;
            for(int i = 0; i < batchSize; i ++)
            {
                for (int j = 0; j < anchorCount; j ++)
                {
                    anchor_x1 = anchors_in[i * anchorCount * 4 + j * 4 + 0];
                    anchor_y1 = anchors_in[i * anchorCount * 4 + j * 4 + 1];
                    anchor_x2 = anchors_in[i * anchorCount * 4 + j * 4 + 2];
                    anchor_y2 = anchors_in[i * anchorCount * 4 + j * 4 + 3];

                    anchor_cy = (anchor_y1 + anchor_y2) / 2;
                    anchor_cx = (anchor_x1 + anchor_x2) / 2;
                    anchor_h = (anchor_y2 - anchor_y1);
                    anchor_w = (anchor_x2 - anchor_x1);

                    for (int k =0; k< num_cls; k ++ )
                    {
                        float cur_anchor_y1, cur_anchor_x1, cur_anchor_y2,cur_anchor_x2;
                        float cur_anchor_cy, cur_anchor_cx, cur_anchor_h, cur_anchor_w;
                        float cur_delta_dy, cur_delta_dx, cur_delta_logdh, cur_delta_logdw;

                        cur_delta_dy = delta_in[i * anchorCount * num_cls * 4 + j * num_cls * 4 + k * 4 + 1];
                        cur_delta_dx = delta_in[i * anchorCount * num_cls * 4 + j * num_cls * 4 + k * 4 + 0];
                        cur_delta_logdh = delta_in[i * anchorCount * num_cls * 4 + j * num_cls * 4 + k * 4 + 3];
                        cur_delta_logdw = delta_in[i * anchorCount * num_cls * 4 + j * num_cls * 4 + k * 4 + 2];

                        cur_delta_logdh = std::max(std::min(cur_delta_logdh, max_ratio), -max_ratio);    
                        cur_delta_logdw = std::max(std::min(cur_delta_logdw, max_ratio), -max_ratio); 
            
                        // multiply std_dev
                        cur_delta_dy *= 0.1;
                        cur_delta_dx *= 0.1;
                        cur_delta_logdh *= 0.2;
                        cur_delta_logdw *= 0.2;

                        // add mean
                        cur_delta_dy += 0.0;
                        cur_delta_dx += 0.0;
                        cur_delta_logdh += 0.0;
                        cur_delta_logdw += 0.0;

                        // apply delta
                        cur_anchor_cy = anchor_cy + cur_delta_dy * anchor_h;
                        cur_anchor_cx = anchor_cx + cur_delta_dx * anchor_w;
                        cur_anchor_h = anchor_h * expf(cur_delta_logdh);
                        cur_anchor_w = anchor_w * expf(cur_delta_logdw);

                        cur_anchor_y1 = cur_anchor_cy - 0.5 * cur_anchor_h;
                        cur_anchor_x1 = cur_anchor_cx - 0.5 * cur_anchor_w;
                        cur_anchor_y2 = cur_anchor_cy + 0.5 * cur_anchor_h;
                        cur_anchor_x2 = cur_anchor_cx + 0.5 * cur_anchor_w;

                        // clip bbox: a more precision clip method based on real window could be implemented
                        cur_anchor_y1 = std::max(std::min(cur_anchor_y1, (float)inputH), 0.0f);
                        cur_anchor_x1 = std::max(std::min(cur_anchor_x1, (float)inputW), 0.0f);
                        cur_anchor_y2 = std::max(std::min(cur_anchor_y2, (float)inputH), 0.0f);
                        cur_anchor_x2 = std::max(std::min(cur_anchor_x2, (float)inputW), 0.0f);

                        bbox_out[i * anchorCount * num_cls + j * num_cls  + k ] = BBox(cur_anchor_x1, cur_anchor_y1,
                            cur_anchor_x2,cur_anchor_y2, score_in[i * anchorCount * num_cls + j * num_cls + k ]);

                        


                    }
                }
            }
        }  

    }
}

#endif //#ifndef IMAGE_POST_PROCESS_H
