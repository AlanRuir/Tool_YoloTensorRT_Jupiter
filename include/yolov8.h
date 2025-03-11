#ifndef __YOLOV8_H__
#define __YOLOV8_H__

#include <cuda_runtime.h>
#include <cuda.h>
#include <memory>
#include <vector>
#include <chrono>
#include <string>
#include <fstream>

#include <NvInfer.h>
#include <opencv2/opencv.hpp>

struct DetectResult
{
    cv::Rect box;      // Bounding box coordinates (x, y, width, height)
    int      class_id; // Class index of the detected object
    float    score;    // Confidence score of the detection
};

class YoloV8
{
public:
    YoloV8();
    ~YoloV8();
    void init(std::string engine_path, float conf, float scored);
    void detect(cv::Mat& frame, std::vector<DetectResult>& results);

private:
    cudaStream_t                                 stream_;
    nvinfer1::ICudaEngine*                       engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;
    void*                                        buffer_[2];
    std::vector<float>                           prob_;
    int                                          output_height_;
    int                                          output_width_;
    int                                          input_height_;
    int                                          input_width_;
    float                                        score_threshold_;
};

#endif // __YOLOV8_H__