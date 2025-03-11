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
    cv::Rect box;      // 边界框
    int      class_id; // 检测到的物体的类别索引
    float    score;    // 检测的置信度得分
};

class YoloV8
{
public:
    YoloV8();
    ~YoloV8();
    void init(std::string engine_path, float conf, float scored);
    void detect(cv::Mat& frame, std::vector<DetectResult>& results); // 检测

private:
    cudaStream_t                                 stream_;    // CUDA流
    nvinfer1::ICudaEngine*                       engine_;    // 引擎
    std::unique_ptr<nvinfer1::IExecutionContext> context_;   // 执行上下文
    void*                                        buffer_[2]; // 输入和输出缓冲
    std::vector<float>                           prob_;      // 检测结果
    int                                          output_height_;
    int                                          output_width_;
    int                                          input_height_;
    int                                          input_width_;
    float                                        score_threshold_; // 置信度阈值
};

#endif // __YOLOV8_H__