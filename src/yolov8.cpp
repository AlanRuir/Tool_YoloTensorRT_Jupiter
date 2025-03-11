#include <iostream>
#include "yolov8.h"
#include "simple_logger.hpp"

YoloV8::YoloV8()
    : stream_(nullptr)
    , engine_(nullptr)
{
    buffer_[0] = nullptr;
    buffer_[1] = nullptr;
}

YoloV8::~YoloV8()
{
    if (engine_)
    {
        delete engine_;
    }

    cudaStreamSynchronize(stream_);
    cudaStreamDestroy(stream_);

    if (buffer_[0])
    {
        cudaFree(buffer_[0]);
    }
    if (buffer_[1])
    {
        cudaFree(buffer_[1]);
    }

    std::cout << "YoloV8 destructor" << std::endl;
}

void YoloV8::init(std::string engine_path, float conf, float scored)
{
    std::ifstream file(engine_path, std::ios::binary | std::ios::ate);
    char*         trt_stream = nullptr;
    int           size       = 0;

    if (file.good())
    {
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);

        trt_stream = new char[size];
        assert(trt_stream);

        file.read(trt_stream, size);
        file.close();
    }

    std::unique_ptr<nvinfer1::IRuntime> runtime(nvinfer1::createInferRuntime(global_logger));
    assert(runtime != nullptr);
    engine_ = runtime->deserializeCudaEngine(trt_stream, size);
    assert(engine_ != nullptr);
    context_ = std::unique_ptr<nvinfer1::IExecutionContext>(engine_->createExecutionContext());
    assert(context_ != nullptr);

    delete[] trt_stream;

    // 获取输入维度并设置
    auto input_dims = engine_->getTensorShape("images");
    input_height_   = input_dims.d[2];
    input_width_    = input_dims.d[3];
    std::cout << "input height: " << input_height_ << ", width: " << input_width_ << std::endl;

    nvinfer1::Dims4 input_shape{1, 3, input_height_, input_width_};
    context_->setInputShape("images", input_shape);

    // 获取输出维度
    auto output_dims = engine_->getTensorShape("output0");
    output_height_   = output_dims.d[1];
    output_width_    = output_dims.d[2];
    std::cout << "output height: " << output_height_ << ", width: " << output_width_ << std::endl;

    // 计算输出总大小并验证
    int output_size = output_height_ * output_width_;
    std::cout << "Total output elements: " << output_size << std::endl;
    if (output_size <= 0)
    {
        std::cerr << "Invalid output size: " << output_size << std::endl;
        exit(-1);
    }

    // 分配内存
    cudaError_t err = cudaMalloc(&buffer_[0], input_height_ * input_width_ * 3 * sizeof(float));
    if (err != cudaSuccess)
    {
        std::cerr << "cudaMalloc failed for buffer_[0]: " << cudaGetErrorString(err) << std::endl;
        exit(-1);
    }
    err = cudaMalloc(&buffer_[1], output_size * sizeof(float));
    if (err != cudaSuccess)
    {
        std::cerr << "cudaMalloc failed for buffer_[1]: " << cudaGetErrorString(err) << std::endl;
        exit(-1);
    }

    prob_.resize(output_size);

    err = cudaStreamCreate(&stream_);
    if (err != cudaSuccess)
    {
        std::cerr << "cudaStreamCreate failed: " << cudaGetErrorString(err) << std::endl;
        exit(-1);
    }

    score_threshold_ = scored;
}

void YoloV8::detect(cv::Mat& frame, std::vector<DetectResult>& results)
{
    int64_t start  = cv::getTickCount();
    int     width  = frame.cols;
    int     height = frame.rows;

    int      max_size = std::max(width, height);
    cv::Mat  image    = cv::Mat::zeros(max_size, max_size, CV_8UC3);
    cv::Rect roi(0, 0, width, height);
    frame.copyTo(image(roi));

    float x_factor = image.cols / static_cast<float>(input_width_);
    float y_factor = image.rows / static_cast<float>(input_height_);

    cv::Mat blob = cv::dnn::blobFromImage(image, 1 / 255.0, cv::Size(input_width_, input_height_), cv::Scalar(0, 0, 0), true, false); // 计算blob，用作输入

    cudaError_t err = cudaMemcpyAsync(buffer_[0], blob.ptr<float>(), input_height_ * input_width_ * 3 * sizeof(float), cudaMemcpyHostToDevice, stream_);
    if (err != cudaSuccess)
    {
        std::cerr << "cudaMemcpyAsync failed: " << cudaGetErrorString(err) << std::endl;
        exit(-1);
    }

    context_->setTensorAddress("images", buffer_[0]);
    context_->setTensorAddress("output0", buffer_[1]);

    if (!context_->executeV2(buffer_)) // 执行推理
    {
        std::cerr << "TensorRT inference failed" << std::endl;
        exit(-1);
    }

    err = cudaMemcpyAsync(prob_.data(), buffer_[1], output_height_ * output_width_ * sizeof(float), cudaMemcpyDeviceToHost, stream_);
    if (err != cudaSuccess)
    {
        std::cerr << "cudaMemcpyAsync failed: " << cudaGetErrorString(err) << std::endl;
        exit(-1);
    }
    cudaStreamSynchronize(stream_);

    std::vector<cv::Rect> boxes;
    std::vector<int>      class_ids;
    std::vector<float>    confidences;

    cv::Mat dout(84, 8400, CV_32F, prob_.data()); // 将检测结果prob_转换为8400x84的矩阵
    cv::Mat det_output = dout.t();                // 矩阵转置

    /* 循环遍历检测结果，筛选出置信度大于阈值的检测结果 */
    for (int i = 0; i < det_output.rows; ++i)
    {
        cv::Mat   class_score = det_output.row(i).colRange(4, 84);
        cv::Point class_point;
        double    score;
        cv::minMaxLoc(class_score, 0, &score, 0, &class_point);

        if (score > score_threshold_)
        {
            float cx = det_output.at<float>(i, 0);
            float cy = det_output.at<float>(i, 1);
            float ow = det_output.at<float>(i, 2);
            float oh = det_output.at<float>(i, 3);
            int   x  = static_cast<int>((cx - ow / 2) * x_factor);
            int   y  = static_cast<int>((cy - oh / 2) * y_factor);
            int   w  = static_cast<int>(ow * x_factor);
            int   h  = static_cast<int>(oh * y_factor);

            cv::Rect box(x, y, w, h);
            boxes.push_back(box);
            class_ids.push_back(class_point.x);
            confidences.push_back(score);
        }
    }

    /* 对检测结果进行非极大值抑制，去除重复的检测结果 */
    std::vector<int> indexes;
    cv::dnn::NMSBoxes(boxes, confidences, 0.25, 0.45, indexes);
    for (int i = 0; i < indexes.size(); ++i)
    {
        DetectResult dr;
        int          index = indexes[i];
        dr.box             = boxes[index];
        dr.class_id        = class_ids[index];
        dr.score           = confidences[index];
        cv::rectangle(frame, dr.box, cv::Scalar(0, 255, 0), 1, 8);
        results.push_back(dr);
    }

    float t = (cv::getTickCount() - start) / static_cast<float>(cv::getTickFrequency());
    cv::putText(frame, cv::format("FPS : %.2f", 1.0 / t), cv::Point(10, 20), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);
}