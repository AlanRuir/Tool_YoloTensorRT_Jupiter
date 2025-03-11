#include <iostream>
#include <fstream>
#include <vector>

#include "yolov8.h"
#include "dataset.hpp"

int main(int argc, char* argv[])
{
    std::shared_ptr<YoloV8> detector = std::make_shared<YoloV8>();
    detector->init("../models/yolov8n.engine", 0.4, 0.25F);

    cv::VideoCapture cap(0);
    if (!cap.isOpened())
    {
        std::cerr << "无法打开摄像头" << std::endl;
        return -1;
    }

    // 设置摄像头分辨率
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);  // 设置宽度为 640
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480); // 设置高度为 480

    std::cout << "摄像头已打开" << std::endl;

    cv::Mat                   frame;
    std::vector<DetectResult> results;

    while (true)
    {
        bool ret = cap.read(frame);
        if (!ret || frame.empty())
        {
            std::cerr << "读取摄像头帧失败" << std::endl;
            break;
        }

        detector->detect(frame, results);
        for (auto& dr : results)
        {
            cv::Rect box = dr.box;
            cv::putText(frame, class_names[dr.class_id], cv::Point(box.tl().x, box.tl().y + 10), cv::FONT_HERSHEY_PLAIN, 1.0, cv::Scalar(0, 255, 0), 2);
        }

        cv::imshow("frame", frame);
        if (cv::waitKey(1) == 'q')
        {
            break;
        }

        results.clear();
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}