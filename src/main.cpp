// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief The entry point for inference engine Mask RCNN demo application
 * @file mask_rcnn_demo/main.cpp
 * @example mask_rcnn_demo/main.cpp
 */
#include <cv_bridge/cv_bridge.h>
#include <ros/ros.h>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <inference_engine.hpp>
#include "openvino_maskrcnn_inference/common.hpp"
#include "openvino_maskrcnn_inference/ocv_common.hpp"

#define RESET "\033[0m"
#define BLACK "\033[30m"  /* Black */
#define RED "\033[31m"    /* Red */
#define GREEN "\033[32m"  /* Green */
#define YELLOW "\033[33m" /* Yellow */
#define BLUE "\033[34m"   /* Blue */

class MaskRCNNInferenceOpenvino {
   private:
    ros::Subscriber rgb_img_sub_;
    ros::Publisher segmentd_img_pub_;

    ros::NodeHandle *nh_;
    InferenceEngine::Core ie;
    InferenceEngine::InputsDataMap inputInfo;
    InferenceEngine::CNNNetwork network;
    InferenceEngine::ExecutableNetwork executable_network;
    InferenceEngine::InferRequest infer_request;

    size_t netBatchSize;
    size_t netInputHeight;
    size_t netInputWidth;

    std::string detection_out_name;

   public:
    MaskRCNNInferenceOpenvino(ros::NodeHandle *nh_ptr_);
    ~MaskRCNNInferenceOpenvino();
    void callBack(const sensor_msgs::ImageConstPtr &msg);
};

MaskRCNNInferenceOpenvino::MaskRCNNInferenceOpenvino(ros::NodeHandle *nh_ptr_) : nh_(nh_ptr_) {
    std::cout << "InferenceEngine: " << InferenceEngine::GetInferenceEngineVersion() << std::endl;

    /** Read network model **/
    network = ie.ReadNetwork("/home/atas/staubli_ws/src/PICK_PLACE_with_ROS_on_STAUBLI_ARM/ROS_maskrcnn_openvino/maksrcnn_mo/frozen_inference_graph.xml");

    // add DetectionOutput layer as output so we can get detected boxes and their
    // probabilities
    detection_out_name = "reshape_do_2d";
    network.addOutput(detection_out_name.c_str(), 0);

    /** Taking information about all topology inputs **/
    inputInfo = network.getInputsInfo();

    // -----------------------------Prepare input
    // blobs-----------------------------------------------------
    std::cout << "Preparing input blobs" << std::endl;

    std::string imageInputName;
    for (const auto &inputInfoItem : inputInfo) {
        if (inputInfoItem.second->getTensorDesc().getDims().size() == 4) {  // first input contains images
            imageInputName = inputInfoItem.first;
            inputInfoItem.second->setPrecision(InferenceEngine::Precision::U8);
        } else if (inputInfoItem.second->getTensorDesc().getDims().size() == 2) {  // second input contains image info
            inputInfoItem.second->setPrecision(InferenceEngine::Precision::FP32);
        } else {
            throw std::logic_error("Unsupported input shape with size = " + std::to_string(inputInfoItem.second->getTensorDesc().getDims().size()));
        }
    }

    /** network dimensions for image input **/
    const InferenceEngine::TensorDesc &inputDesc = inputInfo[imageInputName]->getTensorDesc();
    IE_ASSERT(inputDesc.getDims().size() == 4);
    netBatchSize = getTensorBatch(inputDesc);
    netInputHeight = getTensorHeight(inputDesc);
    netInputWidth = getTensorWidth(inputDesc);

    // -------------------------Load model to the
    // device----------------------------------------------------
    std::cout << "Loading model to the device" << std::endl;
    executable_network = ie.LoadNetwork(network, "CPU");

    // -------------------------Create Infer
    // Request--------------------------------------------------------
    std::cout << "Create infer request" << std::endl;
    infer_request = executable_network.CreateInferRequest();

    rgb_img_sub_ = nh_->subscribe("/camera/color/image_raw", 1, &MaskRCNNInferenceOpenvino::callBack, this);
    segmentd_img_pub_ = nh_->advertise<sensor_msgs::Image>("/output/maskrcnn/segmented", 1);
}

MaskRCNNInferenceOpenvino::~MaskRCNNInferenceOpenvino() {}

void MaskRCNNInferenceOpenvino::callBack(const sensor_msgs::ImageConstPtr &msg) {
    std::cout << " \n" << std::endl;
    cv_bridge::CvImagePtr cv_ptr;
    try {
        cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    } catch (cv_bridge::Exception &e) {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }
    cv::Mat frame = cv_ptr->image;

    try {
        InferenceEngine::OutputsDataMap outputInfo(network.getOutputsInfo());
        for (auto &item : outputInfo) {
            item.second->setPrecision(InferenceEngine::Precision::FP32);
        }

        /** Iterate over all the input blobs **/
        for (const auto &inputInfoItem : inputInfo) {
            InferenceEngine::Blob::Ptr input = infer_request.GetBlob(inputInfoItem.first);

            /** Fill first input tensor with images. First b channel, then g and r
             * channels **/
            if (inputInfoItem.second->getTensorDesc().getDims().size() == 4) {
                /** Iterate over all input images **/
                matU8ToBlob<unsigned char>(frame, input, 0);
            }

            /** Fill second input tensor with image info **/
            if (inputInfoItem.second->getTensorDesc().getDims().size() == 2) {
                InferenceEngine::LockedMemory<void> inputMapped = InferenceEngine::as<InferenceEngine::MemoryBlob>(input)->wmap();
                auto data = inputMapped.as<float *>();
                data[0] = static_cast<float>(netInputHeight);  // height
                data[1] = static_cast<float>(netInputWidth);   // width
                data[2] = 1;
            }
        }

        // ----------------------------Do
        // inference-------------------------------------------------------------
        infer_request.Infer();
        // -----------------------------------------------------------------------------------------------------

        // ---------------------------Postprocess output
        // blobs--------------------------------------------------

        const auto do_blob = infer_request.GetBlob(detection_out_name.c_str());
        InferenceEngine::LockedMemory<const void> doBlobMapped = InferenceEngine::as<InferenceEngine::MemoryBlob>(do_blob)->rmap();
        const auto do_data = doBlobMapped.as<float *>();

        std::string masks_name = "masks";
        const auto masks_blob = infer_request.GetBlob(masks_name.c_str());
        InferenceEngine::LockedMemory<const void> masksBlobMapped = InferenceEngine::as<InferenceEngine::MemoryBlob>(masks_blob)->rmap();
        const auto masks_data = masksBlobMapped.as<float *>();

        const float PROBABILITY_THRESHOLD = 0.2f;
        const float MASK_THRESHOLD = 0.5f;  // threshold used to determine whether mask pixel corresponds to
                                            // object or to background
        // amount of elements in each detected box description (batch, label, prob,
        // x1, y1, x2, y2)
        IE_ASSERT(do_blob->getTensorDesc().getDims().size() == 2);
        size_t BOX_DESCRIPTION_SIZE = do_blob->getTensorDesc().getDims().back();

        const InferenceEngine::TensorDesc &masksDesc = masks_blob->getTensorDesc();
        IE_ASSERT(masksDesc.getDims().size() == 4);
        size_t BOXES = getTensorBatch(masksDesc);
        size_t C = getTensorChannels(masksDesc);
        size_t H = getTensorHeight(masksDesc);
        size_t W = getTensorWidth(masksDesc);

        size_t box_stride = W * H * C;
        std::map<size_t, size_t> class_color;

        // cv::Mat output_image;
        // output_image = frame.clone();
        cv::Mat output_image(frame.rows, frame.cols, CV_8UC3, cv::Scalar(255, 255, 255));
        std::vector<double> detection_box_areas;

        /** Iterating over all boxes **/
        int obj_index = 1;
        for (size_t box = 0; box < BOXES; ++box) {
            float *box_info = do_data + box * BOX_DESCRIPTION_SIZE;

            auto batch = static_cast<int>(box_info[0]);
            if (batch < 0) break;
            if (batch >= static_cast<int>(netBatchSize)) throw std::logic_error("Invalid batch ID within detection output box");
            float prob = box_info[2];
            float x1 = std::min(std::max(0.0f, box_info[3] * frame.cols), static_cast<float>(frame.cols));
            float y1 = std::min(std::max(0.0f, box_info[4] * frame.rows), static_cast<float>(frame.rows));
            float x2 = std::min(std::max(0.0f, box_info[5] * frame.cols), static_cast<float>(frame.cols));
            float y2 = std::min(std::max(0.0f, box_info[6] * frame.rows), static_cast<float>(frame.rows));
            int box_width = std::min(static_cast<int>(std::max(0.0f, x2 - x1)), frame.cols);
            int box_height = std::min(static_cast<int>(std::max(0.0f, y2 - y1)), frame.rows);

            bool is_this_detection_duplicate = false;

            double current_box_area = static_cast<double>(box_width * box_height);
            for (size_t i = 0; i < detection_box_areas.size(); i++) {
                double ratios = current_box_area / detection_box_areas.at(i);
                if (ratios > 0.9 && ratios < 1.1) {
                    std::cout << RED << "FOUND DETECTIONS THAT LOOKS VERY SIMILAR IN SIZE THEY ARE LIKELY SAME OBJECT, GONNA PASS THIS" << std::endl;
                    std::cout << RESET << std::endl;
                    is_this_detection_duplicate = true;
                    continue;
                }
            }
            if (is_this_detection_duplicate) {
                continue;
            }

            detection_box_areas.push_back(static_cast<double>(box_width * box_height));

            auto class_id = static_cast<size_t>(box_info[1] + 1e-6f);

            if (prob > PROBABILITY_THRESHOLD) {
                size_t color_index = class_color.emplace(class_id, class_color.size()).first->second;
                auto &color = CITYSCAPES_COLORS[color_index % arraySize(CITYSCAPES_COLORS)];
                float *mask_arr = masks_data + box_stride * box + H * W * (class_id - 1);
                std::cout << "Detected class" << class_id << " with probability " << prob << " from batch " << batch << ": [" << x1 << ", " << y1 << "], ["
                          << x2 << ", " << y2 << "]" << std::endl;
                std::cout << "After mask_arr successfully !!" << std::endl;

                cv::Mat mask_mat(H, W, CV_32FC1, mask_arr);

                cv::Rect roi = cv::Rect(static_cast<int>(x1), static_cast<int>(y1), box_width, box_height);
                cv::Mat roi_input_img = output_image(roi);
                const float alpha = 1.0f;

                cv::Mat resized_mask_mat(box_height, box_width, CV_32FC1);
                cv::resize(mask_mat, resized_mask_mat, cv::Size(box_width, box_height));

                cv::Mat uchar_resized_mask(box_height, box_width, CV_8UC3, cv::Scalar(40 * obj_index, 40 * obj_index, 30 * obj_index));
                roi_input_img.copyTo(uchar_resized_mask, resized_mask_mat <= MASK_THRESHOLD);

                cv::addWeighted(uchar_resized_mask, alpha, roi_input_img, 1.0f - alpha, 0.0f, roi_input_img);
                // cv::rectangle(output_image, roi, cv::Scalar(color.blue() * box, color.green(), color.red()), -1);
                obj_index++;
            }
        }

        // Prepare and publish KITTI raw image
        cv_bridge::CvImage cv_bridge_image;
        cv_bridge_image.image = output_image;
        cv_bridge_image.encoding = "bgr8";
        cv_bridge_image.header.stamp = ros::Time::now();
        segmentd_img_pub_.publish(cv_bridge_image.toImageMsg());

    }

    catch (const std::exception &error) {
        std::cout << error.what() << std::endl;
        return;
    } catch (...) {
        std::cout << "Unknown/internal exception happened." << std::endl;
        return;
    }
}

int main(int argc, char *argv[]) {
    ros::init(argc, argv, "ros_maskrcnn_openvino_inference_node");
    ros::NodeHandle nodeHandle;
    MaskRCNNInferenceOpenvino maskRCNNInferenceOpenvino(&nodeHandle);
    ros::spin();
    return 0;
}
