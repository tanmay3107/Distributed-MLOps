// cctv_surveillance.cpp
#include <torch/script.h> // The LibTorch C++ API
#include <opencv2/opencv.hpp>
#include <iostream>
#include <memory>

int main() {
    std::cout << "🚀 Booting Bare-Metal Edge Vision Pipeline..." << std::endl;

    // 1. Load the quantized INT8 model we exported on Day 8
    torch::jit::script::Module model;
    try {
        model = torch::jit::load("edge_surveillance_int8.pt");
        std::cout << "✅ INT8 Model successfully loaded into memory." << std::endl;
    }
    catch (const c10::Error& e) {
        std::cerr << "❌ Error loading the model\n";
        return -1;
    }

    // 2. Initialize the hardware camera stream (OpenCV)
    // 0 represents the default /dev/video0 hardware camera
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "❌ Error: Could not release camera hardware lock.\n";
        return -1;
    }

    std::cout << "🎥 Camera hardware acquired. Beginning real-time person detection..." << std::endl;
    
    cv::Mat frame, resized_frame;
    
    // The infinite surveillance loop
    while (true) {
        // Grab the physical frame from the sensor
        cap >> frame;
        if (frame.empty()) break;

        // 3. Preprocess the frame for the CNN (Resize to 224x224 and convert BGR to RGB)
        cv::resize(frame, resized_frame, cv::Size(224, 224));
        cv::cvtColor(resized_frame, resized_frame, cv::COLOR_BGR2RGB);

        // Convert the OpenCV Mat into a PyTorch Tensor
        // We ensure it is a Float tensor and normalize it to [0, 1]
        torch::Tensor img_tensor = torch::from_blob(resized_frame.data, {1, 224, 224, 3}, torch::kByte);
        img_tensor = img_tensor.permute({0, 3, 1, 2}).toType(torch::kFloat) / 255.0;

        // 4. Execute the Forward Pass
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(img_tensor);
        
        // Execute the model on the CPU (taking advantage of INT8 x86/ARM optimizations)
        at::Tensor output = model.forward(inputs).toTensor();
        
        // Apply Softmax to get confidence probabilities
        at::Tensor probabilities = torch::softmax(output, 1);
        
        // Extract the confidence-based detection prediction
        float person_confidence = probabilities[0][1].item<float>();
        
        // 5. Draw Confidence-Based Detection Boxes
        if (person_confidence > 0.85) { // Strict 85% confidence threshold
            std::string label = "PERSON DETECTED: " + std::to_string(person_confidence * 100).substr(0, 5) + "%";
            cv::putText(frame, label, cv::Point(20, 50), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 255), 2);
            cv::rectangle(frame, cv::Point(10, 10), cv::Point(frame.cols - 10, frame.rows - 10), cv::Scalar(0, 0, 255), 3);
        }

        // Display the live feed
        cv::imshow("EdgeVision Zero-Latency Surveillance", frame);

        // Break the loop if the user presses 'ESC'
        if (cv::waitKey(1) == 27) break;
    }

    // Gracefully release the camera hardware
    cap.release();
    cv::destroyAllWindows();
    return 0;
}