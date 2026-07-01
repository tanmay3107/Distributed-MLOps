// async_surveillance.cpp
#include <torch/script.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>

// --- Global Thread-Safe Queue ---
std::queue<cv::Mat> frame_queue;
std::mutex queue_mutex;
std::condition_variable queue_cond_var;
bool system_running = true;
const int MAX_QUEUE_SIZE = 5; // Bounded buffer to prevent RAM exhaustion

// --- Thread 1: The Producer (Camera I/O) ---
void camera_thread(int camera_id) {
    cv::VideoCapture cap(camera_id);
    if (!cap.isOpened()) {
        std::cerr << "❌ Camera failed to open.\n";
        system_running = false;
        return;
    }

    cv::Mat frame;
    while (system_running) {
        cap >> frame;
        if (frame.empty()) break;

        // Lock the mutex before touching the shared queue
        std::unique_lock<std::mutex> lock(queue_mutex);
        
        // If the queue is full, drop the frame to avoid memory leaks
        if (frame_queue.size() >= MAX_QUEUE_SIZE) {
            frame_queue.pop(); 
            std::cout << "⚠️ Queue full: Dropping oldest frame.\n";
        }
        
        frame_queue.push(frame.clone()); // Push a copy into RAM
        lock.unlock();
        
        // Notify the consumer thread that a new frame is ready
        queue_cond_var.notify_one();
    }
    cap.release();
}

// --- Thread 2: The Consumer (Neural Network Compute) ---
void inference_thread(const std::string& model_path) {
    torch::jit::script::Module model;
    try {
        model = torch::jit::load(model_path);
    } catch (const c10::Error& e) {
        std::cerr << "❌ Model failed to load.\n";
        return;
    }

    cv::Mat frame, resized;
    while (system_running) {
        std::unique_lock<std::mutex> lock(queue_mutex);
        
        // Wait until the queue has data OR the system is shutting down
        queue_cond_var.wait(lock, []{ return !frame_queue.empty() || !system_running; });
        
        if (!system_running && frame_queue.empty()) break;

        // Extract the frame and instantly unlock the queue so the camera can keep writing
        frame = frame_queue.front();
        frame_queue.pop();
        lock.unlock(); 

        // --- Execute Heavy Inference ---
        cv::resize(frame, resized, cv::Size(224, 224));
        cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);
        
        torch::Tensor img_tensor = torch::from_blob(resized.data, {1, 224, 224, 3}, torch::kByte);
        img_tensor = img_tensor.permute({0, 3, 1, 2}).toType(torch::kFloat) / 255.0;
        
        std::vector<torch::jit::IValue> inputs{img_tensor};
        at::Tensor output = model.forward(inputs).toTensor();
        
        std::cout << "✅ Inference complete on frame thread.\n";
    }
}

int main() {
    std::cout << "🚀 Booting Multi-Threaded Edge Pipeline...\n";
    
    // Spawn the concurrent threads
    std::thread producer(camera_thread, 0);
    std::thread consumer(inference_thread, "edge_surveillance_int8.pt");

    // Let it run for 60 seconds (Simulated deployment)
    std::this_thread::sleep_for(std::chrono::seconds(60));
    
    std::cout << "\n🛑 Shutting down system...\n";
    system_running = false;
    queue_cond_var.notify_all(); // Wake up any sleeping threads
    
    producer.join();
    consumer.join();
    
    std::cout << "✅ Clean exit.\n";
    return 0;
}