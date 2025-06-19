#pragma once
#include "utils.hpp"
#include <string>

namespace kittens
{
    class kernel_timer
    {
    private:
        hipEvent_t start_event;
        hipEvent_t end_event;
        std::string name;
        float *save_time;
        float scale_factor;
        bool should_print;

        inline void start()
        {
            hipCheck(hipEventRecord(start_event));
        }

        inline void stop()
        {
            float elapsed_time;
            hipCheck(hipEventRecord(end_event));
            hipCheck(hipEventSynchronize(end_event));
            hipCheck(hipEventElapsedTime(&elapsed_time, start_event, end_event));
            hipCheck(hipDeviceSynchronize());
            if (should_print)
                std::cout << "[" << name << "]: " << elapsed_time << " ms" << std::endl;
            if (save_time)
                *save_time += elapsed_time * scale_factor;
        }

    public:
        inline kernel_timer(float *save_time = nullptr, float scale_factor = 1, bool should_print = true, std::string const &name = "kernel") : start_event(nullptr), end_event(nullptr), name(name), save_time(save_time), scale_factor(scale_factor), should_print(should_print)
        {
            hipCheck(hipEventCreate(&start_event));
            hipCheck(hipEventCreate(&end_event));
            this->start();
        }

        inline ~kernel_timer()
        {
            this->stop();
            hipCheck(hipEventDestroy(start_event));
            hipCheck(hipEventDestroy(end_event));
        }
    };
}