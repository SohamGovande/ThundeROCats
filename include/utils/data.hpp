#pragma once

#include <random>
#include <utility>
#include <vector>
#include <concepts>
#include <fstream>
#include <iomanip>

static int seed = 0;

namespace kittens
{

    struct fill_empty
    {
        inline bool has_value() { return false; }
        inline float value() { return 0; }
    };
    struct fill_random
    {
        std::mt19937 gen;
        std::uniform_real_distribution<float> dist;

        inline fill_random()
        {
            gen = std::mt19937(seed++);
            dist = std::uniform_real_distribution<float>(-1, 1);
        }

        inline bool has_value() { return true; }
        inline float value() { return dist(gen); }
    };
    struct fill_zeros
    {
        inline bool has_value() { return true; }
        inline float value() { return 0; }
    };
    struct fill_ones
    {
        inline bool has_value() { return true; }
        inline float value() { return 1; }
    };

    template <typename T>
    concept fill_type = std::is_same_v<T, fill_empty> || std::is_same_v<T, fill_random> || std::is_same_v<T, fill_zeros> || std::is_same_v<T, fill_ones>;

    template <fill_type fill_type, typename T>
    std::pair<std::vector<T>, T *> init(int N)
    {
        std::vector<T> h_data(N);
        fill_type fill;
        if (fill.has_value())
            for (int i = 0; i < N; i++)
                h_data[i] = static_cast<T>(fill.value());

        T *d_data;
        hipCheck(hipMalloc((void **)&d_data, N * sizeof(T)));
        hipCheck(hipMemcpy(d_data, h_data.data(), N * sizeof(T), hipMemcpyHostToDevice));

        return {h_data, d_data};
    }

    template <typename T>
    void print_tensor_to_file(std::string const &filename, std::vector<std::tuple<std::string, T *, int, int>> const &data)
    {
        std::ofstream file(filename);
        bool include_comma = filename.find(".csv") != std::string::npos;
        for (auto const &[name, values, rows, cols] : data)
        {
            file << "BEGIN TENSOR " << name << std::endl;
            for (int row = 0; row < rows; row++)
            {
                for (int col = 0; col < cols; col++)
                {
                    float val = static_cast<float>(values[row * cols + col]);
                    // print a space before positive numbers to align the numbers
                    if (val >= 0)
                        file << " ";
                    file << std::fixed << std::setprecision(3) << std::setw(5) << std::setfill(' ') << val;
                    if (include_comma)
                        file << ",";
                    file << " ";
                }
                file << std::endl;
            }
            file << "END TENSOR " << name << std::endl;
        }
    }

    template <typename T, typename RH>
    void assert_equal(std::vector<T> const &lhs, RH const &rhs)
    {
        int N = lhs.size();
        int num_errors = 0;
        constexpr int max_errors = 50;
        constexpr float epsilon = 2e-2;
        for (int i = 0; i < N; i++)
            if (std::abs(lhs[i] - rhs[i]) > epsilon)
            {
                if (num_errors < max_errors)
                    std::cout << "Error at index " << i << ": " << static_cast<float>(lhs[i]) << " != " << static_cast<float>(rhs[i]) << std::endl;
                num_errors++;
            }
        if (num_errors > max_errors)
            std::cout << "... and " << num_errors - max_errors << " more errors" << std::endl;
        else if (num_errors == 0)
            std::cout << "No errors" << std::endl;
    }
}