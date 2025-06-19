#include <random>
#include <utility>
#include <vector>

namespace kittens
{
    template <typename T>
    std::pair<std::vector<T>, T *> init_random(int N)
    {
        std::vector<T> h_data(N);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dist(-1, 1);

        for (int i = 0; i < N; i++)
            h_data[i] = static_cast<T>(dist(gen));

        T *d_data;
        hipCheck(hipMalloc((void **)&d_data, N * sizeof(T)));
        hipCheck(hipMemcpy(d_data, h_data.data(), N * sizeof(T), hipMemcpyHostToDevice));

        return {h_data, d_data};
    }

    template <typename T, typename RH>
    void assert_equal(std::vector<T> const &lhs, RH const &rhs)
    {
        int N = lhs.size();
        int num_errors = 0;
        constexpr int max_errors = 10;
        for (int i = 0; i < N; i++)
            if (lhs[i] != rhs[i])
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