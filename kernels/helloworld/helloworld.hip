#include <kittens.hpp>

__global__ void helloworld_ker(int *in, int *out)
{
    if (threadIdx.x == 0)
        printf("Hello, World!\n");
    out[threadIdx.x] = in[threadIdx.x];
}

int main()
{
    constexpr int N = 32;
    int *in, *out;
    hipCheck(hipMalloc((void **)&in, N * sizeof(int)));
    hipCheck(hipMalloc((void **)&out, N * sizeof(int)));

    int *h_in = (int *)malloc(N * sizeof(int));
    int *h_out = (int *)malloc(N * sizeof(int));
    for (int i = 0; i < N; i++)
        h_in[i] = i;

    hipCheck(hipMemcpy(in, h_in, N * sizeof(int), hipMemcpyHostToDevice));

    helloworld_ker<<<1, N>>>(in, out);
    hipCheck(hipDeviceSynchronize());

    hipCheck(hipMemcpy(h_out, out, N * sizeof(int), hipMemcpyDeviceToHost));

    for (int i = 0; i < N; i++)
        printf("%d ", h_out[i]);
    printf("\n");

    hipCheck(hipFree(in));
    hipCheck(hipFree(out));
    free(h_in);
    free(h_out);

    return 0;
}