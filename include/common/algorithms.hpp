namespace kittens {

template <typename T, bool should_transpose_A = false, bool should_transpose_B = false>
void cpu_matmul(T *A, T *B, T *C, int M, int N, int K) {
#pragma omp parallel for collapse(2)
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      float sum = 0;
      for (int k = 0; k < K; k++) {
        T &a_elem = should_transpose_A ? A[k * M + i] : A[i * K + k];
        T &b_elem = should_transpose_B ? B[j * K + k] : B[k * N + j];
        auto product = base_types::convertor<float, T>::convert(a_elem * b_elem);
        sum += product;
      }
      C[i * N + j] = base_types::convertor<T, float>::convert(sum);
    }
  }
}
} // namespace kittens