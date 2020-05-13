#include <iostream>
#include <cuda.h>

constexpr int N = 10;

/*
 * NOTE:
 * Come back to this function when directed to.
 *
 * - - - - - - - - - - - - - - - - - - - - - - - 
 *
 * Hopefully by now you've seen how we allocate memory
 * on the host and the device. Now that we've copied our
 * data from the host to the device and called our kernel,
 * we can take a look at our kernel.
 *
 * Please note how similar this is to our `1-threads` example.
 * Each thread sets one index of c. This time, we don't manually
 * pass the thread index to the thread because it's taken care of by
 * CUDA, the library we are using.
 *
 * after examining the kernel you may return to where we called the kernel.
 */
__global__
void add_vectors(
    double* a,
    double* b,
    double* c)
{
  const int thread_id = blockIdx.x;
  c[thread_id] = a[thread_id] + b[thread_id];
}
/*
 *
 * - - - - - - - - - - - - - - - - - - - - - - - 
 *
 */

int main(int, char**)
{
  /*
   * We are setting up the arrays in the same way
   * as before
   */
  std::cout << "Setting a=3, b=5, c=0\n";
  auto a = new double[N];
  auto b = new double[N];
  auto c = new double[N];
  for (int i=0; i<N; i++)
  {
    a[i] = 3.0;
    b[i] = 5.0;
    c[i] = 0.0;
  }

  /*
   * This time, we also have to allocate
   * memory on the 'device' which is our graphics card.
   * our 'host' is the CPU, where this main function will run.
   */
  double* device_a;
  double* device_b;
  double* device_c;

  /*
   * when we call `auto c = new double[N];` we are telling the CPU
   * to allocate enough memory to fit N doubles. Now that we're also
   * using a GPU, we have to tell the GPU to allocate enough memory
   * for N doubles as well. We acomplish this with a cuda function:
   */
  cudaMalloc(&device_a, N * sizeof(double));
  cudaMalloc(&device_b, N * sizeof(double));
  cudaMalloc(&device_c, N * sizeof(double));

  /*
   * Now we have a, b, and c allocated and set to 3, 5, and 0
   * on the host. On the device however, we have only allocated
   * the memory. The memory is uninitialized.
   *
   * To fix this, we will copy the values from a on the host
   * into the memory allocated for a on the device, and same
   * goes for b and c.
   */
  cudaMemcpy(device_a, a, N * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(device_b, b, N * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(device_c, c, N * sizeof(double), cudaMemcpyHostToDevice);

  /*
   * Now that we have our memory copied from the host (cpu) to the
   * device (gpu) we can call our cuda kernel. The kernel can *only*
   * operate on memory allocated on the GPU.
   *
   * After examining the function call below, you may return to the
   * top of the file to take a look at the kernel.
   *
   * Calling the function with function_name<<< , >>>(parameters);
   * is how we inform cuda how it should configure our kernel.
   *
   * the first parameter to the triple angle brackets is the number of blocks
   * per grid that should be allocated, and the second parameter is the number
   * of threads per block that should be allocated.
   * The grid is the largest unit of computation when calling a kernel.
   *
   * Note: grids and blocks are entirely defined in software. threads and
   * warps are determined by the hardware. By aligning the number of
   * blocks and threads in software with the threads in the physical
   * hardware, we can achieve very large increases in performance.
   *
   * For example, calling `add_vectors<<<10, 1>>>(a, b, c)` would tell cuda
   * to allocate it 10 blocks per grid, and 1 thread per block.
   * Alternatively, calling `add_vectors<<<4, 10>>>(a, b, c)` would tell
   * cuda to allocate 4 blocks, each with 10 threads per block totalling
   * 40 threads.
   */
  add_vectors<<<N, 1>>>(device_a, device_b, device_c);

  /*
   * Hopefully by now you have some understanding of the calling conventions
   * for cuda kernels and the nature of the grid, blocks, and threads.
   *
   * Now let us copy the data back from the device to the host, and see if
   * we still get what we expect.
   */
  cudaMemcpy(c, device_c, N * sizeof(double), cudaMemcpyDeviceToHost);

  for (int i=0; i<N; i++)
  {
    std::cout << "c["<<i<<"] = " << c[i] << "\n";
  }

  delete[] a;
  delete[] b;
  delete[] c;

  /*
   * We also have to free memory on the device since we allocated
   * it in two places.
   */
  cudaFree(device_a);
  cudaFree(device_b);
  cudaFree(device_c);

  return 0;
}
