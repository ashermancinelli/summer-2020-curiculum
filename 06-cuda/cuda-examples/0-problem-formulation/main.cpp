#include <iostream>

constexpr int N = 10;

int main(int, char**)
{
  /*
   * Below we have created three arrays set to 3, 5, and 0.
   *
   * How might we add these two arrays element-wise?
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
   * We could iterate over the length of the arrays like below...
   *
   * This accomplishes what we want, however it is slow.
   * Each operation could happen in another thread, since no iteration
   * depends on another iteration.
   *
   * Each operation can happen completely independently. This is referred
   * to as 'embarassingly parallel'. Many vector and matrix operations
   * are like this.
   */
  std::cout << "Setting c[i] = a[i] + b[i]\n";
  for (int i = 0; i < N; i++)
  {
    c[i] = a[i] + b[i];
  }

  for (int i=0; i<N; i++)
    std::cout << "c[" << i << "] = " << c[i] << "\n";

  delete[] a;
  delete[] b;
  delete[] c;

  return 0;
}
