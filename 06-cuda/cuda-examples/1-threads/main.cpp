#include <thread>
#include <iostream>

constexpr int N = 15;

/*
 * This is an extremely inefficient way to accomplish
 * the same vector addition as in part 0 on multiple threads.
 *
 * Each thread will run the function below, performing the addition
 * based on their thread id.
 */
void add_vectors(
    double* a,
    double* b,
    double* c,
    const int thread_id)
{
  c[thread_id] = a[thread_id] + b[thread_id];
  std::cout << "Thread number " << thread_id
    << " setting c["<<thread_id<<"] = " << c[thread_id]
    << "\n";
}

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
   * After we spawn our threads, we will not wait for them to
   * finish before conitinuing to run. In order to interact with
   * the threads after spawning them, we must hold on to 'handles'
   * which allow us to 'join' them later on.
   *
   * This is an array which will contain our thread handles.
   */
  auto thread_handles = new std::thread[N];

  /*
   * This loop will spawn N threads.
   * Each thread will add one element of a and b into c.
   * Running with N threads means that every thread will
   * be perform 1/N of the total operations needed.
   * Theoretically, this could speed up our program by
   * a factor of N.
   */
  for (int i=0; i<N; i++)
  {
    /*
     * Calling std::thread(function_name, parameters...);
     * is the same as calling function_name(parameters);
     * in a new thread.
     *
     * This is the same as calling `add_vectors(a, b, c, i);`
     * in N different threads.
     */
    thread_handles[i] = std::thread(add_vectors, a, b, c, i);
  }

  /*
   * Now that we have spawned all our threads, we will wait
   * for them all to finish working.
   */
  for (int i=0; i<N; i++)
  {
    thread_handles[i].join();
  }

  /*
   * Now let's see if our output is what we expected it to be.
   */
  for (int i=0; i<N; i++)
  {
    std::cout << "c["<<i<<"] = " << c[i] << "\n";
  }

  /*
   * When you run this program, pay attention to the order in which
   * each thread is ran. Do all the threads print in the same order they
   * are spawned? Or does it seems random? Run the program a few times
   * and make note.
   */

  delete[] a;
  delete[] b;
  delete[] c;

  return 0;
}
