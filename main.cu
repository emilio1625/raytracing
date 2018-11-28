//=============================================================================
//  Heavily based on the work of Peter Shirley and Roger Allen
//
//=============================================================================

#include <curand_kernel.h>
#include <float.h>
#include <chrono>
#include <iostream>
#include "camera.cuh"
#include "hitable_list.cuh"
#include "material.cuh"
#include "sphere.cuh"

/* Constants */

#define MAX_RECURSION 20

/* Macros */

#define checkCudaErrors(val) check_cuda((val), #val, __FILE__, __LINE__)

/* Variable declaration */

/* Function prototypes */

// Check for error in cuda funtion calls
void check_cuda(cudaError_t result,
                char const* const func,
                const char* const file,
                int const line);
// Determines the color of the point where the ray hits
__device__ vec3 color(const ray& r, hitable* world, curandState rand_state);
// Initializes the random state for all the pixels
__global__ void rand_state_init(int width,
                                int height,
                                curandState* local_state);
// Creates the image to draw
__global__ void render(vec3* fb,
                       int width,
                       int height,
                       int samples,
                       camera** cam,
                       hitable** world,
                       curandState* rand_state);
// Creates the camera and geometry objects in the GPU memory
__global__ void create_world(hitable** list,
                             int count,
                             hitable** world,
                             camera** cam,
                             float fov,
                             const vec3 pos,
                             const vec3 look_at,
                             int width,
                             int height);
// Frees the world in the only possible way, destroying it :)
__global__ void destroy_world(hitable** list,
                              size_t count,
                              hitable** world,
                              camera** cam);

/* Main Program */

int main(int argc, char const* argv[])
{
    int width = 384;
    int height = 216;
    int pixels = width * height;
    int samples = 200;   // number of samples per pixel
    int tx = 8, ty = 8;  // threads

    size_t fb_size = pixels * sizeof(vec3);  // image size

    // allocate shared memory (CPU & GPU)
    vec3* fb;
    checkCudaErrors(cudaMallocManaged((void**)&fb, fb_size));

    // random state
    curandState* rand_state_d;
    checkCudaErrors(
        cudaMalloc((void**)&rand_state_d, pixels * sizeof(curandState)));

    // array of objects to hit
    size_t hitable_count = 2;
    hitable** list_d;
    checkCudaErrors(
        cudaMalloc((void**)&list_d, hitable_count * sizeof(hitable*)));

    // list of objects to hit
    hitable** world_d;
    checkCudaErrors(cudaMalloc((void**)&world_d, sizeof(hitable*)));

    // Camera
    camera** camera_d;
    checkCudaErrors(cudaMalloc((void**)&camera_d, sizeof(camera*)));

    // build the world
    create_world<<<1, 1>>>(list_d, hitable_count, world_d, camera_d, 90.0,
                           vec3(0, 2, 3), vec3(0, 0, 0), width, height);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    dim3 blocks(width / tx + 1, height / ty + 1);
    dim3 threads(tx, ty);
    rand_state_init<<<blocks, threads>>>(width, height, rand_state_d);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    render<<<blocks, threads>>>(fb, width, height, samples, camera_d, world_d,
                                rand_state_d);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    std::cout << "P3\n"
              << width << " " << height << std::endl
              << "255" << std::endl;
    for (int j = height - 1; j >= 0; j--) {
        for (int i = 0; i < width; i++) {
            size_t pixel = j * width + i;
            int ir = int(255.99f * fb[pixel].r());
            int ig = int(255.99f * fb[pixel].g());
            int ib = int(255.99f * fb[pixel].b());
            std::cout << ir << " " << ig << " " << ib << std::endl;
        }
    }

    checkCudaErrors(cudaDeviceSynchronize());
    destroy_world<<<1, 1>>>(list_d, hitable_count, world_d, camera_d);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaFree(camera_d));
    checkCudaErrors(cudaFree(world_d));
    checkCudaErrors(cudaFree(list_d));
    checkCudaErrors(cudaFree(rand_state_d));
    checkCudaErrors(cudaFree(fb));

    cudaDeviceReset();

    return 0;
}

/* Function definitions */

void check_cuda(cudaError_t result,
                char const* const func,
                const char* const file,
                int const line)
{
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result)
                  << " at " << file << ":" << line << " '" << func << std::endl;
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}

__device__ vec3 color(const ray& r, hitable** world, curandState* local_state)
{
    ray cur_ray = r;
    vec3 cur_att = vec3(1.0f, 1.0f, 1.0f);
    hit_record rec;
    for (int i = 0; i < MAX_RECURSION; i++) {
        if ((*world)->hit(cur_ray, 0.001f, MAXFLOAT, rec)) {
            ray scattered;     // output parameter
            vec3 attenuation;  // output parameter
            if (rec.mat_ptr->scatter(cur_ray, rec, attenuation, scattered,
                                     local_state)) {
                cur_att *= attenuation;
                cur_ray = scattered;
            } else {
                return vec3(0.0f, 0.0f, 0.0f);
            }
        } else {
            vec3 unit_dir = unit_vector(cur_ray.direction());
            float t = 0.5 * (unit_dir.y() + 1.0f);
            return lerp(t, vec3(1.0f, 1.0f, 1.0f), vec3(0.5f, 0.7f, 1.0f));
        }
    }
    return vec3(0.0f, 0.0f, 0.0f);
}

__global__ void rand_state_init(int width, int height, curandState* rand_state)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= width) || (j >= height))
        return;
    int pixel = j * width + i;
    curand_init(0, pixel, 0, &rand_state[pixel]);
}

__global__ void render(vec3* fb,
                       int width,
                       int height,
                       int samples,
                       camera** cam,
                       hitable** world,
                       curandState* rand_state)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= width) || (j >= height))
        return;
    int pixel = j * width + i;
    float u, v;
    curandState local_state = rand_state[pixel];
    vec3 col(0.0f, 0.0f, 0.0f);

    for (int k = 0; k < samples; k++) {
        u = float(i + curand_uniform(&local_state)) / float(width);
        v = float(j + curand_uniform(&local_state)) / float(height);
        ray r = (*cam)->get_ray(u, v);
        col += color(r, world, &local_state);
    }

    rand_state[pixel] = local_state;
    col /= float(samples);
    col[0] = sqrt(col[0]);
    col[1] = sqrt(col[1]);
    col[2] = sqrt(col[2]);
    fb[pixel] = col;
}

__global__ void create_world(hitable** list,
                             int count,
                             hitable** world,
                             camera** cam,
                             float fov,
                             const vec3 pos,
                             const vec3 look_at,
                             int width,
                             int height)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {  // instantiate only once
        list[0] =
            new sphere(vec3(0.0f, 0.0f, -1.0f), 0.5f, new dielectric(2.3f));
        list[1] = new sphere(vec3(0.0f, -100.5f, -1.0f), 100.0f,
                             new diffuse(vec3(0.1f, 0.5f, 0.7f)));
        *world = new hitable_list(list, count);
        *cam = new camera(fov, float(width) / float(height), pos, look_at);
    }
}

__global__ void destroy_world(hitable** list,
                              size_t count,
                              hitable** world,
                              camera** cam)
{
    for (int i = 0; i < count; i++) {
        delete ((sphere*)list[i])->mat_ptr;
        delete list[i];
    }
    delete *world;
    delete *cam;
}
