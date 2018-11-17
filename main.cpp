#include <float.h>
#include <chrono>
#include <iostream>
#include "camera.hpp"
#include "hitable_list.hpp"
#include "material.hpp"
#include "sphere.hpp"

#define MAX_RECURSION 20

unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
std::default_random_engine generator(seed);
std::uniform_real_distribution<float> canonical(0.0f, 1.0f);
std::normal_distribution<float> gaussi(0.0f, 0.3f);

vec3 color(const ray& r, hitable* world, int recursion_depth);

int main(int argc, char const* argv[])
{
    hitable* list[5];
    list[0] = new sphere{vec3{0.0f, 0.0f, -1.0f}, 0.5f,
                         new dielectric(2.3f)};
    list[1] = new sphere{vec3{0.0f, 0.0f, -1.0f}, 0.1f,
                         new diffuse(random_canonical())};
    list[2] = new sphere{vec3{0.0f, -100.5f, -1.0f}, 100.0f,
                         new diffuse(random_canonical())};
    list[3] =
        new sphere{vec3{1.0f, -0.0f, -2.0f}, 0.5f,
                   new specular(random_canonical(), canonical(generator))};
    list[4] =
        new sphere{vec3{-1.0f, 0.0f, -1.0f}, 0.5f,
                   new specular(random_canonical(), 0.0f)};
    hitable* world = new hitable_list{list, 5};
    camera cam{};
    int width = 720;
    int height = 360;
    int samples = 100;  // number of samples per pixel
    std::cout << "P3\n" << width << " " << height << "\n255\n";
    for (int j = height - 1; j >= 0; j--) {
        for (int i = 0; i < width; i++) {
            vec3 col{0.0f, 0.0f, 0.0f};
            for (int s = 0; s < samples; s++) {
                float u = float(i + gaussi(generator)) / float(width);
                float v = float(j + gaussi(generator)) / float(height);
                ray r = cam.get_ray(u, v);
                col += color(r, world, 0);
                // vec3 p = r.point_at_parameter(2.0);
            }
            col /= float(samples);
            // gamma correction, gamma = 2
            col = vec3(sqrt(col[0]), sqrt(col[1]), sqrt(col[2]));
            int ir = int(255.99f * col[0]);
            int ig = int(255.99f * col[1]);
            int ib = int(255.99f * col[2]);
            std::cout << ir << " " << ig << " " << ib << std::endl;
        }
    }
    return 0;
}

vec3 color(const ray& r, hitable* world, int recursion_depth)
{
    hit_record rec;
    if (world->hit(r, 0.0001f, MAXFLOAT, rec)) {
        ray scattered;     // output parameter
        vec3 attenuation;  // output parameter
        if (recursion_depth < MAX_RECURSION &&
            rec.mat_ptr->scatter(r, rec, attenuation, scattered)) {
            return attenuation * color(scattered, world, recursion_depth + 1);
        } else {
            return vec3(0.0f, 0.0f, 0.0f);
        }
    } else {
        vec3 unit_dir = unit_vector(r.direction());
        float t = 0.5 * (unit_dir.y() + 1.0f);
        return lerp(t, vec3{1.0f, 1.0f, 1.0f}, vec3{0.5f, 0.7f, 1.0f});
    }
}
