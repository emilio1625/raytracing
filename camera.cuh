//=============================================================================
//  Heavily based on the work of Peter Shirley and Roger Allen
//
//=============================================================================

#ifndef CAMERAH
#define CAMERAH

#include <curand_kernel.h>
#include "ray.cuh"

class camera
{
public:
    /* Based on
     * https://drive.google.com/drive/folders/1a2dLqvRTh6_7zT4guFmQJDYZ9PZ21PJV
     */
    __device__ camera(float fov,
                      float aspect,
                      const vec3& pos,
                      const vec3& look_at,
                      float time_i,
                      float time_f)
        : world_up(0.0f, 1.0f, 0.0f)
    {
        float theta = fov * float(M_PI) / 180.0f;
        float half_height = tan(theta / 2.0f);
        float half_width = aspect * half_height;
        origin = pos;
        front = unit_vector(pos - look_at);
        right = unit_vector(cross(world_up, front));
        up = cross(front, right);
        lower_left_corner =
            origin - half_width * right - half_height * up - front;
        horizontal = 2.0f * half_width * right;
        vertical = 2.0f * half_height * up;

        ti = time_i;
        tf = time_f;
    }

    __device__ ray get_ray(float s, float t, curandState* rand_state)
    {
        float time = ti + curand_uniform(rand_state) * (tf - ti);
        return ray(origin,
                   lower_left_corner + s * horizontal + t * vertical - origin, time);
    }

    vec3 origin, lower_left_corner, horizontal, vertical, up, right, front,
        world_up;
    float ti, tf;  // time of aperture
};

#endif
