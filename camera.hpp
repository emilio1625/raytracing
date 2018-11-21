#ifndef CAMERAH
#define CAMERAH

#include "ray.hpp"

class camera
{
public:
    /* Based on
     * https://drive.google.com/drive/folders/1a2dLqvRTh6_7zT4guFmQJDYZ9PZ21PJV
     */
    camera(float fov, float aspect, const vec3& pos, const vec3& look_at)
        : world_up{0.0f, 1.0f, 0.0f}
    {
        float theta = fov * M_PI / 180;
        float half_height = tan(theta / 2);
        float half_width = aspect * half_height;
        origin = pos;
        front = unit_vector(pos - look_at);
        right = unit_vector(cross(world_up, front));
        up = cross(front, right);
        lower_left_corner =
            origin - half_width * right - half_height * up - front;
        horizontal = 2 * half_width * right;
        vertical = 2 * half_height * up;
    }

    ray get_ray(float u, float v)
    {
        return ray(origin,
                   lower_left_corner + u * horizontal + v * vertical - origin);
    }

    vec3 origin, lower_left_corner, horizontal, vertical, up, right, front,
        world_up;
};

#endif
