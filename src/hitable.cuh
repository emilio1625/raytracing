//=============================================================================
//  Heavily based on the work of Peter Shirley and Roger Allen
//
//=============================================================================

#ifndef _HITABLE_H_
#define _HITABLE_H_

#include "ray.cuh"
#include "aabb.cuh"
#include <float.h>

class material;

struct hit_record {
    float t;
    float u;
    float v;
    vec3 p;
    vec3 normal;
    material* mat_ptr;
};

class hitable
{
public:
    __device__ virtual bool hit(const ray& r,
                                float t_min,
                                float t_max,
                                hit_record& rec) const = 0;

    __device__ virtual bool bounding_box(float t0,
                                         float t1,
                                         aabb& box) const = 0;
};

__device__ void get_sphere_uv(const vec3& p, float& u, float& v) {
    float phi = atan2(p.z(), p.x());
    float theta = asin(p.y());
    u = 1-(phi + M_PI) / (2*M_PI);
    v = (theta + M_PI/2) / M_PI;
}

#endif
