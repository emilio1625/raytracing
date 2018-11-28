//=============================================================================
//  Heavily based on the work of Peter Shirley and Roger Allen
//
//=============================================================================

#ifndef _HITABLE_H_
#define _HITABLE_H_

#include "ray.cuh"

class material;

struct hit_record {
    float t;
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
};

#endif
