//=============================================================================
//  Heavily based on the work of Peter Shirley and Roger Allen
//
//=============================================================================

#ifndef _RAY_H_
#define _RAY_H_
#include "vec3.cuh"

class ray
{
public:
    __device__ ray() = default;
    __device__ ray(const vec3& O, const vec3& D, float time)
    {
        ray::O = O;
        ray::D = D;
        ray::_time = time;
    }
    __device__ vec3 origin() const { return ray::O; }
    __device__ vec3 direction() const { return ray::D; }
    __device__ float time() const { return _time; }
    __device__ vec3 point_at_parameter(float t) const
    {
        return ray::O + t * ray::D;
    }

    vec3 O;
    vec3 D;
    float _time;
};

#endif
