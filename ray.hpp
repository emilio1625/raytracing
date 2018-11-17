//=============================================================================
// Written in 2016 by Peter Shirley <ptrshrl@gmail.com>
//
// To the extent possible under law, the author(s) have dedicated all copyright
// and related and neighboring rights to this software to the public domain
// worldwide. This software is distributed without any warranty.
//
// You should have received a copy (see file COPYING.txt) of the CC0 Public
// Domain Dedication along with this software. If not, see
// <http://creativecommons.org/publicdomain/zero/1.0/>.
//=============================================================================

#ifndef _RAY_H_
#define _RAY_H_
#include "vec3.hpp"

class ray
{
public:
    ray() = default;
    ray(const vec3& O, const vec3& D)
    {
        ray::O = O;
        ray::D = D;
    }
    vec3 origin() const { return ray::O; }
    vec3 direction() const { return ray::D; }
    vec3 point_at_parameter(float t) const { return ray::O + t * ray::D; }

    vec3 O;
    vec3 D;
};

#endif
