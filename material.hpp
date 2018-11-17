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

#ifndef _MATERIAL_H_
#define _MATERIAL_H_

#include "hitable.hpp"
#include "ray.hpp"

struct hit_record;

inline vec3 lerp(float t, const vec3& start, const vec3& stop)
{
    return (1.0f - t) * start + t * stop;
}

float schlick(float cosine, float ref_idx)
{
    float r0 = (1 - ref_idx) / (1 + ref_idx);
    r0 = r0 * r0;
    return r0 + (1 - r0) * pow((1 - cosine), 5);
}

bool refract(const vec3& v, const vec3& n, float ni_over_nt, vec3& refracted)
{
    vec3 uv = unit_vector(v);
    float dt = dot(uv, n);
    float discriminant = 1.0 - ni_over_nt * ni_over_nt * (1 - dt * dt);
    if (discriminant > 0) {
        refracted = ni_over_nt * (uv - n * dt) - n * sqrt(discriminant);
        return true;
    } else
        return false;
}

vec3 reflect(const vec3& v, const vec3& n)
{
    return v - 2 * dot(v, n) * n;
}

class material
{
public:
    virtual bool scatter(const ray& r,
                         const hit_record& rec,
                         vec3& attenuation,
                         ray& scattered) const = 0;
};

/* Lambertian reflection model
 * Follows the Lambert's cosine law
 * It reflects the ray randomly as a new ray inside a unit sphere
 */
class diffuse : public material
{
public:
    diffuse(const vec3& a) : attenuation(a) {}
    virtual bool scatter(const ray& r,
                         const hit_record& rec,
                         vec3& attenuation,
                         ray& scattered) const
    {
        vec3 target = rec.p + rec.normal + random_in_unit_sphere();
        scattered = ray(rec.p, target - rec.p);
        attenuation = diffuse::attenuation;
        return true;
    }
    vec3 attenuation;
};

class specular : public material
{
public:
    specular(const vec3& a, float fuzziness) : attenuation(a)
    {
        if (fuzziness < 1.0f)
            specular::fuzziness = fuzziness;
        else
            specular::fuzziness = 1.0f;
    }
    virtual bool scatter(const ray& r,
                         const hit_record& rec,
                         vec3& attenuation,
                         ray& scattered) const
    {
        vec3 reflected = reflect(unit_vector(r.direction()), rec.normal);
        scattered = ray(rec.p, reflected + fuzziness * random_in_unit_sphere());
        attenuation = specular::attenuation;
        return (dot(scattered.direction(), rec.normal) > 0.0f);
    }
    vec3 attenuation;
    float fuzziness;
};

class dielectric : public material
{
public:
    dielectric(float ri) : ref_idx(ri) {}
    virtual bool scatter(const ray& r,
                         const hit_record& rec,
                         vec3& attenuation,
                         ray& scattered) const
    {
        vec3 outward_normal;
        vec3 reflected = reflect(r.direction(), rec.normal);
        float ni_over_nt;
        attenuation = vec3(1.0, 1.0, 1.0);
        vec3 refracted;
        float reflect_prob;
        float cosine;
        if (dot(r.direction(), rec.normal) > 0) {
            outward_normal = -rec.normal;
            ni_over_nt = ref_idx;
            //         cosine = ref_idx * dot(r.direction(), rec.normal) /
            //         r.direction().length();
            cosine = dot(r.direction(), rec.normal) / r.direction().length();
            cosine = sqrt(1 - ref_idx * ref_idx * (1 - cosine * cosine));
        } else {
            outward_normal = rec.normal;
            ni_over_nt = 1.0 / ref_idx;
            cosine = -dot(r.direction(), rec.normal) / r.direction().length();
        }
        if (refract(r.direction(), outward_normal, ni_over_nt, refracted))
            reflect_prob = schlick(cosine, ref_idx);
        else
            reflect_prob = 1.0;
        if (drand48() < reflect_prob)
            scattered = ray(rec.p, reflected);
        else
            scattered = ray(rec.p, refracted);
        return true;
    }

    float ref_idx;
};

#endif
