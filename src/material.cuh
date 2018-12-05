//=============================================================================
//  Heavily based on the work of Peter Shirley and Roger Allen
//
//=============================================================================

// Modified by Emilio Cabrera <emilio1625@gmail.com>

#ifndef _MATERIAL_H_
#define _MATERIAL_H_

#include "hitable.cuh"
#include "ray.cuh"
#include "texture.cuh"

struct hit_record;

/* Linear interpolation */
__device__ inline vec3 lerp(float t, const vec3& start, const vec3& stop)
{
    return (1.0f - t) * start + t * stop;
}

/* Aproximacion de Schlick
 * https://en.wikipedia.org/wiki/Schlick%27s_approximation
 */
__device__ float schlick(float cosine, float ref_idx)
{
    float r0 = (1.0f - ref_idx) / (1.0f + ref_idx);
    r0 = r0 * r0;
    return r0 + (1.0f - r0) * pow((1.0f - cosine), 5);
}

/* Verifica si un rayo puede ser refractado
 * dadas las restricciones de la ley de Snell
 * si puede ser refractado, calcula el rayo refractado
 * ver la implementacion de la clase dielectric
 *   refracted (vec3): parametro de salida, el rayo refractado
 */
__device__ bool refract(const vec3& v,
                        const vec3& n,
                        float ni_over_nt,
                        vec3& refracted)
{
    vec3 uv = unit_vector(v);
    float dt = dot(uv, n);
    float discriminant = 1.0f - ni_over_nt * ni_over_nt * (1.0f - dt * dt);
    if (discriminant > 0.0f) {
        refracted = ni_over_nt * (uv - n * dt) - n * sqrt(discriminant);
        return true;
    } else
        return false;
}

/* Refleccion especular perfecta */
__device__ vec3 reflect(const vec3& v, const vec3& n)
{
    return v - 2.0f * dot(v, n) * n;
}

/* Clase virtual, todos los materiales deben implementar una funcion que
 * devuelva:
 *   - scattered (vec3): el rayo reflejado
 *   - attenuation (vec3): un factor de atenuacion por componente de color
 */
class material
{
public:
    __device__ virtual bool scatter(const ray& r,
                                    const hit_record& rec,
                                    vec3& attenuation,
                                    ray& scattered,
                                    curandState* rand_state) const = 0;
    __device__ virtual vec3 emitted(float u, float v, const vec3& p) const
    {
        return vec3(0.0f, 0.0f, 0.0f);
    }
};

/* Modelo de iluminacion de Lambert
 * Refleja los rayos recibidos en un punto de forma aleatoria
 * (uniformemente distribuida)
 * ver
 * https://en.wikipedia.org/wiki/Bidirectional_reflectance_distribution_function#/media/File:BRDF_diffuse.svg
 * TODO:
 *   - Probar la apariencia con una distribucion normal
 */
class diffuse : public material
{
public:
    __device__ diffuse(tex* a) : albedo(a) {}
    __device__ virtual bool scatter(const ray& r,
                                    const hit_record& rec,
                                    vec3& attenuation,
                                    ray& scattered,
                                    curandState* rand_state) const
    {
        vec3 target = rec.p + rec.normal + random_in_unit_sphere(rand_state);
        scattered =
            ray(rec.p, target - rec.p,
                r.time());  // en realidad el rayo se reflejaria un poco
                            // despues, pero has visto la velocidad de la luz?
        // seria un ejercicio interesante ver el efecto de añadir un delay al
        // rayo
        attenuation = albedo->value(rec.u, rec.v, rec.p);
        return true;
    }
    tex* albedo;
};

/* Modelo de iluminacion especular
 * Modela un material reflejante
 * El factor fuzziness ([0.0, 1.0]) aleatoriza la direccion del rayo reflejado
 * ver
 * https://en.wikipedia.org/wiki/Bidirectional_reflectance_distribution_function#/media/File:BRDF_glossy.svg
 * si es 0, el material es equivalente a un espejo, ver
 * https://en.wikipedia.org/wiki/Bidirectional_reflectance_distribution_function#/media/File:BRDF_mirror.svg
 */
class specular : public material
{
public:
    __device__ specular(tex* a, float fuzziness) : albedo(a)
    {
        if (fuzziness < 1.0f)
            specular::fuzziness = fuzziness;
        else
            specular::fuzziness = 1.0f;
    }
    __device__ virtual bool scatter(const ray& r,
                                    const hit_record& rec,
                                    vec3& attenuation,
                                    ray& scattered,
                                    curandState* rand_state) const
    {
        vec3 reflected = reflect(unit_vector(r.direction()), rec.normal);
        scattered = ray(
            rec.p, reflected + fuzziness * random_in_unit_sphere(rand_state),
            r.time());
        attenuation = albedo->value(rec.u, rec.v, rec.p);
        return (dot(scattered.direction(), rec.normal) > 0.0f);
    }
    tex* albedo;
    float fuzziness;
};

/* Simula un material transparente
 * Basado en el articulo de wikipedia de la ley de Snell
 * https://en.wikipedia.org/wiki/Snell%27s_law#Vector_form
 * De acuerdo a la ley de Snell:
 * Para el rayo reflejado:
 *   r = ni/nt
 *   v_reflect = r*ray + 2*c*normal
 *   Donde c:
 *   c = angulo de incidencia = - dot(normal, ray)
 *   Nota: si c es negativo entonces n apunta en la misma direccion que el
 *   rayo (esto sucede cuando el rayo viene de dentro del objeto)
 *   y debe recalcularse c como c = - dot(- normal, ray)
 * Para el rayo refractado:
 *   v_refract = r*ray + (r*c - sqrt(1 - r^2*(1 - c^2)))*normal
 * Otra consideracion es que si
 *   1 - r^2 * (1 - c^2) < 0
 * entonces no se puede satisfacer la ecuacion para el rayo refractado y
 * solo habra refleccion. Ademas dependiendo del angulo de incidencia los
 * materiales dielectricos refractan mayor o menor cantidad de luz, para
 * aproximar este efecto se usa la aproximacion de Schlick, ver
 * https://en.wikipedia.org/wiki/Schlick%27s_approximation
 * TODO:
 *   - Añadir la posibilidad de especificar un factor de atenuacion
 * al material que actuara como factor de aberracion cromatica (cambiara el
 * color del rayo)
 */

class dielectric : public material
{
public:
    __device__ dielectric(float ri) : ref_idx(ri) {}

    /*
     * Nota sobre la implementacion:
     * Solo se espera un rayo de salida de esta funcion (ver definicion de
     * la clase base). Sin embargo los materiales dielecticos suelen
     * producir al mismo tiempo dos rayos, el reflejante y el refractante.
     * Para seleccionar este problema, esta funcion produce de forma
     * uniformemente aleatoria uno de los dos rayos, que con una cantidad
     * mayor de rayos, converge al mismo efecto visual
     * TODO:
     *   - Optimizar esta funcion
     */
    __device__ virtual bool scatter(const ray& r,
                                    const hit_record& rec,
                                    vec3& attenuation,
                                    ray& scattered,
                                    curandState* rand_state) const
    {
        vec3 outward_normal;
        vec3 reflected = reflect(r.direction(), rec.normal);
        float ni_over_nt;
        attenuation = vec3(1.0f, 1.0f, 1.0f);
        vec3 refracted;
        float reflect_prob;
        float cosine;
        // el rayo viene de dentro o de fuera?
        if (dot(r.direction(), rec.normal) > 0.0f) {
            outward_normal = -rec.normal;
            ni_over_nt = ref_idx;
            //         cosine = ref_idx * dot(r.direction(), rec.normal) /
            //         r.direction().length();
            cosine = dot(r.direction(), rec.normal) / r.direction().length();
            cosine = sqrt(1 - ref_idx * ref_idx * (1 - cosine * cosine));
        } else {
            outward_normal = rec.normal;
            ni_over_nt = 1.0f / ref_idx;
            cosine = -dot(r.direction(), rec.normal) / r.direction().length();
        }
        // se puede refractar?
        if (refract(r.direction(), outward_normal, ni_over_nt, refracted))
            reflect_prob = schlick(cosine, ref_idx);
        else
            reflect_prob = 1.0f;
        // refractamos o reflejamos aleatoriamente
        if (curand_uniform(rand_state) < reflect_prob)
            scattered = ray(rec.p, reflected, r.time());
        else
            scattered = ray(rec.p, refracted, r.time());
        return true;
    }

    float ref_idx;
};

class diffuse_light : public material
{
public:
    __device__ diffuse_light(tex* a) : emit(a) {}
    __device__ virtual bool scatter(const ray& r_in,
                                    const hit_record& rec,
                                    vec3& attenuation,
                                    ray& scattered, curandState* rand_state) const
    {
        return false;
    }
    __device__ virtual vec3 emitted(float u, float v, const vec3& p) const
    {
        return emit->value(u, v, p);
    }

    tex* emit;
};

#endif
