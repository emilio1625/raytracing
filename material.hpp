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

// Modified by Emilio Cabrera <emilio1625@gmail.com>

#ifndef _MATERIAL_H_
#define _MATERIAL_H_

#include "hitable.hpp"
#include "ray.hpp"

struct hit_record;

/* Linear interpolation */
inline vec3 lerp(float t, const vec3& start, const vec3& stop)
{
    return (1.0f - t) * start + t * stop;
}

/* Aproximacion de Schlick
 * https://en.wikipedia.org/wiki/Schlick%27s_approximation
 */
float schlick(float cosine, float ref_idx)
{
    float r0 = (1 - ref_idx) / (1 + ref_idx);
    r0 = r0 * r0;
    return r0 + (1 - r0) * pow((1 - cosine), 5);
}

/* Verifica si un rayo puede ser refractado
 * dadas las restricciones de la ley de Snell
 * si puede ser refractado, calcula el rayo refractado
 * ver la implementacion de la clase dielectric
 *   refracted (vec3): parametro de salida, el rayo refractado
 */
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

/* Refleccion especular perfecta */
vec3 reflect(const vec3& v, const vec3& n)
{
    return v - 2 * dot(v, n) * n;
}

/* Clase virtual, todos los materiales deben implementar una funcion que
 * devuelva:
 *   - scattered (vec3): el rayo reflejado
 *   - attenuation (vec3): un factor de atenuacion por componente de color
 */
class material
{
public:
    virtual bool scatter(const ray& r,
                         const hit_record& rec,
                         vec3& attenuation,
                         ray& scattered) const = 0;
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
    dielectric(float ri) : ref_idx(ri) {}

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
        // el rayo viene de dentro o de fuera?
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
        // se puede refractar?
        if (refract(r.direction(), outward_normal, ni_over_nt, refracted))
            reflect_prob = schlick(cosine, ref_idx);
        else
            reflect_prob = 1.0;
        // refractamos o reflejamos aleatoriamente
        if (drand48() < reflect_prob)
            scattered = ray(rec.p, reflected);
        else
            scattered = ray(rec.p, refracted);
        return true;
    }

    float ref_idx;
};

#endif
