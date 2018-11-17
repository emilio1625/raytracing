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

#ifndef _HITABLE_H_
#define _HITABLE_H_

#include "hitable.hpp"

class hitable_list : public hitable
{
public:
    hitable_list() {}
    hitable_list(hitable** l, int n)
    {
        list = l;
        list_size = n;
    }
    virtual bool hit(const ray& r,
                     float tmin,
                     float tmax,
                     hit_record& rec) const;
    hitable** list;
    int list_size;
};

bool hitable_list::hit(const ray& r,
                       float t_min,
                       float t_max,
                       hit_record& rec) const
{
    hit_record temp_rec;
    bool hit_anything = false;
    double closest_so_far = t_max;
    for (int i = 0; i < list_size; i++) {
        if (list[i]->hit(r, t_min, closest_so_far, temp_rec)) {
            hit_anything = true;
            closest_so_far = temp_rec.t;
            rec = temp_rec;
        }
    }
    return hit_anything;
}

#endif