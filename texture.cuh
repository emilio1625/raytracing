#ifndef _TEXTURE_H_
#define _TEXTURE_H_

class tex
{
public:
    __device__ virtual vec3 value(float u, float v, const vec3& p) const = 0;
};

class color_tex : public tex
{
public:
    __device__ color_tex() {}
    __device__ color_tex(vec3 c) : color(c) {}
    __device__ virtual vec3 value(float u, float v, const vec3& p) const
    {
        return color;
    }
    vec3 color;
};

class image_tex : public tex
{
public:
    __device__ image_tex() {}
    __device__ image_tex(unsigned char* pixels, int w, int h)
        : data(pixels), width(w), height(h)
    {
    }
    __device__ virtual vec3 value(float u, float v, const vec3& p) const;
    unsigned char* data;
    int width, height;
};

__device__ vec3 image_tex::value(float u, float v, const vec3& p) const
{
    int i = (u)*width;
    int j = (1 - v) * height - 0.001;
    if (i < 0)
        i = 0;
    if (j < 0)
        j = 0;
    if (i > width - 1)
        i = width - 1;
    if (j > height - 1)
        j = height - 1;
    float r = int(data[3 * i + 3 * width * j]) / 255.0;
    float g = int(data[3 * i + 3 * width * j + 1]) / 255.0;
    float b = int(data[3 * i + 3 * width * j + 2]) / 255.0;
    return vec3(r, g, b);
}

#endif
