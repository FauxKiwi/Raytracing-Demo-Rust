/*
 * Copyright LWJGL. All rights reserved.
 * License terms: https://www.lwjgl.org/license
 */
#version 430 core

layout(binding = 0, rgba32f) uniform image2D framebufferImage;

uniform vec3 eye;
uniform vec3 ray00;
uniform vec3 ray01;
uniform vec3 ray10;
uniform vec3 ray11;

uniform float blendFactor;
uniform float time;
uniform int bounceCount;

struct box {
  vec3 min;
  vec3 max;
  int mat;
};

#define MAX_SCENE_BOUNDS 100.0
#define NUM_BOXES 7

const box boxes[] = {
  /* The ground */
  {vec3(-5.0, -0.1, -5.0), vec3(5.0, 0.0, 5.0), 0},
  /* Box in the middle */
  {vec3(-0.5, 0.0, -0.5), vec3(0.5, 1.0, 0.5), 1},
  /* Left wall */
  {vec3(-5.1, 0.0, -5.0), vec3(-5.0, 5.0, 5.0), 2},
  /* Right wall */
  {vec3(5.0, 0.0, -5.0), vec3(5.1, 5.0, 5.0), 3},
  /* Back wall */
  {vec3(-5.0, 0.0, -5.1), vec3(5.0, 5.0, -5.0), 0},
  /* Front wall */
  {vec3(-5.0, 0.0, 5.0), vec3(5.0, 5.0, 5.1), 0},
  /* Top wall */
  {vec3(-5.0, 5.0, -5.0), vec3(5.0, 5.1, 5.0), 4}
};

#define EPSILON 0.00001
#define LIGHT_RADIUS 0.4

#define LIGHT_BASE_INTENSITY 20.0
const vec3 lightCenterPosition = vec3(1.5, 2.9, 3);
const vec4 lightColor = vec4(1);

vec3 random3(vec3 f);
vec3 randomDiskPoint(vec3 rand, vec3 n);
vec3 randomHemispherePoint(vec3 rand, vec3 n);
vec3 randomCosineWeightedHemispherePoint(vec3 rand, vec3 n);

struct hitinfo {
  vec3 normal;
  float near;
  float far;
  int bi;
};

/*
 * We need random values every now and then.
 * So, they will be precomputed for each ray we trace and
 * can be used by any function.
 */
vec3 rand;
vec3 cameraUp;

vec2 intersectBox(vec3 origin, vec3 dir, const box b, out vec3 normal) {
  vec3 tMin = (b.min - origin) / dir;
  vec3 tMax = (b.max - origin) / dir;
  vec3 t1 = min(tMin, tMax);
  vec3 t2 = max(tMin, tMax);
  float tNear = max(max(t1.x, t1.y), t1.z);
  float tFar = min(min(t2.x, t2.y), t2.z);
  normal = vec3(equal(t1, vec3(tNear))) * sign(-dir);
  return vec2(tNear, tFar);
}

vec4 colorOfBox(const box b) {
  vec4 col;
  if (b.mat == 0) {
    col = vec4(1.0, 1.0, 1.0, 1.0);
  } else if (b.mat == 1) {
    col = vec4(1.0, 0.2, 0.2, 1.0);
  } else if (b.mat == 2) {
    col = vec4(0.2, 0.2, 1.0, 1.0);
  } else if (b.mat == 3) {
    col = vec4(0.2, 1.0, 0.2, 1.0);
  } else {
    col = vec4(0.5, 0.5, 0.5, 1.0);
  }
  return col;
}

bool intersectBoxes(vec3 origin, vec3 dir, out hitinfo info) {
  float smallest = MAX_SCENE_BOUNDS;
  bool found = false;
  vec3 normal;
  for (int i = 0; i < NUM_BOXES; i++) {
    vec2 lambda = intersectBox(origin, dir, boxes[i], normal);
    if (lambda.y >= 0.0 && lambda.x < lambda.y && lambda.x < smallest) {
      info.normal = normal;
      info.near = lambda.x;
      info.far = lambda.y;
      info.bi = i;
      smallest = lambda.x;
      found = true;
    }
  }
  return found;
}

vec4 trace(vec3 origin, vec3 dir) {
  hitinfo i;
  vec4 accumulated = vec4(0.0);
  vec4 attenuation = vec4(1.0);
  for (int bounce = 0; bounce < bounceCount; bounce++) {
    if (intersectBoxes(origin, dir, i)) {
      box b = boxes[i.bi];
      vec3 hitPoint = origin + i.near * dir;
      vec3 normal = i.normal;
      vec3 lightNormal = normalize(hitPoint - lightCenterPosition);
      vec3 lightPosition = lightCenterPosition + randomDiskPoint(rand, lightNormal) * LIGHT_RADIUS;
      vec3 shadowRayDir = lightPosition - hitPoint;
      vec3 shadowRayStart = hitPoint + normal * EPSILON;
      hitinfo shadowRayInfo;
      bool lightObstructed = intersectBoxes(shadowRayStart, shadowRayDir, shadowRayInfo);
      attenuation *= colorOfBox(b);
      if (shadowRayInfo.near >= 1.0) {
        float cosineFallOff = max(0.0, dot(normal, normalize(shadowRayDir)));
        float oneOverR2 = 1.0 / dot(shadowRayDir, shadowRayDir);
        accumulated += attenuation * vec4(lightColor * LIGHT_BASE_INTENSITY * cosineFallOff * oneOverR2);
      }
      origin = shadowRayStart;
      dir = randomCosineWeightedHemispherePoint(rand, normal);
      //dir = randomHemispherePoint(rand, normal);
      attenuation *= dot(normal, dir);
    } else {
      break;
    }
  }
  return accumulated;
}

layout (local_size_x = 16, local_size_y = 8) in;

void main(void) {
  ivec2 pix = ivec2(gl_GlobalInvocationID.xy);
  ivec2 size = imageSize(framebufferImage);
  if (pix.x >= size.x || pix.y >= size.y) {
    return;
  }
  vec2 pos = (vec2(pix) + vec2(0.5, 0.5)) / vec2(size.x, size.y);
  vec4 color = vec4(0.0, 0.0, 0.0, 1.0);
  cameraUp = normalize(ray01 - ray00);
  for (int s = 0; s < 1; s++) {
    rand = random3(vec3(pix, time + float(s)));
    vec2 jitter = rand.xy / vec2(size);
    vec2 p = pos + jitter;
    vec3 dir = mix(mix(ray00, ray01, p.y), mix(ray10, ray11, p.y), p.x);
    color += trace(eye, dir);
  }
  color /= 1;
  vec4 oldColor = vec4(0.0);
  if (blendFactor > 0.0) {
    /* Without explicitly disabling imageLoad for
     * the first frame, we get VERY STRANGE corrupted image!
     * 'mix' SHOULD mix oldColor out, but strangely it does not!
     */
    oldColor = imageLoad(framebufferImage, pix);
  }
  vec4 finalColor = mix(color, oldColor, blendFactor);
  imageStore(framebufferImage, pix, finalColor);
}