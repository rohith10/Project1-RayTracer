// CIS565 CUDA Raytracer: A parallel raytracer for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania
// This file includes code from:
// Yining Karl Li's TAKUA Render, a massively parallel pathtracing renderer: http://www.yiningkarlli.com

#ifndef INTERSECTIONS_H
#define INTERSECTIONS_H

#include "sceneStructs.h"
#include "cudaMat4.h"
#include "glm/glm.hpp"
#include "utilities.h"
#include <thrust/random.h>

//Self explanatory
__host__ __device__ glm::vec3 getPointOnRay(ray r, float t);

//LOOK: This is a custom function for multiplying cudaMat4 4x4 matrixes with vectors.
//This is a workaround for GLM matrix multiplication not working properly on pre-Fermi NVIDIA GPUs.
//Multiplies a cudaMat4 matrix and a vec4 and returns a vec3 clipped from the vec4
__host__ __device__ glm::vec3 multiplyMV(cudaMat4 m, glm::vec4 v);

//Gets sign of each component of a ray's inverse direction
__host__ __device__ glm::vec3 getSignOfRay(ray r);

//Gets 1/direction for a ray
__host__ __device__ glm::vec3 getInverseDirectionOfRay(ray r);

//TODO: IMPLEMENT THIS FUNCTION
//Cube intersection test, return -1 if no intersection, otherwise, distance to intersection
__host__ __device__ float boxIntersectionTest(staticGeom sphere, ray r, glm::vec3& intersectionPoint, glm::vec3& normal);

//LOOK: Here's an intersection test example from a sphere. Now you just need to figure out cube and, optionally, triangle.
//Sphere intersection test, return -1 if no intersection, otherwise, distance to intersection
__host__ __device__ float sphereIntersectionTest(staticGeom sphere, ray r, glm::vec3& intersectionPoint, glm::vec3& normal);

//LOOK: Example for generating a random point on an object using thrust.
//Generates a random point on a given cube
__host__ __device__ glm::vec3 getRandomPointOnCube(staticGeom cube, float randomSeed);

//Handy dandy little hashing function that provides seeds for random number generation
__host__ __device__ unsigned int hash(unsigned int a);

//Quick and dirty epsilon check
__host__ __device__ bool epsilonCheck(float a, float b);

//returns x,y,z half-dimensions of tightest bounding box
__host__ __device__ glm::vec3 getRadiuses(staticGeom geom);

//TODO: IMPLEMENT THIS FUNCTION
//Generates a random point on a given sphere
__host__ __device__ glm::vec3 getRandomPointOnSphere(staticGeom sphere, float randomSeed);

#endif


