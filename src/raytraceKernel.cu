// CIS565 CUDA Raytracer: A parallel raytracer for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania
// This file includes code from:
//       Rob Farber for CUDA-GL interop, from CUDA Supercomputing For The Masses: http://www.drdobbs.com/architecture-and-design/cuda-supercomputing-for-the-masses-part/222600097
//       Peter Kutz and Yining Karl Li's GPU Pathtracer: http://gpupathtracer.blogspot.com/
//       Yining Karl Li's TAKUA Render, a massively parallel pathtracing renderer: http://www.yiningkarlli.com

#include <thrust/device_vector.h>
#include <stdio.h>
#include <cuda.h>
#include <cmath>
#include "sceneStructs.h"
#include "glm/glm.hpp"
#include "utilities.h"
#include "raytraceKernel.h"
#include "intersections.h"
#include "interactions.h"
//#include <vector>

#if CUDA_VERSION >= 5000
    #include <helper_math.h>
#else
    #include <cutil_math.h>
#endif

const glm::vec3 bgColour = glm::vec3 (0.55, 0.25, 0);

void checkCUDAError(const char *msg) {
  cudaError_t err = cudaGetLastError();
  if( cudaSuccess != err) {
    fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) ); 
    exit(EXIT_FAILURE); 
  }
} 

//Sets up the projection half vectors.
void	setupProjection (projectionInfo &ProjectionParams, glm::vec3 eye, glm::vec3 view, glm::vec3 up, glm::vec2 fov)
{
	//Set up the projection variables:
	float	degToRad = 3.1415926 / 180.0;
	float	radToDeg = 1.0 / degToRad;

	ProjectionParams.centreProj = eye+view;
	glm::vec3	eyeToProjCentre = ProjectionParams.centreProj - eye;
	glm::vec3	A = glm::cross (ProjectionParams.centreProj, up);
	glm::vec3	B = glm::cross (A, ProjectionParams.centreProj);
	float		lenEyeToProjCentre = glm::length (eyeToProjCentre);
	
	ProjectionParams.halfVecH = glm::normalize (A) * lenEyeToProjCentre * (float)tan ((fov.x*degToRad));
	ProjectionParams.halfVecV = glm::normalize (B) * lenEyeToProjCentre * (float)tan ((fov.y*degToRad));
}

// Reflects the incidentRay around the normal.
__host__ __device__ glm::vec3 reflectRay (glm::vec3 incidentRay, glm::vec3 normal)
{
	glm::vec3 reflectedRay = incidentRay - (2.0f*glm::dot (incidentRay, normal))*normal;
	return reflectedRay;
}

//LOOK: This function demonstrates how to use thrust for random number generation on the GPU!
//Function that generates static.
__host__ __device__ glm::vec3 generateRandomNumberFromThread(glm::vec2 resolution, float time, int x, int y){
  int index = x + (y * resolution.x);
   
  thrust::default_random_engine rng(hash(index*time));
  thrust::uniform_real_distribution<float> u01(0,1);

  return glm::vec3((float) u01(rng), (float) u01(rng), (float) u01(rng));
}

//TODO: IMPLEMENT THIS FUNCTION
//Function that does the initial raycast from the camera
__host__ __device__ ray raycastFromCameraKernel(glm::vec2 resolution, float time, int x, int y, glm::vec3 eye, glm::vec3 view, glm::vec3 up, glm::vec2 fov, glm::vec3 centreProj,
													glm::vec3	halfVecH, glm::vec3 halfVecV)
{
  ray r;
  r.origin = eye;
  r.direction = glm::vec3(0,0,-1);

  float normDeviceX = (float)x / (resolution.x-1);
  float normDeviceY = 1 - ((float)y / (resolution.y-1));

  glm::vec3 P = centreProj + (2*normDeviceX - 1)*halfVecH + (2*normDeviceY - 1)*halfVecV;
  r.direction = glm::normalize (P - r.origin);

  return r;
}

//Kernel that blacks out a given image buffer
__global__ void clearImage(glm::vec2 resolution, glm::vec3* image){
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int index = x + (y * resolution.x);
    if(x<=resolution.x && y<=resolution.y){
      image[index] = glm::vec3(0,0,0);
    }
}

//Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4* PBOpos, glm::vec2 resolution, glm::vec3* image){
  
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);
  
  if(x<=resolution.x && y<=resolution.y){

      glm::vec3 color;
      color.x = image[index].x*255.0;
      color.y = image[index].y*255.0;
      color.z = image[index].z*255.0;

      if(color.x>255){
        color.x = 255;
      }

      if(color.y>255){
        color.y = 255;
      }

      if(color.z>255){
        color.z = 255;
      }
      
      // Each thread writes one pixel location in the texture (textel)
      PBOpos[index].w = 0;
      PBOpos[index].x = color.x;
      PBOpos[index].y = color.y;
      PBOpos[index].z = color.z;
  }
}

//TODO: Done!
//Core raytracer kernel
__global__ void raytraceRay(glm::vec2 resolution, float time, cameraData cam, int rayDepth, glm::vec3* colors,
                            staticGeom* geoms, int numberOfGeoms, material* textureArray, projectionInfo ProjectionParams)
{
 // __shared__ bool lightSet;
  __shared__ staticGeom light;
  __shared__ float ks;
  __shared__ float ka;
  __shared__ float kd;
  __shared__ glm::vec3 lightPos;

  if ((threadIdx.x == 0) && (threadIdx.y == 0))
  {
//	  lightSet = false;
	  ks = 0.3;
	  ka = 0.2;
	  kd = 1-ks-ka;
	  light = geoms [0];
	  lightPos = multiplyMV (light.transform, glm::vec4 (0, -0.5, 0, 1.0));
  }
  __syncthreads ();

  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);

  glm::vec3 shadedColour;

  if((x<=resolution.x && y<=resolution.y))
  {
    ray castRay = raycastFromCameraKernel(resolution, time, x, y, cam.position, cam.view, cam.up, cam.fov, 
					ProjectionParams.centreProj, ProjectionParams.halfVecH, ProjectionParams.halfVecV);
	
	glm::vec3 intrPoint = glm::vec3 (0, 0, 0);
	glm::vec3 intrNormal = glm::vec3 (0, 0, 0);

	float interceptValue = -32767;

	interceptInfo theRightIntercept;					// Stores the lowest intercept.
	theRightIntercept.interceptVal = interceptValue;	// Initially, it is empty/invalid
	theRightIntercept.intrNormal = intrNormal;			// Intially, Normal - 0,0,0

	float min = 1e6;
	// TODO: Refactor code to have Cube and Sphere intersections in different loops.
	for (int i = 0; i < numberOfGeoms; ++i)
	{
		staticGeom currentGeom = geoms [i];
		if (currentGeom.type == SPHERE)
		{	
			interceptValue = sphereIntersectionTest(currentGeom, castRay, intrPoint, intrNormal);
			if (interceptValue > 0)
			{
				if (interceptValue < min)
				{
					min = interceptValue;

					theRightIntercept.interceptVal = min;
					theRightIntercept.intrNormal = intrNormal;
					theRightIntercept.intrMaterial = textureArray [currentGeom.materialid];
				}
			}
		}
		else if (currentGeom.type == CUBE)
		{	
			interceptValue = boxIntersectionTest(currentGeom, castRay, intrPoint, intrNormal);
			if (interceptValue > 0)
			{
				if (interceptValue < min)
				{
					min = interceptValue;

					theRightIntercept.interceptVal = min;
					theRightIntercept.intrNormal = intrNormal;
					theRightIntercept.intrMaterial = textureArray [currentGeom.materialid];
				}
			}
		}

		//if (!lightSet)
		//	if (currentGeom.materialid == 8)
		//	{
		//		light = currentGeom;
		//		lightSet = true;
		//	}
	}

	glm::vec3 lightVec;
	if (theRightIntercept.interceptVal > 0)
	{
//		glm::vec3 lightVec = glm::vec3 (0, -0.5, 0);
//		lightVec = multiplyMV (light.transform, glm::vec4 (lightVec.x, lightVec.y, lightVec.z, 1.0));
		// lightVec actually stores the light's position until here.

		// Ambient shading
		shadedColour = glm::vec3 (ka * theRightIntercept.intrMaterial.color);

		// Diffuse shading
		intrPoint = castRay.origin + theRightIntercept.interceptVal*castRay.direction;
		glm::vec3 lightVec = glm::normalize (lightPos - intrPoint);	// Now it stores the vector pointing toward the light from the intersection point.
		intrNormal = glm::normalize (cam.position - intrPoint); // Refurbish intrNormal for use as the view vector.
		interceptValue = max (glm::dot (theRightIntercept.intrNormal, lightVec), (float)0); // interceptValue is reused to compute dot product.
		intrPoint = (theRightIntercept.intrMaterial.color * kd * interceptValue);			// Reuse intrPoint to store partial product (kdId) of the diffuse shading computation.
		shadedColour += multiplyVV (textureArray [light.materialid].color, intrPoint);		

		// Specular shading	
		lightVec = glm::normalize (reflectRay (-lightVec, theRightIntercept.intrNormal)); // Reuse lightVec for storing the reflection of light ray around the normal.
		interceptValue = max (glm::dot (lightVec, intrNormal), (float)0);				// Reuse interceptValue for computing dot pdt of specular.
		shadedColour += (textureArray [light.materialid].color * ks * pow (interceptValue, theRightIntercept.intrMaterial.specularExponent));

		colors [index] = shadedColour;
	}
	else
	{
		colors[index] = glm::vec3 (0, 0, 0);
	}

	// TODO: ShadowRayUnblocked for Shadows!
	lightVec = glm::normalize (reflectRay (-lightVec, theRightIntercept.intrNormal)); // Reflect the reflection of light ray around the intersection point to get the original lightVec back!
	castRay.origin = intrPoint;
	castRay.direction = lightVec;

	if (isShadowRayBlocked (castRay, lightPos, geoms, numberOfGeoms))
		colors[index] = glm::vec3 (0, 0, 0);
  }
}

__device__ bool isShadowRayBlocked (ray r, glm::vec3 lightPos, staticGeom *geomsList, int nGeoms)
{
	float min = 1e6, interceptValue;
	glm::vec3 intrPoint, intrNormal;
	for (int i = 0; i < nGeoms; ++i)
	{
		staticGeom currentGeom = geomsList [i];
		if (currentGeom.type == SPHERE)
		{	
			interceptValue = sphereIntersectionTest(currentGeom, r, intrPoint, intrNormal);
			if (interceptValue > 0)
			{
				if (interceptValue < min)
					min = interceptValue;
			}
		}
		else if (currentGeom.type == CUBE)
		{	
			interceptValue = boxIntersectionTest(currentGeom, r, intrPoint, intrNormal);
			if (interceptValue > 0)
			{
				if (interceptValue < min)
					min = interceptValue;
			}
		}
	}

//	if (min > 0)
		if (glm::length (lightPos - r.origin) > min)
			return true;
	return false;
}

// At each pixel, trace a shadow ray to the light and see if it intersects something else.
__global__ void		shadowFeeler (glm::vec3 startPoint, glm::vec3 lightPosition, glm::vec3 *colorBuffer, staticGeom *geoms, int nGeoms)
{
	;
}

// This function intersects a ray r with all the cubes in the scene and returns the lowest positive intersection value.
//__device__ float intersectRayWithCubes (ray r, staticGeom *cubesList, int nCubes)
//{
//	float min = -0.001;
//	for (int i = 0; i < nCubes; i ++)
//	{
//		staticGeom currentGeom = cubesList [i];
//		
//		interceptValue = boxIntersectionTest(currentGeom, castRay, intrPoint, intrNormal);
//		if (interceptValue < abs (min))
//		{
//			min = interceptValue;
//
//			theRightIntercept.interceptVal = min;
//			theRightIntercept.intrNormal = intrNormal;
//			theRightIntercept.intrMaterial = textureArray [currentGeom.materialid];
//		}
//	}
//}

//// This funcion intersects a ray r with all the spheres in the scene and returns the lowest positive intersection value.
//__device__ float intersectRayWithSpheres (ray r, staticGeom *spheresList, int nSpheres)
//{
//	float min = -0.001;
//	for (int i = 0; i < nCubes; i ++)
//	{	
//		staticGeom currentGeom = cubesList [i];
//
//		interceptValue = sphereIntersectionTest(currentGeom, castRay, intrPoint, intrNormal);
//		if (interceptValue <  abs (min))
//		{
//			min = interceptValue;
//
//			theRightIntercept.interceptVal = min;
//			theRightIntercept.intrNormal = intrNormal;
//			theRightIntercept.intrMaterial = textureArray [currentGeom.materialid];
//		}
//	}
//}

// Kernel for shading cubes.
__global__ void		cubeShade  (glm::vec2 resolution, int nIteration, cameraData camDetails, int rayDepth, 
								glm::vec3 *colorBuffer, staticGeom *cubesList, int nCubes, material *textureData, projectionInfo ProjParams)
{
	;
}

// Kernel for shading spheres.
__global__ void		sphereShade  (glm::vec2 resolution, int nIteration, cameraData camDetails, int rayDepth, 
								glm::vec3 *colorBuffer, staticGeom *spheresList, int nSpheres, material *textureData, projectionInfo ProjParams)
{
	;
}

//TODO: Almost Done!
// Wrapper for the __global__ call that sets up the kernel calls and does a ton of memory management
void cudaRaytraceCore(uchar4* PBOpos, camera* renderCam, int frame, int iterations, material* materials, int numberOfMaterials, geom* geoms, int numberOfGeoms){
  
  int traceDepth = 1; //determines how many bounces the raytracer traces
  projectionInfo	ProjectionParams;
//  setupProjection (ProjectionParams, renderCam->positions [frame], renderCam->ups [frame], renderCam->views [frame], renderCam->fov);
  float degToRad = 3.1415926 / 180.0;
  ProjectionParams.centreProj = renderCam->positions [frame]+renderCam->views [frame];
	glm::vec3	eyeToProjCentre = ProjectionParams.centreProj - renderCam->positions [frame];
	glm::vec3	A = glm::cross (eyeToProjCentre, renderCam->ups [frame]);
	glm::vec3	B = glm::cross (A, eyeToProjCentre);
	float		lenEyeToProjCentre = glm::length (eyeToProjCentre);
	
	ProjectionParams.halfVecH = glm::normalize (A) * lenEyeToProjCentre * (float)tan ((renderCam->fov.x*degToRad) / 2.0);
	ProjectionParams.halfVecV = glm::normalize (B) * lenEyeToProjCentre * (float)tan ((renderCam->fov.y*degToRad) / 2.0);

  // set up crucial magic
  int tileSize = 8;
  dim3 threadsPerBlock(tileSize, tileSize);
  dim3 fullBlocksPerGrid((int)ceil(float(renderCam->resolution.x)/float(tileSize)), (int)ceil(float(renderCam->resolution.y)/float(tileSize)));
  
  //send image to GPU
  glm::vec3* cudaimage = NULL;
  cudaMalloc((void**)&cudaimage, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3));
  cudaMemcpy( cudaimage, renderCam->image, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3), cudaMemcpyHostToDevice);
  
  //package geometry and materials and sent to GPU
  staticGeom* geomList = new staticGeom[numberOfGeoms];
  sceneInfo		primCounts;
  
  int count = 1;
  bool lightSet = false;
  for(int i=0; i<numberOfGeoms; i++)
  {
	  if ((geoms [i].materialid == 8) && !lightSet)
	  {
		staticGeom newStaticGeom;
		newStaticGeom.type = geoms[i].type;
		newStaticGeom.materialid = geoms[i].materialid;
		newStaticGeom.translation = geoms[i].translations[frame];
		newStaticGeom.rotation = geoms[i].rotations[frame];
		newStaticGeom.scale = geoms[i].scales[frame];
		newStaticGeom.transform = geoms[i].transforms[frame];
		newStaticGeom.inverseTransform = geoms[i].inverseTransforms[frame];
		geomList[0] = newStaticGeom;
		
		lightSet = true;
	  }

	  else if (geoms [i].type == CUBE)
	  {
		staticGeom newStaticGeom;
		newStaticGeom.type = geoms[i].type;
		newStaticGeom.materialid = geoms[i].materialid;
		newStaticGeom.translation = geoms[i].translations[frame];
		newStaticGeom.rotation = geoms[i].rotations[frame];
		newStaticGeom.scale = geoms[i].scales[frame];
		newStaticGeom.transform = geoms[i].transforms[frame];
		newStaticGeom.inverseTransform = geoms[i].inverseTransforms[frame];
		geomList[count] = newStaticGeom;
		count ++;
	  }
  }

  primCounts.nCubes = count;
  
  for(int i=0; i<numberOfGeoms; i++)
  {
	  if (geoms [i].type == SPHERE)
	  {
		staticGeom newStaticGeom;
		newStaticGeom.type = geoms[i].type;
		newStaticGeom.materialid = geoms[i].materialid;
		newStaticGeom.translation = geoms[i].translations[frame];
		newStaticGeom.rotation = geoms[i].rotations[frame];
		newStaticGeom.scale = geoms[i].scales[frame];
		newStaticGeom.transform = geoms[i].transforms[frame];
		newStaticGeom.inverseTransform = geoms[i].inverseTransforms[frame];
		geomList[count] = newStaticGeom;
		count ++;
	  }
  }

  primCounts.nSpheres = count - primCounts.nCubes;

  if (!lightSet)
  {
		staticGeom newStaticGeom;
		newStaticGeom.type = geoms[0].type;
		newStaticGeom.materialid = geoms[0].materialid;
		newStaticGeom.translation = geoms[0].translations[frame];
		newStaticGeom.rotation = geoms[0].rotations[frame];
		newStaticGeom.scale = geoms[0].scales[frame];
		newStaticGeom.transform = geoms[0].transforms[frame];
		newStaticGeom.inverseTransform = geoms[0].inverseTransforms[frame];
		geomList[0] = newStaticGeom;
  }

  staticGeom* cudageoms = NULL;
  cudaMalloc((void**)&cudageoms, numberOfGeoms*sizeof(staticGeom));
  cudaMemcpy( cudageoms, geomList, numberOfGeoms*sizeof(staticGeom), cudaMemcpyHostToDevice);
  
  material		*materialColours = NULL;
  cudaError_t returnCode = cudaMalloc((void**)&materialColours, numberOfMaterials*sizeof(material));
  if (returnCode != cudaSuccess)
  {
	  std::cout << "\nError while trying to send texture data to the GPU!";
	  std::cin.get ();

	  if (cudaimage)
		  cudaFree( cudaimage );
	  if (cudageoms)
		  cudaFree( cudageoms );
	  if (materialColours)
		  cudaFree (materialColours);
	  
	  cudaimage = NULL;
	  cudageoms = NULL;
	  materialColours = NULL;
	  exit (EXIT_FAILURE);
  }
  else
	  cudaMemcpy( materialColours, materials, numberOfMaterials*sizeof(material), cudaMemcpyHostToDevice);

  //package camera
  cameraData cam;
  cam.resolution = renderCam->resolution;
  cam.position = renderCam->positions[frame];
  cam.view = renderCam->views[frame];
  cam.up = renderCam->ups[frame];
  cam.fov = renderCam->fov;

//  cudaPrintfInit ();
  //kernel launches
  raytraceRay<<<fullBlocksPerGrid, threadsPerBlock>>>(renderCam->resolution, (float)iterations, cam, traceDepth, cudaimage, cudageoms, numberOfGeoms, materialColours, ProjectionParams);
  
  sendImageToPBO<<<fullBlocksPerGrid, threadsPerBlock>>>(PBOpos, renderCam->resolution, cudaimage);

  //retrieve image from GPU
  cudaMemcpy( renderCam->image, cudaimage, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3), cudaMemcpyDeviceToHost);

 // cudaPrintfDisplay (stdout, true);
 // cudaPrintfEnd ();
  //free up stuff, or else we'll leak memory like a madman
   if (cudaimage)
		cudaFree( cudaimage );
   if (cudageoms)
		cudaFree( cudageoms );
   if (materialColours)
		cudaFree (materialColours);

   cudaimage = NULL;
   cudageoms = NULL;
   materialColours = NULL;

 // make certain the kernel has completed
  cudaThreadSynchronize();
  
  //  cudaFree( cudaimage );
//  cudaFree( cudageoms );
//  cudaFree (materialColours);
  delete geomList;

  checkCUDAError("Kernel failed!");
}
