// CIS565 CUDA Raytracer: A parallel raytracer for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania
// This file includes code from:
//       Rob Farber for CUDA-GL interop, from CUDA Supercomputing For The Masses: http://www.drdobbs.com/architecture-and-design/cuda-supercomputing-for-the-masses-part/222600097
//       Peter Kutz and Yining Karl Li's GPU Pathtracer: http://gpupathtracer.blogspot.com/
//       Yining Karl Li's TAKUA Render, a massively parallel pathtracing renderer: http://www.yiningkarlli.com

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
#include <thrust/device_vector.h>
#include <thrust/fill.h>

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
	
	ProjectionParams.halfVecH = glm::normalize (A) * lenEyeToProjCentre * (float)tan ((fov.x*degToRad) / 2.0);
	ProjectionParams.halfVecV = glm::normalize (B) * lenEyeToProjCentre * (float)tan ((fov.y*degToRad) / 2.0);
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

 // float	degToRad = 3.1415926 / 180.0;
 // float	radToDeg = 1.0 / degToRad;

	//ProjectionParams.centreProj = eye+view;
	//glm::vec3	eyeToProjCentre = ProjectionParams.centreProj - eye;
	//glm::vec3	A = glm::cross (ProjectionParams.centreProj, up);
	//glm::vec3	B = glm::cross (A, ProjectionParams.centreProj);
	//float		lenEyeToProjCentre = glm::length (eyeToProjCentre);
	//
	//ProjectionParams.halfVecH = glm::normalize (A) * lenEyeToProjCentre * (float)tan ((fov.x*degToRad) / 2.0);
	//ProjectionParams.halfVecV = glm::normalize (B) * lenEyeToProjCentre * (float)tan ((fov.y*degToRad) / 2.0);


  float normDeviceX = (float)x / (resolution.x-1);
  float normDeviceY = (float)y / (resolution.y-1);

  glm::vec3 P = /*ProjectionParams.*/centreProj + (2*normDeviceX - 1)*/*ProjectionParams.*/halfVecH + (2*normDeviceY - 1)*/*ProjectionParams.*/halfVecV;
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

//TODO: IMPLEMENT THIS FUNCTION
//Core raytracer kernel
__global__ void raytraceRay(glm::vec2 resolution, float time, cameraData cam, int rayDepth, glm::vec3* colors,
                            staticGeom* geoms, int numberOfGeoms, glm::vec3* textureArray, projectionInfo ProjectionParams){

  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);
  staticGeom *light = NULL;

  thrust::device_vector<interceptInfo> interceptVec;

  interceptInfo theRightIntercept;					// Stores the lowest intercept.
  theRightIntercept.interceptVal = -32767;			// Initially, it is empty/invalid
  theRightIntercept.intrNormal = intrNormal;		// Normal - 0,0,0
  theRightIntercept.intrMaterial = intrPoint;		// Colour - black;

  if((x<=resolution.x && y<=resolution.y))
  {
	  ray castRay = raycastFromCameraKernel(resolution, time, x, y, cam.position, cam.view, cam.up, cam.fov, 
					ProjectionParams.centreProj, ProjectionParams.halfVecH, ProjectionParams.halfVecV);
//	
	glm::vec3 materialColour = glm::vec3 (0, 0, 0);
	glm::vec3 intrPoint = glm::vec3 (0, 0, 0);
	glm::vec3 intrNormal = glm::vec3 (0, 0, 0);
	
	for (int i = 0; i < numberOfGeoms; ++i)
	{
		if (geoms [i].type == SPHERE)
		{	
			interceptValue = sphereIntersectionTest(geoms [i], castRay, intrPoint, intrNormal);
			if (interceptValue > 0)
			{
				materialColour = textureArray [geoms [i].materialid];

				interceptInfo thisIntercept;
				thisIntercept.interceptVal = interceptValue;
				thisIntercept.intrNormal = intrNormal;
				thisIntercept.intrMaterial = materialColour;
				
				interceptVec.push_back (thisIntercept);
			}
		}
		else if (geoms [i].type == CUBE)
		{	
			interceptValue = boxIntersectionTest(geoms [i], castRay, intrPoint, intrNormal);
			if (interceptValue > 0)
			{
				materialColour = textureArray [geoms [i].materialid];

				interceptInfo thisIntercept;
				thisIntercept.interceptVal = interceptValue;
				thisIntercept.intrNormal = intrNormal;
				thisIntercept.intrMaterial = materialColour;
				
				interceptVec.push_back (thisIntercept);
			}
		}
//		materialColour = textureArray [y%numberOfGeoms];
//		colors[index].y = fabs (castRay.direction.y);
//		colors[index].z = fabs (castRay.direction.z);//generateRandomNumberFromThread(resolution, time, x, y);//materialColour;
	}

	float min = 1e6;
	for (int i = 0; i < interceptVec.size (); i++)
	{
		if (interceptVec [i].interceptVal < min)
		{
			min = interceptVec [i].interceptVal;

			theRightIntercept.interceptVal = min;
			theRightIntercept.intrNormal = interceptVec [i].intrNormal;
			theRightIntercept.intrMaterial = interceptVec [i].intrMaterial;
		}
	}

	for (int i = 0; i < numberOfGeoms; ++i)
	{
		if (geoms [i].materialid == 8)
			light = &geoms [i];
	}

	if ((light) && (interceptVec.size() > 0))
	{
		// Ambient shading
		colors [index] = glm::vec3 (0.25 * theRightIntercept.intrMaterial.x, 0.25 * theRightIntercept.intrMaterial.y, 0.25 * theRightIntercept.intrMaterial.z);

		//Diffuse shading
		intrPoint = castRay.origin + theRightIntercept.interceptVal*castRay.direction;
		glm::vec3 lightVec = light->translation - intrPoint;
		float dotPdt = max (glm::dot (theRightIntercept.intrNormal, -lightVec), (float)0);
		colors [index] += glm::vec3 (textureArray [light->materialid].x * theRightIntercept.intrMaterial.x, 
									textureArray [light->materialid].y * theRightIntercept.intrMaterial.y, 
									textureArray [light->materialid].z * theRightIntercept.intrMaterial.z) * dotPdt;
	}
	else
	{
		colors[index] = materialColour;
	}
 //generateRandomNumberFromThread(resolution, time, x, y);
  }
}

//TODO: FINISH THIS FUNCTION
// Wrapper for the __global__ call that sets up the kernel calls and does a ton of memory management
void cudaRaytraceCore(uchar4* PBOpos, camera* renderCam, int frame, int iterations, material* materials, int numberOfMaterials, geom* geoms, int numberOfGeoms){
  
  int traceDepth = 1; //determines how many bounces the raytracer traces
  projectionInfo	ProjectionParams;
//  setupProjection (ProjectionParams, renderCam->positions [frame], renderCam->ups [frame], renderCam->views [frame], renderCam->fov);
  float degToRad = 3.1415926 / 180.0;
  ProjectionParams.centreProj = renderCam->positions [frame]+renderCam->views [frame];
	glm::vec3	eyeToProjCentre = ProjectionParams.centreProj - renderCam->positions [frame];
	glm::vec3	A = glm::cross (ProjectionParams.centreProj, renderCam->ups [frame]);
	glm::vec3	B = glm::cross (A, ProjectionParams.centreProj);
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
  for(int i=0; i<numberOfGeoms; i++){
    staticGeom newStaticGeom;
    newStaticGeom.type = geoms[i].type;
    newStaticGeom.materialid = geoms[i].materialid;
    newStaticGeom.translation = geoms[i].translations[frame];
    newStaticGeom.rotation = geoms[i].rotations[frame];
    newStaticGeom.scale = geoms[i].scales[frame];
    newStaticGeom.transform = geoms[i].transforms[frame];
    newStaticGeom.inverseTransform = geoms[i].inverseTransforms[frame];
    geomList[i] = newStaticGeom;
  }
  
  staticGeom* cudageoms = NULL;
  cudaMalloc((void**)&cudageoms, numberOfGeoms*sizeof(staticGeom));
  cudaMemcpy( cudageoms, geomList, numberOfGeoms*sizeof(staticGeom), cudaMemcpyHostToDevice);
  
  glm::vec3		*materialColours = NULL;
  cudaError_t returnCode = cudaMalloc((void**)&materialColours, numberOfMaterials*sizeof(glm::vec3));
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
	  
	  exit (EXIT_FAILURE);
  }
  else
  {
	  for (int loopVar = 0; loopVar < numberOfMaterials; ++loopVar)
	  {
		  glm::vec3 *index = materialColours+loopVar;
		  cudaMemcpy( index, &(materials [loopVar].color), sizeof(glm::vec3), cudaMemcpyHostToDevice);
	  }
  }

  //package camera
  cameraData cam;
  cam.resolution = renderCam->resolution;
  cam.position = renderCam->positions[frame];
  cam.view = renderCam->views[frame];
  cam.up = renderCam->ups[frame];
  cam.fov = renderCam->fov;

  cudaPrintfInit ();
  //kernel launches
  raytraceRay<<<fullBlocksPerGrid, threadsPerBlock>>>(renderCam->resolution, (float)iterations, cam, traceDepth, cudaimage, cudageoms, numberOfGeoms, materialColours, ProjectionParams);
  
  sendImageToPBO<<<fullBlocksPerGrid, threadsPerBlock>>>(PBOpos, renderCam->resolution, cudaimage);

  //retrieve image from GPU
  cudaMemcpy( renderCam->image, cudaimage, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3), cudaMemcpyDeviceToHost);

  // make certain the kernel has completed
  cudaThreadSynchronize();
  cudaPrintfDisplay (stdout, true);
  cudaPrintfEnd ();
  //free up stuff, or else we'll leak memory like a madman
   if (cudaimage)
		cudaFree( cudaimage );
   if (cudageoms)
		cudaFree( cudageoms );
   if (materialColours)
		cudaFree (materialColours);
//  cudaFree( cudaimage );
//  cudaFree( cudageoms );
//  cudaFree (materialColours);
  delete geomList;

  checkCUDAError("Kernel failed!");
}
