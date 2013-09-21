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
__global__ void sendImageToPBO(uchar4* PBOpos, glm::vec2 resolution, glm::vec3* image, int nLights){
  
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);
  
  if(x<=resolution.x && y<=resolution.y){
	  image [index] /= nLights;
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

__device__ interceptInfo getIntercept (staticGeom * geoms, sceneInfo objectCountInfo, ray castRay, material* textureArray)
{
	glm::vec3 intrPoint = glm::vec3 (0, 0, 0);
	glm::vec3 intrNormal = glm::vec3 (0, 0, 0);

	float interceptValue = -32767;

	material newMaterial;
	newMaterial.color = glm::vec3 (0,0,0);
	newMaterial.specularExponent = 1.0;
	newMaterial.hasReflective = 0.0;
	newMaterial.hasRefractive = 0.0;

	interceptInfo theRightIntercept;					// Stores the lowest intercept.
	theRightIntercept.interceptVal = interceptValue;	// Initially, it is empty/invalid
	theRightIntercept.intrNormal = intrNormal;			// Intially, Normal - 0,0,0
	theRightIntercept.intrMaterial = newMaterial;

	float min = 1e6;
	// Two different loops to intersect ray with cubes and spheres.
	for (int i = 0; i < objectCountInfo.nCubes; ++i)
	{
		staticGeom currentGeom = geoms [i];

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

	for (int i = objectCountInfo.nCubes; i <= (objectCountInfo.nCubes+objectCountInfo.nSpheres); ++i)
	{
		staticGeom currentGeom = geoms [i];

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

	return theRightIntercept;
}

__device__ glm::vec3 calcShade (interceptInfo theRightIntercept, glm::vec3 lightVec, glm::vec3 eye, ray castRay, material* textureArray, float ka, float ks, float kd, glm::vec3 lightCol)
{
	glm::vec3 shadedColour = glm::vec3 (0,0,0);
	if (theRightIntercept.interceptVal > 0)
	{
		// Ambient shading
		shadedColour = ka * theRightIntercept.intrMaterial.color;

		// Diffuse shading
		glm::vec3 intrPoint = castRay.origin + theRightIntercept.interceptVal*castRay.direction;	// The intersection point.
		glm::vec3 intrNormal = glm::normalize (eye - intrPoint); // intrNormal is the view vector.
		float interceptValue = max (glm::dot (theRightIntercept.intrNormal, lightVec), (float)0); // Diffuse Lighting is given by (N.L); N being normal at intersection pt and L being light vector.
		intrPoint = (theRightIntercept.intrMaterial.color * kd * interceptValue);			// Reuse intrPoint to store partial product (kdId) of the diffuse shading computation.
		shadedColour += multiplyVV (lightCol, intrPoint);		// shadedColour will have diffuse shaded colour. 

		// Specular shading
		lightVec = glm::normalize (reflectRay (-lightVec, theRightIntercept.intrNormal)); // Reuse lightVec for storing the reflection of light vector around the normal.
		interceptValue = max (glm::dot (lightVec, intrNormal), (float)0);				// Reuse interceptValue for computing dot pdt of specular.
		shadedColour += (lightCol * ks * pow (interceptValue, theRightIntercept.intrMaterial.specularExponent));
	}
	
	return	shadedColour;
}

//TODO: Done!
//Core raytracer kernel
__global__ void raytraceRay(glm::vec2 resolution, float time, cameraData cam, int rayDepth, glm::vec3* colors,
                            staticGeom* geoms, sceneInfo objectCountInfo, material* textureArray, projectionInfo ProjectionParams,
							glm::vec3 lightPosition)
{
  __shared__ staticGeom light;
  __shared__ float ks;
  __shared__ float ka;
  __shared__ float kd;
  __shared__ glm::vec3 lightPos;
  __shared__ glm::vec3 lightCol;
  __shared__ float nLights;
  __shared__ int sqrtLights;
  __shared__ float stepSize;

  if ((threadIdx.x == 0) && (threadIdx.y == 0))
  {
	  ks = 0.5;
	  ka = 0.1;
	  kd = 1-ks-ka;
	  nLights = 64;
	  sqrtLights = sqrt (nLights);
	  stepSize = 1.0/(sqrtLights-1);
	  light = geoms [0];
	  lightPos = /*multiplyMV (light.transform, */lightPosition/*)*/;
	  lightCol = (textureArray [light.materialid].color /** textureArray [light.materialid].emittance*/);
  }
  __syncthreads ();

  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);

  glm::vec3 shadedColour = glm::vec3 (0);

  if((x<=resolution.x && y<=resolution.y))
  {
    ray castRay = raycastFromCameraKernel(resolution, time, x, y, cam.position, cam.view, cam.up, cam.fov, 
					ProjectionParams.centreProj, ProjectionParams.halfVecH, ProjectionParams.halfVecV);

	interceptInfo theRightIntercept = getIntercept (geoms, objectCountInfo, castRay, textureArray);
	glm::vec3 lightVec; 
//	for (int i = 0; i < nLights; ++ i)
//	{
//		glm::vec3 tmpLightPos = multiplyMV (light.transform, lightPosition/*glm::vec3 (lightPos.x+ ((i%sqrtLights)*stepSize), lightPos.y, lightPos.z + ((i/sqrtLights)*stepSize)*/));
		lightVec = glm::normalize (lightPosition - (castRay.origin + (castRay.direction*theRightIntercept.interceptVal)));
		shadedColour += calcShade (theRightIntercept, lightVec, cam.position, castRay, textureArray, ka, ks, kd, lightCol);
//	}

//	shadedColour /= nLights;
	glm::vec3 rightnormal = theRightIntercept.intrNormal;

	// Specular reflection
	// -------------------
	castRay.origin += theRightIntercept.interceptVal*castRay.direction;	// Store the intersection point in castRay.
	castRay.direction = castRay.origin - cam.position;		// We have ray starting at camera and pointing toward intersection point
	castRay.direction = glm::normalize (reflectRay (castRay.direction, theRightIntercept.intrNormal)); // Reflect around intersection normal to compute shade of reflections. 
	
	// Find the intersection point of reflected ray and calculate shade there.
	float hasReflective = theRightIntercept.intrMaterial.hasReflective;
	theRightIntercept = getIntercept (geoms, objectCountInfo, castRay, textureArray);
	// Use only a point light to calculate the shade of reflection, since it doesn't matter much anyway.
	lightVec = glm::normalize (lightPosition - (castRay.origin + (castRay.direction*theRightIntercept.interceptVal)));
	if (hasReflective)
		shadedColour = ((shadedColour * (float)0.92) + (calcShade (theRightIntercept, lightVec, cam.position, castRay, textureArray, ka, ks, kd, lightCol) * (float)0.08));

//	 Shadow shading
//	 --------------
	castRay.origin += ((float)0.04*rightnormal);		// Perturb the intersection pt along the normal a slight distance 
														// to avoid self intersection.
	glm::vec3 shadowColour = glm::vec3 (0);
//	for (int i = 0; i < nLights; ++ i)
//	{
//		lightVec = multiplyMV (light.transform, glm::vec4 (lightPos.x + ((i%sqrtLights)*stepSize), lightPos.y, lightPos.z + ((i/sqrtLights)*stepSize), 1.0));
		castRay.direction = glm::normalize (lightPosition - castRay.origin);

		if (isShadowRayBlocked (castRay, lightPosition, geoms, objectCountInfo))
			/*shadowColour +=*/shadedColour = ka * theRightIntercept.intrMaterial.color;	// If point in shadow, add ambient colour to shadowColour
//		else
//			shadowColour += shadedColour;								// Otherwise, add the computed shade.
//	}
//	shadedColour = shadowColour/nLights;

	colors [index] += shadedColour;
  }
}

__device__ bool isShadowRayBlocked (ray r, glm::vec3 lightPos, staticGeom *geomsList, sceneInfo objectCountInfo)
{
	float min = 1e6, interceptValue;
	glm::vec3 intrPoint, intrNormal;
	
	for (int i = 0; i < objectCountInfo.nCubes; ++i)
	{
		staticGeom currentGeom = geomsList [i];
		interceptValue = boxIntersectionTest(currentGeom, r, intrPoint, intrNormal);
		if (interceptValue > 0)
		{
			if (interceptValue < min)
				min = interceptValue;
		}
	}

	for (int i = objectCountInfo.nCubes; i <= (objectCountInfo.nCubes+objectCountInfo.nSpheres); ++i)
	{
		staticGeom currentGeom = geomsList [i];
		interceptValue = sphereIntersectionTest(currentGeom, r, intrPoint, intrNormal);
		if (interceptValue > 0)
		{
			if (interceptValue < min)
				min = interceptValue;
		}
	}

//	if (min > 0)
		if (glm::length (lightPos - r.origin) > (min+0.1))
			return true;
	return false;
}

// At each pixel, trace a shadow ray to the light and see if it intersects something else.
__global__ void		shadowFeeler (glm::vec2 resolution, float time, cameraData cam, int rayDepth, glm::vec3* colors,
                            staticGeom* geoms, sceneInfo objectCountInfo, material* textureArray, projectionInfo ProjectionParams, 
							renderInfo* renderParams)
{
	__shared__ staticGeom light;
	__shared__ float ks;
	__shared__ float ka;
	__shared__ float kd;
	__shared__ glm::vec3 lightPos;
	__shared__ glm::vec3 lightCol;
	__shared__ float nLights;
	__shared__ int sqrtLights;
	__shared__ float stepSize;

	if ((threadIdx.x == 0) && (threadIdx.y == 0))
	{
		ks = renderParams->ks;
		ka = renderParams->ka;
		kd = renderParams->kd;
		nLights = renderParams->nLights;
		sqrtLights = renderParams->sqrtLights;
		stepSize = renderParams->lightStepSize;
		light = geoms [0];
		lightPos = renderParams->lightPos;
		lightCol = renderParams->lightCol;
	}
	__syncthreads ();

	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int index = x + (y * resolution.x);
	
	if ((x <= resolution.x) && (y <= resolution.y)) 
	{
		ray castRay = raycastFromCameraKernel(resolution, time, x, y, cam.position, cam.view, cam.up, cam.fov, 
					ProjectionParams.centreProj, ProjectionParams.halfVecH, ProjectionParams.halfVecV);

		interceptInfo theRightIntercept = getIntercept (geoms, objectCountInfo, castRay, textureArray);
		glm::vec3 lightVec; 
	
	//	Shadow shading
	//	--------------
		// Perturb the intersection pt along the normal a slight distance to avoid self intersection. 
		castRay.origin += (castRay.direction * (float)(theRightIntercept.interceptVal - 0.001));
															
		glm::vec3 shadedColour = colors [index];
		glm::vec3 shadowColour = glm::vec3 (0);
		for (int i = 0; i < nLights; ++ i)
		{
			lightVec = multiplyMV (light.transform, glm::vec4 (lightPos.x + ((i%sqrtLights)*stepSize), lightPos.y, lightPos.z + ((i/sqrtLights)*stepSize), 1.0));
			castRay.direction = glm::normalize (lightVec - castRay.origin);

			if (isShadowRayBlocked (castRay, lightVec, geoms, objectCountInfo))
				shadowColour += ka * theRightIntercept.intrMaterial.color;	// If point in shadow, add ambient colour to shadowColour
			else
				shadowColour += shadedColour;								// Otherwise, add the computed shade.
		}
		shadedColour = shadowColour/nLights;

		colors [index] = shadedColour;
	}
}

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

//TODO: Done!
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
  
  // Package render information such as no. of point lights to use to approximate area light and diffuse, specular and ambient coeffs
  renderInfo *renderParamsDeviceCopy = NULL;
  cudaMalloc((void**)&renderParamsDeviceCopy, sizeof(renderInfo));  
  renderInfo renderParams;
  renderParams.ks = 0.3;
  renderParams.ka = 0.1;
  renderParams.kd = 1-renderParams.ks-renderParams.ka;
  renderParams.nLights = 64;
  renderParams.sqrtLights = sqrt ((float)renderParams.nLights);
  renderParams.lightStepSize = 1.0/(renderParams.sqrtLights-1);
  cudaMemcpy (renderParamsDeviceCopy, &renderParams, sizeof(renderInfo), cudaMemcpyHostToDevice);

  //package geometry and materials and sent to GPU
  staticGeom* geomList = new staticGeom[numberOfGeoms];
  sceneInfo		primCounts;
  
  int count = 1;	int lightIndex = 0;
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
		lightIndex = i;
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

  if (!lightSet)
  {
	  geomList [0] = geomList [count-1];
	  count --;
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

  //if (!lightSet)
  //{
		//staticGeom newStaticGeom;
		//newStaticGeom.type = geoms[0].type;
		//newStaticGeom.materialid = geoms[0].materialid;
		//newStaticGeom.translation = geoms[0].translations[frame];
		//newStaticGeom.rotation = geoms[0].rotations[frame];
		//newStaticGeom.scale = geoms[0].scales[frame];
		//newStaticGeom.transform = geoms[0].transforms[frame];
		//newStaticGeom.inverseTransform = geoms[0].inverseTransforms[frame];
		//geomList[0] = newStaticGeom;
  //}

  primCounts.nMeshes = 0;

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

  renderInfo	RenderParams;
  RenderParams.kd;
  RenderParams.ka;
  RenderParams.ks;
  RenderParams.nLights = 64;
  RenderParams.sqrtLights = sqrt ((float)RenderParams.nLights);
  RenderParams.lightStepSize = 1.0/(RenderParams.sqrtLights-1);
  RenderParams.lightPos = glm::vec3 (-0.5, -0.6, -0.5);
  RenderParams.lightCol = materials [geoms [lightIndex].materialid].color;

  //package camera
  cameraData cam;
  cam.resolution = renderCam->resolution;
  cam.position = renderCam->positions[frame];
  cam.view = renderCam->views[frame];
  cam.up = renderCam->ups[frame];
  cam.fov = renderCam->fov;

  glm::vec3 lightPos = glm::vec3 (-0.5, -0.6, -0.5);
  for (int i = 0; i < RenderParams.nLights; i ++)
  {
	  lightPos = multiplyMV (geomList [0].transform, glm::vec4 (RenderParams.lightPos.x + ((i%RenderParams.sqrtLights)*RenderParams.lightStepSize), 
				RenderParams.lightPos.y, RenderParams.lightPos.z + ((i/RenderParams.sqrtLights)*RenderParams.lightStepSize), 1.0));
	  
	  // kernel launches
	  raytraceRay<<<fullBlocksPerGrid, threadsPerBlock>>>(renderCam->resolution, (float)iterations, cam, traceDepth, cudaimage, cudageoms, primCounts, materialColours, ProjectionParams, lightPos);
	  cudaThreadSynchronize();
	  std::cout << "\rRendering.. " <<  ceil ((float)i/(RenderParams.nLights-1) * 100) << " percent complete.";
  }
  sendImageToPBO<<<fullBlocksPerGrid, threadsPerBlock>>>(PBOpos, renderCam->resolution, cudaimage, RenderParams.nLights);
  std::cout << "\n";
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
