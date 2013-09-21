-------------------------------------------------------------------------------
CIS565: Project 1: CUDA Raytracer
-------------------------------------------------------------------------------
I implemented a raytracer in CUDA for this project. The base code provided was based 
on TAKUA Render, a massively parallel path tracer written by Yining Karl Li. The base code 
takes care of basic stuff such as creating a GLFW window, reading config files and setting up 
the scene and so on, so that I could focus on getting the raytracer working.

This is my first hands-on experience with CUDA. Needless to say, I made some terrible design 
decisions that hampered my progress toward the end. If it hadn't been for these, I could have 
spruced it up more. Also, the base code implements a left-handed co-ordinate system, which 
threw me off the rails since the rendered image was displayed using OpenGL, which 
has a right-handed system (and which is what I had been expecting).


-------------------------------------------------------------------------------
FEATURES:
-------------------------------------------------------------------------------
Current
-------
The current version of the raytracer supports sphere and cube primitives. It can handle reflective 
materials and treats all lights as area lights, sampled regularly at 8x8 grid points. Because of this, 
there is a very visible banding effect on the soft shadows. It does not perform antialiasing.

On the way
----------
* Antialiasing (Supersampling and Multisampling)
* Fresnel refraction
* Texture and Normal Mapping
* Depth of field
* Motion blur

-------------------------------------------------------------------------------
SCREENSHOT
-------------------------------------------------------------------------------
<img src="https://raw.github.com/rohith10/Project1-RayTracer/master/renders/Final2-800px.png" height="350" width="350"/>

-------------------------------------------------------------------------------
VIDEO
-------------------------------------------------------------------------------
Click on the image below to watch the video on YouTube: 

[![YouTube Link](http://img.youtube.com/vi/dKMg6kBt8Ek/0.jpg)](http://www.youtube.com/watch?v=dKMg6kBt8Ek)
