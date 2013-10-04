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
materials and treats all lights as area lights. The no. of samples for the area light can be specified 
by the programmer and sampling is done using a jittered grid. It supports Supersampled Antialiasing, the 
maximum number of samples being bound only by the number of samples for the area light. It also has support 
for motion blur. Due to the nature of implementation, both of these features are free of performance overheads.

On the way
----------
* Fresnel refraction
* Texture and Normal Mapping
* Depth of field

-------------------------------------------------------------------------------
SCREENSHOT
-------------------------------------------------------------------------------
<img src="https://raw.github.com/rohith10/Project1-RayTracer/master/PROJ1_WIN/565Raytracer/test.0.png" height="350" width="350"/>
<br />The circular pattern seen on the back wall is due to floating point errors exacerbated by the old 
driver on the lab machine I used to make this build.

-------------------------------------------------------------------------------
VIDEO
-------------------------------------------------------------------------------
Click on the image below to watch the video on YouTube: 

[![YouTube Link](http://img.youtube.com/vi/dKMg6kBt8Ek/0.jpg)](http://www.youtube.com/watch?v=dKMg6kBt8Ek)

-------------------------------------------------------------------------------
PERFORMANCE ANALYSIS
-------------------------------------------------------------------------------
A performance analysis was performed for this project and can be found in the root folder with the 
name Project1-PerfAnalysis. It is a Word Document. 
