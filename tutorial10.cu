
/*
 * Copyright (c) 2008 - 2009 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and proprietary
 * rights in and to this software, related documentation and any modifications thereto.
 * Any use, reproduction, disclosure or distribution of this software and related
 * documentation without an express license agreement from NVIDIA Corporation is strictly
 * prohibited.
 *
 * TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED *AS IS*
 * AND NVIDIA AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS OR IMPLIED,
 * INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
 * PARTICULAR PURPOSE.  IN NO EVENT SHALL NVIDIA OR ITS SUPPLIERS BE LIABLE FOR ANY
 * SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES WHATSOEVER (INCLUDING, WITHOUT
 * LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF
 * BUSINESS INFORMATION, OR ANY OTHER PECUNIARY LOSS) ARISING OUT OF THE USE OF OR
 * INABILITY TO USE THIS SOFTWARE, EVEN IF NVIDIA HAS BEEN ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGES
 */

#include "tutorial.h"
#include <optixu/optixu_aabb.h>

rtDeclareVariable(float3, geometric_normal, attribute geometric_normal, ); 
rtDeclareVariable(float3, shading_normal,   attribute shading_normal, );
rtDeclareVariable(float3, worldHitPoint, attribute worldHitPoint, ); 
rtDeclareVariable(float3, passedVelocity, attribute passedVelocity, );

rtDeclareVariable(PerRayData_radiance, prd_radiance, rtPayload, );
rtDeclareVariable(PerRayData_shadow,   prd_shadow,   rtPayload, );

rtDeclareVariable(optix::Ray, ray,          rtCurrentRay, );
rtDeclareVariable(float,      t_hit,        rtIntersectionDistance, );
rtDeclareVariable(uint2,      launch_index, rtLaunchIndex, );

rtDeclareVariable(unsigned int, radiance_ray_type, , );
rtDeclareVariable(unsigned int, shadow_ray_type , , );
rtDeclareVariable(float,        scene_epsilon, , );
rtDeclareVariable(rtObject,     top_object, , );


//Color shift variables, used to make guassians for XYZ curves
#define xla 0.39952807612909519f
#define xlb 444.63156780935032f
#define xlc 20.095464678736523f

#define xha 1.1305579611401821f
#define xhb 593.23109262398259f
#define xhc 34.446036241271742f

#define ya 1.0098874822455657f
#define yb 556.03724875218927f
#define yc 46.184868454550838f

#define za 2.0648400466720593f
#define zb 448.45126344558236f
#define zc 22.357297606503543f

//Used to determine where to center UV/IR curves
#define IR_RANGE 400.0f
#define IR_START 700.0f
#define UV_RANGE 380.0f
#define UV_START 0.0f
 //Color functions, there's no check for division by 0 which may cause issues on
//some graphics cards.
static __inline__ __host__ __device__ float3 RGBToXYZC(  float r,  float g,  float b)
{
        float3 xyz;
        xyz.x = 0.13514*r + 0.120432*g + 0.057128*b;
        xyz.y = 0.0668999*r + 0.232706*g + 0.0293946*b;
        xyz.z = 0.0*r + 0.0000218959*g + 0.358278*b;
        return xyz;
}
static __inline__ __host__ __device__ float3 XYZToRGBC(  float x,  float y,  float z)
{
        float3 rgb;
        rgb.x = (9.94845*x) - (5.1485*y) - 1.16389*z;
        rgb.y = -2.86007*x + 5.77745*y - 0.0179627*z;
        rgb.z = 0.000174791*x - 0.000353084*y + 2.79113*z;

        return rgb;
}
static __inline__ __host__ __device__ float3 weightFromXYZCurves(float3 xyz)
{
        float3 returnVal;
        returnVal.x = (0.0735806 * xyz.x) - (0.0380793 * xyz.y) - (0.00860837 * xyz.z);
        returnVal.y = (-0.0665378 * xyz.x) +  (0.134408 * xyz.y) - (0.000417865 * xyz.z);
        returnVal.z = (0.00000299624 * xyz.x) - (0.00000605249 * xyz.y) + (0.0484424 * xyz.z);
        return returnVal;
}
        
static __inline__ __host__ __device__ float getXFromCurve(float3 param,  float shift)
{
            float top1 = param.x * xla * exp( (float)(-(pow((param.y*shift) - xlb,2)
                /(2.0f*(pow(param.z*shift,2)+pow(xlc,2))))))*sqrt( (float)(float(2)*(float)3.14159265358979323));
            float bottom1 = sqrt((float)(1.0f/pow(param.z*shift,2))+(1.0f/pow(xlc,2))); 

            float top2 = param.x * xha * exp( float(-(pow((param.y*shift) - xhb,2)
                /(2.0f*(pow(param.z*shift,2)+pow(xhc,2))))))*sqrt( (float)(float(2)*(float)3.14159265358979323));
            float bottom2 = sqrt((float)(1.0f/pow(param.z*shift,2))+(1.0f/pow(xhc,2)));

        return (top1/bottom1) + (top2/bottom2);
}
static __inline__ __host__ __device__ float getYFromCurve(float3 param,  float shift)
{
            float top = param.x * ya * exp( float(-(pow((param.y*shift) - yb,2)
                /(2.0f*(pow(param.z*shift,2)+pow(yc,2))))))*sqrt( float(float(2)*(float)3.14159265358979323));
            float bottom = sqrt((float)(1.0f/pow(param.z*shift,2))+(1.0f/pow(yc,2))); 

        return top/bottom;
}

static __inline__ __host__ __device__ float getZFromCurve(float3 param,  float shift)
{
            float top = param.x * za * exp( float(-(pow((param.y*shift) - zb,2)
                /(2.0f*(pow(param.z*shift,2)+pow(zc,2))))))*sqrt( float(float(2)*(float)3.14159265358979323));
            float bottom = sqrt((float)(1.0f/pow(param.z*shift,2))+(1.0f/pow(zc,2)));

        return top/bottom;
}
        
static __inline__ __host__ __device__ float3 constrainRGB( float r,  float g,  float b)
{
        float w;
            
        w = (0 < r) ? 0 : r;
        w = (w < g) ? w : g;
        w = (w < b) ? w : b;
        w = -w;
            
        if (w > 0) {
                r += w;  g += w; b += w;
        }
        w = r;
        w = ( w < g) ? g : w;
        w = ( w < b) ? b : w;

        if ( w > 1 )
        {
                r /= w;
                g /= w;
                b /= w;
        }        
        float3 rgb;
        rgb.x = r;
        rgb.y = g;
        rgb.z = b;
        return rgb;

};   

//
// Pinhole camera implementation
//
rtDeclareVariable(float3,        eye, , );
rtDeclareVariable(float3,        U, , );
rtDeclareVariable(float3,        V, , );
rtDeclareVariable(float3,        W, , );
rtDeclareVariable(float3,        bad_color, , );
rtBuffer<uchar4, 2>              output_buffer;
rtDeclareVariable(float3, playerVel, , );

RT_PROGRAM void pinhole_camera()
{
  size_t2 screen = output_buffer.size();

  float2 d = make_float2(launch_index) / make_float2(screen) * 2.f - 1.f;
  float3 ray_origin = eye;
  float3 ray_direction = normalize(d.x*U + d.y*V + W);

  float j = ray_direction.x;
  float k = ray_direction.y;
  float l = ray_direction.z;

  float x = playerVel.x;
  float y = playerVel.y;
  float z = playerVel.z;

  float q = (sqrt(pow(((-2.0f*j*x) - (2.0f*k*y) - (2.0f*l*z)),2) - (4.0f * ( pow(j,2) + pow(k,2) + pow(l,2))*(pow(x,2) + pow(y,2) + pow(z,2) - 1.0f))) + (2.0f*j*x) + (2.0f*k*y) + (2.0f*l*z))/
			(2.0*(pow(j,2) + pow(k,2) + pow(l,2)));
  

  float resx = (j*q) - x;
  float resy = (k*q) - y;
  float resz = (l*q) - z;

  float3 new_ray_direction = make_float3(resx,resy,resz);


  optix::Ray ray(ray_origin, new_ray_direction, radiance_ray_type, scene_epsilon);

  PerRayData_radiance prd;
  prd.importance = 1.f;
  prd.depth = 0;

  rtTrace(top_object, ray, prd);



    float speed = length(playerVel);
	float vuDot = dot(playerVel, prd.viw);
	float3 uparra;
	if ( speed != 0 )
	{
		uparra = (vuDot/(speed*speed)) * playerVel;
	}
	else
	{
		uparra = make_float3(0.0f);
	}
	float3 uperp = prd.viw - uparra;
	float3 vr = (playerVel - uparra - (sqrt(1.0f-(speed*speed)))*uperp)/(1.0f+vuDot);
			
	float costheta = dot(ray_direction, vr) / (length(ray_direction)*length(vr));
	float vc = length(vr);

		// ( 1 - (v/c)cos(theta) ) / sqrt ( 1 - (v/c)^2 )
	float shift = ( 1.0f - vc*costheta) / sqrt ( 1.0f - (vc*vc));
	//shift = 1.0f - shift;
	float3 rgb = prd.result;
                        
    //Color shift due to doppler, go from RGB -> XYZ, shift, then back to RGB.
    float3 xyz = RGBToXYZC(float(rgb.x),float(rgb.y),float(rgb.z));
    float3 weights = weightFromXYZCurves(xyz);
    float3 rParam,gParam,bParam;
    rParam.x = weights.x; rParam.y = ( float) 615; rParam.z = ( float)8;
    gParam.x = weights.y; gParam.y = ( float) 550; gParam.z = ( float)4;
    bParam.x = weights.z; bParam.y = ( float) 463; bParam.z = ( float)5; 
                
    float xf = pow((1.0/shift),3)*getXFromCurve(rParam, shift) + getXFromCurve(gParam,shift) + getXFromCurve(bParam,shift);
    float yf = pow((1.0/shift),3)*getYFromCurve(rParam, shift) + getYFromCurve(gParam,shift) + getYFromCurve(bParam,shift);
    float zf = pow((1.0/shift),3)*getZFromCurve(rParam, shift) + getZFromCurve(gParam,shift) + getZFromCurve(bParam,shift);
                
    float3 rgbFinal = XYZToRGBC(xf,yf,zf);
    rgbFinal = constrainRGB(rgbFinal.x,rgbFinal.y, rgbFinal.z); //might not be needed

    if ( shift >= -0.1) { 
		output_buffer[launch_index] = make_color(rgbFinal);
  } else {
	output_buffer[launch_index] = make_color( make_float3(0.0f,1.0f,0.0f) );
  }
}


//
// Environment map background
//
rtTextureSampler<float4, 2> envmap;
RT_PROGRAM void envmap_miss()
{
  float theta = atan2f( ray.direction.x, ray.direction.z );
  float phi   = M_PIf * 0.5f -  acosf( ray.direction.y );
  float u     = (theta + M_PIf) * (0.5f * M_1_PIf);
  float v     = 0.5f * ( 1.0f + sin(phi) );	
  prd_radiance.result = make_float3( tex2D(envmap, u, v) );
}
  

//
// Terminates and fully attenuates ray after any hit
//
RT_PROGRAM void any_hit_shadow()
{
  // this material is opaque, so it fully attenuates all shadow rays
  prd_shadow.attenuation = make_float3(0);

  rtTerminateRay();
}


//
// Terminates and fully attenuates ray after any hit
//
RT_PROGRAM void any_hit_shadow_rel()
{

	if ( prd_shadow.attenuation.x + scene_epsilon < t_hit ) {
	  // this material is opaque, so it fully attenuates all shadow rays
	  prd_shadow.attenuation = make_float3(0);

	  rtTerminateRay();
	}
	else {
		rtIgnoreIntersection();
	}
}
  

//
// Procedural rusted metal surface shader
//

/*
 * Translated to CUDA C from Larry Gritz's LGRustyMetal.sl shader found at:
 * http://renderman.org/RMR/Shaders/LGShaders/LGRustyMetal.sl
 *
 * Used with permission from tal AT renderman DOT org.
 */

rtDeclareVariable(float3,   ambient_light_color, , );
rtBuffer<BasicLight>        lights;   
rtDeclareVariable(rtObject, top_shadower, , );
rtDeclareVariable(float,    importance_cutoff, , );      
rtDeclareVariable(int,      max_depth, , );
rtDeclareVariable(float3,   reflectivity_n, , );

rtDeclareVariable(float, metalKa, , ) = 1;
rtDeclareVariable(float, metalKs, , ) = 1;
rtDeclareVariable(float, metalroughness, , ) = .1;
rtDeclareVariable(float, rustKa, , ) = 1;
rtDeclareVariable(float, rustKd, , ) = 1;
rtDeclareVariable(float3, rustcolor, , ) = {.437, .084, 0};
rtDeclareVariable(float3, metalcolor, , ) = {.7, .7, .7};
rtDeclareVariable(float, txtscale, , ) = .02;
rtDeclareVariable(float, rusty, , ) = 0.2;
rtDeclareVariable(float, rustbump, , ) = 0.85;
rtDeclareVariable(float3, Kd, , );
#define MAXOCTAVES 0

rtTextureSampler<float, 3> noise_texture;
static __device__ __inline__ float snoise(float3 p)
{
  return tex3D(noise_texture, p.x, p.y, p.z) * 2 -1;
}

RT_PROGRAM void box_closest_hit_radiance()
{
  float3 world_geo_normal   = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, geometric_normal ) );
  float3 world_shade_normal = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, shading_normal ) );
  float3 ffnormal     = faceforward( world_shade_normal, -ray.direction, world_geo_normal );
  float3 hit_point = ray.origin + t_hit * ray.direction;

  /* Sum several octaves of abs(snoise), i.e. turbulence.  Limit the
   * number of octaves by the estimated change in PP between adjacent
   * shading samples.
   */
  float3 PP = txtscale * hit_point;
  float a = 1;
  float sum = 0;
  for(int i = 0; i < MAXOCTAVES; i++ ){
    sum += a * fabs(snoise(PP));
    PP *= 2.0f;
    a *= 0.5f;
  }

  /* Scale the rust appropriately, modulate it by another noise 
   * computation, then sharpen it by squaring its value.
   */
  float rustiness = step (1-rusty, clamp (sum,0.0f,1.0f));
  rustiness *= clamp (abs(snoise(PP)), 0.0f, .08f) / 0.08f;
  rustiness *= rustiness;

  /* If we have any rust, calculate the color of the rust, taking into
   * account the perturbed normal and shading like matte.
   */
  float3 Nrust = ffnormal;
  if (rustiness > 0) {	
    /* If it's rusty, also add a high frequency bumpiness to the normal */
    Nrust = normalize(ffnormal + rustbump * snoise(PP));
    Nrust = faceforward (Nrust, -ray.direction, world_geo_normal);
  }

  float3 color = mix(metalcolor * metalKa, rustcolor * rustKa, rustiness) * ambient_light_color;
  float3 new_origin = worldHitPoint; //Cast from the point where it would have been in the world reference frame at that point
  //float3 new_origin = make_float3(0.0);
  for(int i = 0; i < lights.size(); ++i) {
    BasicLight light = lights[i];
    float3 L = normalize(light.pos - worldHitPoint);
    float nmDl = dot( ffnormal, L);
    float nrDl = dot( Nrust, L);

    if( nmDl > 0.0f || nrDl > 0.0f ){
      // cast shadow ray
	  float3 L2 = normalize(light.pos - worldHitPoint); //The direction from the point in the world frame to the light
      PerRayData_shadow shadow_prd;
      shadow_prd.attenuation = make_float3(t_hit, 0.0, 0.0);
      
	  //float3 newDir = L2 - playerVel;
	  float3 newDir = L2;
	  float3 newOriginShadow = new_origin - (t_hit*newDir); //Adjust the start point to account for where the light started
	  //float3 newOriginShadow = new_origin;
	  float Ldist = length(light.pos - newOriginShadow);  

      optix::Ray shadow_ray( newOriginShadow, newDir, shadow_ray_type, scene_epsilon, Ldist );
      rtTrace(top_shadower, shadow_ray, shadow_prd);
      float3 light_attenuation = shadow_prd.attenuation;

      if( fmaxf(light_attenuation) > 0.0f ){
        float nDl = dot( ffnormal, L); 
 
		if( nDl > 0 ) {
		 color += Kd * nDl * light.color;
		}
      }

    }
  }

  float3 r = schlick(-dot(ffnormal, ray.direction), reflectivity_n * (1-rustiness));
  float importance = prd_radiance.importance * optix::luminance( r );

  // reflection ray
  /*
  if( importance > importance_cutoff && prd_radiance.depth < max_depth) {
    PerRayData_radiance refl_prd;
    refl_prd.importance = importance;
    refl_prd.depth = prd_radiance.depth+1;
    float3 R = reflect( ray.direction, ffnormal );
    optix::Ray refl_ray( hit_point, R, radiance_ray_type, scene_epsilon );
    rtTrace(top_object, refl_ray, refl_prd);
    color += r * refl_prd.result;
  }
  */
  
  /*
            float speed = length(playerVel);
			float vuDot = dot(playerVel, passedVelocity);
			float3 uparra;
			if ( speed != 0 )
			{
				uparra = (vuDot/(speed*speed)) * playerVel;
			}
			else
			{
				uparra = make_float3(0.0f);
			}
			float3 uperp = passedVelocity - uparra;
			float3 vr = (playerVel - uparra - (sqrt(1-(speed*speed)))*uperp)/(1+vuDot);
			
			float costheta = dot(ray.direction, vr) / (length(ray.direction)*length(vr));
			float vc = 0.8;

			  // ( 1 - (v/c)cos(theta) ) / sqrt ( 1 - (v/c)^2 )
			float shift = ( 1.0f - vc*costheta) / sqrt ( 1 - (vc*vc));
			shift = clamp(shift, 0.1f, 0.9f);
			*/

		//color
      prd_radiance.viw = passedVelocity;
	  prd_radiance.result = color;//(make_float3(1.0,0.0,0.0));
			
  
}


//
// Returns solid color for miss rays
//

rtDeclareVariable(float3, bg_color, , );
RT_PROGRAM void miss()
{
  prd_radiance.result = bg_color;
  prd_radiance.viw = make_float3(0.0,0.0,0.0);
}
  


//
// Phong surface shading with shadows and schlick-approximated fresnel reflections.
// Uses procedural texture to determine diffuse response.
//
rtDeclareVariable(float,  phong_exp, , );
rtDeclareVariable(float3, tile_v0, , );
rtDeclareVariable(float3, tile_v1, , );   
rtDeclareVariable(float3, crack_color, , );
rtDeclareVariable(float,  crack_width, , );
rtDeclareVariable(float3, Ka, , );
rtDeclareVariable(float3, Ks, , );

RT_PROGRAM void floor_closest_hit_radiance()
{
  float3 world_geo_normal   = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, geometric_normal ) );
  float3 world_shade_normal = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, shading_normal ) );
  float3 ffnormal     = faceforward( world_shade_normal, -ray.direction, world_geo_normal );
  float3 color = Ka * ambient_light_color;

  float3 hit_point = ray.origin + t_hit * ray.direction;

  float v0 = dot(tile_v0, hit_point);
  float v1 = dot(tile_v1, hit_point);
  v0 = v0 - floor(v0);
  v1 = v1 - floor(v1);

  float3 local_Kd;
  if( v0 > crack_width && v1 > crack_width ){
    local_Kd = Kd;
  } else {
    local_Kd = crack_color;
  }

  for(int i = 0; i < lights.size(); ++i) {
    BasicLight light = lights[i];
    float3 L = normalize(light.pos - worldHitPoint);
    float nDl = dot( ffnormal, L);

    if( nDl > 0.0f ){
      // cast shadow ray
      PerRayData_shadow shadow_prd;
      shadow_prd.attenuation = make_float3(1.0f);
      float Ldist = length(light.pos - hit_point);
      optix::Ray shadow_ray( hit_point, L, shadow_ray_type, scene_epsilon, Ldist );
      rtTrace(top_shadower, shadow_ray, shadow_prd);
      float3 light_attenuation = shadow_prd.attenuation;

      if( fmaxf(light_attenuation) > 0.0f ){
        float3 Lc = light.color * light_attenuation;
        color += local_Kd * nDl * Lc;

        float3 H = normalize(L - ray.direction);
        float nDh = dot( ffnormal, H );
        if(nDh > 0)
          color += Ks * Lc * pow(nDh, phong_exp);
      }

    }
  }

  float3 r = schlick(-dot(ffnormal, ray.direction), reflectivity_n);
  float importance = prd_radiance.importance * optix::luminance( r );

  // reflection ray
  if( importance > importance_cutoff && prd_radiance.depth < max_depth) {
    PerRayData_radiance refl_prd;
    refl_prd.importance = importance;
    refl_prd.depth = prd_radiance.depth+1;
    float3 R = reflect( ray.direction, ffnormal );
    optix::Ray refl_ray( hit_point, R, radiance_ray_type, scene_epsilon );
    rtTrace(top_object, refl_ray, refl_prd);
    color += r * refl_prd.result;
  }

  prd_radiance.result = color;
}
  

//
// Bounding box program for programmable convex hull primitive
//
rtDeclareVariable(float3, chull_bbmin, , );
rtDeclareVariable(float3, chull_bbmax, , );
RT_PROGRAM void chull_bounds (int primIdx, float result[6])
{
  optix::Aabb* aabb = (optix::Aabb*)result;
  aabb->m_min = chull_bbmin;
  aabb->m_max = chull_bbmax;
}


//
// Intersection program for programmable convex hull primitive
//
rtBuffer<float4> planes;
RT_PROGRAM void chull_intersect(int primIdx)
{
  int n = planes.size();
  float t0 = -FLT_MAX;
  float t1 = FLT_MAX;
  float3 t0_normal = make_float3(0);
  float3 t1_normal = make_float3(0);
  for(int i = 0; i < n && t0 < t1; ++i ) {
    float4 plane = planes[i];
    float3 n = make_float3(plane);
    float  d = plane.w;

    float denom = dot(n, ray.direction);
    float t = -(d + dot(n, ray.origin))/denom;
    if( denom < 0){
      // enter
      if(t > t0){
        t0 = t;
        t0_normal = n;
      }
    } else {
      //exit
      if(t < t1){
        t1 = t;
        t1_normal = n;
      }
    }
  }

  if(t0 > t1)
    return;

  if(rtPotentialIntersection( t0 )){
    shading_normal = geometric_normal = t0_normal;
    rtReportIntersection(0);
  } else if(rtPotentialIntersection( t1 )){
    shading_normal = geometric_normal = t1_normal;
    rtReportIntersection(0);
  }
}


//
// (NEW)
// Attenuates shadow rays for shadowing transparent objects
//

rtDeclareVariable(float3, shadow_attenuation, , );
rtDeclareVariable(float, shadow_timeStart, , );

RT_PROGRAM void glass_any_hit_shadow()
{
  float3 world_normal = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, shading_normal ) );
  float nDi = fabs(dot(world_normal, ray.direction));

  prd_shadow.attenuation *= 1-fresnel_schlick(nDi, 5, 1-shadow_attenuation, make_float3(1));

  rtIgnoreIntersection();
}


//
// Dielectric surface shader
//
rtDeclareVariable(float3,       cutoff_color, , );
rtDeclareVariable(float,        fresnel_exponent, , );
rtDeclareVariable(float,        fresnel_minimum, , );
rtDeclareVariable(float,        fresnel_maximum, , );
rtDeclareVariable(float,        refraction_index, , );
rtDeclareVariable(int,          refraction_maxdepth, , );
rtDeclareVariable(int,          reflection_maxdepth, , );
rtDeclareVariable(float3,       refraction_color, , );
rtDeclareVariable(float3,       reflection_color, , );
rtDeclareVariable(float3,       extinction_constant, , );
RT_PROGRAM void glass_closest_hit_radiance()
{
  // intersection vectors
  const float3 h = ray.origin + t_hit * ray.direction;            // hitpoint
  const float3 n = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal)); // normal
  const float3 i = ray.direction;                                            // incident direction

  float reflection = 1.0f;
  float3 result = make_float3(0.0f);

  float3 beer_attenuation;
  if(dot(n, ray.direction) > 0){
    // Beer's law attenuation
    beer_attenuation = exp(extinction_constant * t_hit);
  } else {
    beer_attenuation = make_float3(1);
  }

  // refraction
  if (prd_radiance.depth < min(refraction_maxdepth, max_depth))
  {
    float3 t;                                                            // transmission direction
    if ( refract(t, i, n, refraction_index) )
    {

      // check for external or internal reflection
      float cos_theta = dot(i, n);
      if (cos_theta < 0.0f)
        cos_theta = -cos_theta;
      else
        cos_theta = dot(t, n);

      reflection = fresnel_schlick(cos_theta, fresnel_exponent, fresnel_minimum, fresnel_maximum);

      float importance = prd_radiance.importance * (1.0f-reflection) * optix::luminance( refraction_color * beer_attenuation );
      if ( importance > importance_cutoff ) {
        optix::Ray ray( h, t, radiance_ray_type, scene_epsilon );
        PerRayData_radiance refr_prd;
        refr_prd.depth = prd_radiance.depth+1;
        refr_prd.importance = importance;

        rtTrace( top_object, ray, refr_prd );
        result += (1.0f - reflection) * refraction_color * refr_prd.result;
      } else {
        result += (1.0f - reflection) * refraction_color * cutoff_color;
      }
    }
    // else TIR
  }

  // reflection
  if (prd_radiance.depth < min(reflection_maxdepth, max_depth))
  {
    float3 r = reflect(i, n);

    float importance = prd_radiance.importance * reflection * optix::luminance( reflection_color * beer_attenuation );
    if ( importance > importance_cutoff ) {
      optix::Ray ray( h, r, radiance_ray_type, scene_epsilon );
      PerRayData_radiance refl_prd;
      refl_prd.depth = prd_radiance.depth+1;
      refl_prd.importance = importance;

      rtTrace( top_object, ray, refl_prd );
      result += reflection * reflection_color * refl_prd.result;
    } else {
      result += reflection * reflection_color * cutoff_color;
    }
  }

  result = result * beer_attenuation;

  //prd_radiance.result = result;
  prd_radiance.result = make_float3(1,0,0);
}


//
// Set pixel to solid color upon failure
//
RT_PROGRAM void exception()
{
  output_buffer[launch_index] = make_color( bad_color );
}
