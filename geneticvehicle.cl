#pragma OPENCL EXTENSION cl_amd_printf : enable

#define NUMBER_OF_VERTICES_PER_CAR %(vertices_per_car)d
#define NUMBER_OF_CARS %(number_of_cars)d
#define NUMBER_OF_WHEELS_PER_CAR %(number_of_wheels)d
#define NUMBER_OF_CONTACT_POINTS %(number_of_contact_points)d
#define DENSITY %(density)f
#define CROSSOVER_POINTS %(crossover_points)d
#define POINT_MUTATIONS %(point_mutations)d


#define ISLAND_ACCELERATION %(island_acceleration)d
#define ISLAND_START %(island_start)s
#define ISLAND_STEP %(island_step)s
#define ISLAND_END %(island_end)s
#define ISLAND_RANGE %(island_range)s


#define GRAVITY 9.81f
#define EPSILON 0.001f
#define COLLISION_TOLERANCE 0.025f
#define FRICTIONCOEFFICIENT  0.35f
#define LINEARDRAGCOEFFICIENT 0.25f

float cross2d(float2 a, float2 b) {
    return a.x*b.y-a.y*b.x;
}

float dot2d(float2 a, float2 b) {
    return a.x*b.x+a.y*b.y;
}

float2 perp(float2 a) {
    return (float2)(-a.y, a.x);
}

float2 rotate2d(float orientation, float2 input) {
    float cs = cos(radians(orientation));
    float sn = sin(radians(orientation));
    return (float2)(input.x*cs-input.y*sn, input.x*sn+input.y*cs);
}



__kernel void generate_angles(__global float *angles, __global float *ordered_angles)
{
    // this kernel repeats the angles stored in ordered_angles until the angles array is filled
    int gid = get_global_id(0);
    angles[gid] = ordered_angles[gid%%NUMBER_OF_VERTICES_PER_CAR];

}

__kernel void generate_colors(__global float4 *vertex_colors, __global float4 *vehicle_colors)
{
    int gid = get_global_id(0);
    int vertex_start_index =  gid*NUMBER_OF_VERTICES_PER_CAR;

    for(int i=0; i<NUMBER_OF_VERTICES_PER_CAR; i++) {
        vertex_colors[vertex_start_index+i] = vehicle_colors[gid];
    }

}


__kernel void generate_vertices(__global float *magnitudes, __global float *angles, __global float2 *vertices)
{

    int gid = get_global_id(0);
    vertices[gid] = magnitudes[gid]*(float2)(cos(angles[gid]), sin(angles[gid]));

}

__kernel void generate_wheel_properties(__global float *masses, __global float *inertias, __global float *radii) {
    // adaption of http://code.google.com/p/box2d/source/browse/trunk/Box2D/Box2D/Collision/Shapes/b2CircleShape.cpp
    int gid = get_global_id(0);
    masses[gid] = DENSITY*M_PI*radii[gid]*radii[gid];
    inertias[gid] = masses[gid]*(0.5f*radii[gid]*radii[gid]);
}


__kernel void generate_bounding_volumes(__global float2 *vertices, __global int *wheel_vertex_positions , __global float *wheel_radii, __global float *bounding_volumes) {
    int gid = get_global_id(0);
    int vertex_start_index =  gid*NUMBER_OF_VERTICES_PER_CAR;
    int wheel_start_index = gid*NUMBER_OF_WHEELS_PER_CAR;

    float radius = 0;

    for(int i=0; i<NUMBER_OF_VERTICES_PER_CAR; i++) {
        radius = max(radius, length(vertices[vertex_start_index+i]));
    }

    for(int k=0; k<NUMBER_OF_WHEELS_PER_CAR; k++) {
        int wheel_vertex_position = wheel_vertex_positions[wheel_start_index+k];
        float wheel_radius = wheel_radii[wheel_start_index+k];
        float2 local_wheel_position = vertices[vertex_start_index+wheel_vertex_position];
        radius = max(radius, length(local_wheel_position)+wheel_radius);
    }

    bounding_volumes[gid] = radius;

}

 __kernel void generate_vehicle_properties(__global float *masses, __global float2 *center_masses, __global float *inertias, __global float2 *vertices)
{

    if(get_global_id(0) >= NUMBER_OF_CARS)
        return;

    // adaption of http://code.google.com/p/box2d/source/browse/trunk/Box2D/Box2D/Collision/Shapes/b2PolygonShape.cpp
    int gid = get_global_id(0);
    int vertex_start_index =  gid*NUMBER_OF_VERTICES_PER_CAR;

    float area = 0;
    float2 center = 0;
    float inertia = 0;

    for(int i=0; i<NUMBER_OF_VERTICES_PER_CAR; i++) {
        int begin_index = i+vertex_start_index;
        int end_index = (i+1)%%NUMBER_OF_VERTICES_PER_CAR+vertex_start_index;

        float2 e1 = vertices[begin_index];
        float2 e2 = vertices[end_index];

        float D = cross2d(e1, e2);
        float triangle_area = 0.5f*D;
        area += triangle_area;
        center += triangle_area*(e1+e2)/3.0f;

        float intx2 = e1.x*e1.x + e2.x*e1.x + e2.x*e2.x;
        float inty2 = e1.y*e1.y + e2.y*e1.y + e2.y*e2.y;

        inertia += (0.25f/3.0f * D) * (intx2+inty2);
    }

    masses[gid] = DENSITY*area;
    inertias[gid] = DENSITY*inertia;
    center_masses[gid] = center/area;

}

__kernel void assign_score(__global float2 *positions, __global float2 *velocities, __global float *score) {
    int gid = get_global_id(0);
    score[gid] = max(0.0f, positions[gid].x);
}


__kernel void evaluate_score(__global float *score, __global float *old_score, __global float *alive) {
    int gid = get_global_id(0);
    if (length(score[gid]-old_score[gid]) < 0.75f) {
        alive[gid] = 0;
    }
}

__kernel void calculate_loads(__global float2 *forces, __global float *momenta, __global float *velocities, __global float *angular_velocities) {
    if(get_global_id(0) >= NUMBER_OF_CARS)
        return;
    int gid = get_global_id(0);
    forces[gid] = (float2)(0,GRAVITY);
    velocities[gid] *= 0.999;
    angular_velocities[gid] *= 0.992;
}


__kernel void integrate(__global float *alive,
                        __global float2 *positions,
                        __global float *masses,
                        __global float2 *forces,
                        __global float2 *velocities,
                        __global float *angular_velocities,
                        __global float *orientations,
                        __global float *momenta,
                        __global float *inertias,
                        float delta,
                        __global float *wheel_angular_velocities,
                        __global float *wheel_radii,
                        __global float *wheel_momenta,
                        __global float *wheel_masses,
                        __global float *wheel_inertias,
                        __global float *wheel_orientations,
                        __global float2 *vertices,
                        __global int *wheel_vertex_positions)
{

    if(get_global_id(0) >= NUMBER_OF_CARS)
        return;

    int gid = get_global_id(0);
    int wheel_start_index =  gid*NUMBER_OF_WHEELS_PER_CAR;
    int vertex_start_index =  gid*NUMBER_OF_VERTICES_PER_CAR;
    float dt = delta;

    if (alive[gid] >= 1.0f) {

        for(int k=0; k<NUMBER_OF_WHEELS_PER_CAR; k++) {
            int wheel_id = wheel_start_index+k;
            if (wheel_momenta[wheel_start_index+k] != 0) {
                int wheel_vertex_position = wheel_vertex_positions[wheel_id];
                float2 local_wheel_position = vertices[vertex_start_index+wheel_vertex_position];
                float wheel_radius = wheel_radii[wheel_id];
                float w = -500*wheel_momenta[wheel_id]/wheel_inertias[wheel_id];
                wheel_angular_velocities[wheel_id] += w*delta;
                wheel_orientations[wheel_id] += wheel_angular_velocities[wheel_id]*delta;
                forces[gid] += normalize(velocities[gid])*wheel_momenta[wheel_id]/wheel_radius;
                //momenta[gid] += 50*cross2d(local_wheel_position, w);
            }

            wheel_momenta[wheel_id] = 0;
            wheel_angular_velocities[wheel_id] *= 0.95;
        }

        //float2 a = forces[gid]/masses[gid];
        float2 a = forces[gid];
        positions[gid] += velocities[gid]*dt+0.5*a*dt*dt;
        velocities[gid] += 0.5*a*a*dt*dt;

        float w = momenta[gid]/inertias[gid];
        angular_velocities[gid] += w*dt;
        orientations[gid] += angular_velocities[gid]*dt;

    }

    if (positions[gid].y > 50)
        alive[gid] = 0.0f;

    momenta[gid] *= 0.95;
    //momenta[gid] = 0;
}

float2 vector_rejection(float2 point, float2 v1, float2 v2) {
    float2 b = v2-v1;
    float2 a = point-v1;
    return a-(dot2d(a,b)/dot2d(b,b))*b;
}


bool VertexBelowLine(float2 x0, float2 x1, float2 v1,
                     float2 *contact_point, float2 *n, float *penetration_depth) {


    if (x0.x < x1.x) {

        if (x0.x <= v1.x && v1.x <= x1.x) {
            if ((v1.y < x0.y) &&  (v1.y < x1.y)) {
                return false;
            }

            float k = (v1.x-x0.x)/(x1.x-x0.x);
            float2 p = x0+k*(x1-x0);

            if (p.y < v1.y) {
                *penetration_depth = length(vector_rejection(v1, x0, x1));
                //*penetration_depth = v1.y-p.y;
                *n = normalize(perp(x1-x0));
                *contact_point = p;
                return true;
            }
        }
    }
    return false;
 }


 bool VertexBelowLine2(float2 x0, float2 x1, float2 v1,
                      float2 *contact_point, float2 *n, float *penetration_depth) {

      // check if vertex is below line
      float2 line = normalize(x1-x0);
      float2 v = v1-x0;
      float k = dot2d(v,line);
      float t = cross2d(v,line);

      if (k >= 0 && k <= length(x1-x0) && t < 0) {
         *contact_point = x0+k*line;
         *n = perp(line);
         *penetration_depth = length(v1-(k*line+x0));
         return true;
      }

      return false;
 }



bool VertexTriangleIntersection2d(float2 p, float2 a, float2 b, float2 c,
                              float2 *contact_point, float2 *n, float *penetration_depth) {
    // adaption of http://www.blackpawn.com/texts/pointinpoly/default.html
    float2 v0 = c-a;
    float2 v1 = b-a;
    float2 v2 = p-a;

    float dot00 = dot2d(v0, v0);
    float dot01 = dot2d(v0, v1);
    float dot02 = dot2d(v0, v2);
    float dot11 = dot2d(v1, v1);
    float dot12 = dot2d(v1, v2);

    float invDenom = 1/(dot00*dot11-dot01*dot01);
    float u = (dot11*dot02-dot01*dot12)*invDenom;
    float v = (dot00*dot12-dot01*dot02)*invDenom;

    if ((u >= 0) && (v >= 0) && (u+v < 1)) {

        float2 ab = vector_rejection(p, b, a);
        float distance = length(ab);
        *n = -normalize(ab);

        float2 ac = vector_rejection(p, c, a);
        if (length(ac) < distance) {
            distance = length(ac);
            *n = -normalize(ac);
        }


        float2 bc = vector_rejection(p, c, b);
        if (length(bc) < distance) {
            distance = length(bc);
            *n = -normalize(bc);
        }

        *contact_point = p;
        *penetration_depth = distance;

        return true;
    }
    return false;

}



bool detect_collision(float2 x0, float2 x1,
                      float2 v0, float2 v1, float2 v2,
                      float2 *contact_point, float2 *n, float *penetration_depth){

    float tentative_penetration_depth = MAXFLOAT;


    // check if geometry vertex is inside triangle
    if (true) {
        if (VertexTriangleIntersection2d(x0,v0,v1,v2,contact_point,n,&tentative_penetration_depth)) {

                  if (tentative_penetration_depth > *penetration_depth) {
                      *penetration_depth = tentative_penetration_depth;
                  }
        }
    }

    // check if triangle vertices are below ground
    if (true) {

        // this check is probably not necessary
        if (VertexBelowLine(x0,x1,v0,contact_point,n,&tentative_penetration_depth)) {

                  if (tentative_penetration_depth > *penetration_depth) {
                      *penetration_depth = tentative_penetration_depth;
                  }
        }

        if (VertexBelowLine(x0,x1,v1,contact_point,n,&tentative_penetration_depth)) {

                  if (tentative_penetration_depth > *penetration_depth) {
                      *penetration_depth = tentative_penetration_depth;
                  }
        }

        if (VertexBelowLine(x0,x1,v2,contact_point,n,&tentative_penetration_depth)) {

                  if (tentative_penetration_depth > *penetration_depth) {
                      *penetration_depth = tentative_penetration_depth;
                  }
        }

    }

    if (*penetration_depth > 0)
        return true;

    return false;

}

bool CircleVertexIntersection2d(float2 x0, float2 center, float r,
                                 float2 *contact_point, float2 *n, float *penetration_depth) {

    if (length((x0-center)) < r) {
        *contact_point = x0;
        *n = normalize(x0-center);
        *penetration_depth = r-length((x0-center));
        return true;
    }
    return false;
}


bool CircleSegmentIntersection2d(float2 x0, float2 x1, float2 center, float r,
                                 float2 *contact_point, float2 *n, float *penetration_depth) {
    float2 d = x1-x0;
    float2 f = x0-center;

    float a = dot2d(d,d);
    float b = 2*dot2d(f,d);
    float c = dot2d(f,f)-r*r;

    float discriminant = b*b-4*a*c;

    if( discriminant < 0 ) {
          // no intersection
          return false;
    } else {
        // ray didn't totally miss sphere,
        // so there is a solution to
        // the equation.
        discriminant = sqrt(discriminant);
        // either solution may be on or off the ray so need to test both
        float sol1 = (-b + discriminant)/(2*a);
        float sol2 = (-b - discriminant)/(2*a);

        float t0 = min(sol1, sol2);
        float t1 = max(sol1, sol2);

        *n = perp(normalize(d));

        // Line segment intersects at two points, in which case both values of t will be between 0 and 1.
        if (0 <= t1 && t1 <= 1 && 0 <= t0 && t0 <= 1) {
            *contact_point = x0+((t0+t1)/2)*d;
            *penetration_depth = r-length(*contact_point-center);
            return true;
        }

        if (t1 < 0)
            return false;

        // Line segment doesn't intersect and on outside of sphere, in which case both values of t will either be less than 0 or greater than 1.
        if ((0 > t0 && 0 > t1) || (t0 > 1 && t1 > 1))
            return false;

        // Line segment doesn't intersect and is inside sphere, in which case one value of t will be negative and the other greater than 1.
        if (t0 < 0 && t1 > 1) {
            *contact_point = x0;
            *penetration_depth = r;
            return true;
        }

        // Line segment intersects at one point, in which case one value of t will be between 0 and 1 and the other not.

        if (t0 < 0 && 0 <= t1 && t1 <= 1) {
            float2 middle = x0+(t1/2.0f)*d;
            *contact_point = x0+t1*d;
            //*contact_point = middle;
            *penetration_depth = (r-length(vector_rejection(center, x0, x1)));
            return true;
        }

        if (t1 > 1 && 0 <= t0 && t0 <= 1) {
            *contact_point = x0+t0*d;
            *penetration_depth = (r-length(vector_rejection(center, x0, x1)));
            return true;
        }

        // Line segment is tangential to the sphere, in which case both values of t will be the same and between 0 and 1.
        if (length(t0-t1) < 0.005f) {
            return true;
        }

    }

    return false;

}

bool CircleSegmentIntersection2dsimple(float2 x0, float2 x1, float2 center, float r) {
    float2 contact_point = 0;
    float2 n = 0;
    float penetration_depth = 0;
    return CircleSegmentIntersection2d(x0, x1, center, r, &contact_point, &n, &penetration_depth);
}



__kernel void collision(__global float *alive,
                        __global float2 *positions,
                        __global float *masses,
                        __global float2 *velocities,
                        __global float *orientations,
                        __global float2 *geometry,
                         int geo_size,
                         __global float2 *vertices,
                         __global float *inertias,
                         __global float *angular_velocities,
                         __global float2 *global_contact_points,
                         __global float2 *global_contact_normals,
                         __global float2 *centerofmasses,
                         float delta,
                         __global int *wheel_vertex_positions,
                         __global float *wheel_radii,
                         __global float *bounding_volumes,
                         __global float *wheel_momenta,
                         __global float *wheel_masses)
{

        if(get_global_id(0) >= NUMBER_OF_CARS)
            return;

        int gid = get_global_id(0);

        if (alive[gid] == 0)
            return;

        int vertex_start_index =  gid*NUMBER_OF_VERTICES_PER_CAR;
        int wheel_start_index =  gid*NUMBER_OF_WHEELS_PER_CAR;
        int contact_point_id = gid*NUMBER_OF_CONTACT_POINTS;

        float2 v1 = positions[gid];
        float orientation = orientations[gid];
        float bounding_volume = bounding_volumes[gid];

        float2 contact_points[NUMBER_OF_CONTACT_POINTS];
        float penetration_depths[NUMBER_OF_CONTACT_POINTS];
        float2 contact_normals[NUMBER_OF_CONTACT_POINTS];
        bool contact_mask[NUMBER_OF_CONTACT_POINTS];
        int contact_wheel[NUMBER_OF_CONTACT_POINTS];

        for (int i=0; i < NUMBER_OF_CONTACT_POINTS; i++) {
            contact_wheel[i] = -1;
            contact_mask[i] = false;
            contact_normals[i] = (float2)(0.0f, 0.0f);
            penetration_depths[i] = 0.0f;
            contact_points[i] = (float2)(0.0f, 0.0f);
        }

        int contact_pointer = 0;

        int start_index = 0;
        int end_index = geo_size-1;

        if (ISLAND_ACCELERATION) {
            int middle_index = (v1.x-ISLAND_START)/ISLAND_STEP;
            // calculate index range of bounding volume
            int bounding_range = ceil(2*bounding_volume/ISLAND_STEP);
            start_index = max(start_index, middle_index-bounding_range);
            end_index = min(end_index, middle_index+bounding_range);
        }

        for(int j=start_index; j<end_index; j++) {

            float2 x0 = geometry[j];
            float2 x1 = geometry[j+1];

            // broad phase collision detection
            if (CircleSegmentIntersection2dsimple(x0,x1,v1,bounding_volume)) {

                // narrow phase collision detection

                // check if wheels hit geometry
                if (true) {
                    for(int k=0; k<NUMBER_OF_WHEELS_PER_CAR; k++) {

                        int wheel_vertex_position = wheel_vertex_positions[wheel_start_index+k];
                        float2 local_wheel_position = vertices[vertex_start_index+wheel_vertex_position];
                        float2 wheel_position = rotate2d(orientation, local_wheel_position)+v1;
                        float wheel_radius = wheel_radii[wheel_start_index+k];

                        if (CircleVertexIntersection2d(x0, wheel_position, wheel_radius, &contact_points[contact_pointer], &contact_normals[contact_pointer], &penetration_depths[contact_pointer]) ||
                            CircleVertexIntersection2d(x1, wheel_position, wheel_radius, &contact_points[contact_pointer], &contact_normals[contact_pointer], &penetration_depths[contact_pointer]) ||
                            CircleSegmentIntersection2d(x0, x1, wheel_position, wheel_radius, &contact_points[contact_pointer], &contact_normals[contact_pointer], &penetration_depths[contact_pointer])) {

                            contact_mask[contact_pointer] = true;
                            contact_wheel[contact_pointer] = k;
                            contact_pointer++;
                            contact_pointer = min(contact_pointer, NUMBER_OF_CONTACT_POINTS-1);
                        }

                    }
               }

                // check if triangle defined by v1, v2, v3 is colliding with geometry
                if (true) {
                    for(int i=0; i<NUMBER_OF_VERTICES_PER_CAR; i++) {
                       int s_index = i+vertex_start_index;
                       int e_index = (i+1)%%NUMBER_OF_VERTICES_PER_CAR+vertex_start_index;

                       float2 v2 = rotate2d(orientations[gid], vertices[s_index])+v1;
                       float2 v3 = rotate2d(orientations[gid], vertices[e_index])+v1;

                       if (detect_collision(x0, x1, v1, v2, v3, &contact_points[contact_pointer], &contact_normals[contact_pointer], &penetration_depths[contact_pointer])) {
                            contact_mask[contact_pointer] = true;
                            contact_pointer++;
                            contact_pointer = min(contact_pointer, NUMBER_OF_CONTACT_POINTS-1);
                            contact_wheel[contact_pointer] = -1;
                            // only consider one side of the vehicle intersection with geometry
                            break;
                       }
                    }
                }

             }
        }

        for (int i=0; i<NUMBER_OF_CONTACT_POINTS; i++) {
            if (contact_mask[i]) {
                float2 n = contact_normals[i];
                float penetration_depth = penetration_depths[i];
                float2 contact_point = contact_points[i];


                float restitution = 0.125;
                n = -n;
                float2 CMToCornerPerp = (contact_point-(centerofmasses[gid]+positions[gid]));
                CMToCornerPerp = (float2)(-CMToCornerPerp.y, CMToCornerPerp.x);
                float2 velocity = velocities[gid] + angular_velocities[gid]*CMToCornerPerp;
                float impulsenumerator = (-(1+restitution)*dot2d(velocities[gid],n));
                float perpdot = dot2d(CMToCornerPerp, n);
                float impulsedenominator = 1.0f/(masses[gid])+1.0f/(inertias[gid])*perpdot*perpdot;
                float impulse = impulsenumerator/impulsedenominator;

                velocities[gid] += (impulse*n)/masses[gid];
                angular_velocities[gid] += impulse*1.0f/(inertias[gid])*perpdot;

                // poor man's friction
                velocities[gid] *= 0.989;
                positions[gid] += penetration_depth*n;


                if (contact_wheel[i] != -1) {
                    // calculate torque of wheel idea taken from http://www.boxcar2d.com/about.html
                    float2 slope = perp(n);
                    int wheel_index = contact_wheel[i]+wheel_start_index;
                    // TODO
                    wheel_momenta[wheel_index] = (3*wheel_masses[wheel_index]*GRAVITY*sin(atan2(slope.y,slope.x))/wheel_radii[wheel_index]);
                }
            }

        }

        for (int i=0; i<NUMBER_OF_CONTACT_POINTS; i++) {
            if (contact_mask[i]) {
                float2 n = -contact_normals[i];
                float2 contact_point = contact_points[i];
                // rotate vectors for visualization
                global_contact_points[contact_point_id+i] = rotate2d(-orientations[gid], contact_point-positions[gid]);
                global_contact_normals[contact_point_id+i] = rotate2d(-orientations[gid], n);
                //global_contact_normals[contact_point_id+i] = velocities[gid];
            } else {
                global_contact_points[contact_point_id+i] = (float2)(0,0);
            }

         }
}



__kernel void roulette_wheel_selection(__global float* selection_probabilities,
                             __global float* accumulated_probabilities,
                             __global uint* population_indexes,
                             uint population_size) {
     int gid = get_global_id(0);
     float probability = selection_probabilities[gid];

     // bisection to find index
     uint low = 0;
     uint high = population_size;
     uint mid;

     while (low < high) {
        mid = (low+high)/2;
        if (accumulated_probabilities[mid] < probability) {
            low = mid+1;
        } else {
            high = mid;
        }
     }

     population_indexes[gid] = low;
}

__kernel void crossover(__global float* magnitudes,
                        __global float* angles,
                        __global float4* vertex_colors,
                        __global float* old_magnitudes,
                        __global float* old_angles,
                        __global float4* old_vertex_colors,
                        __global float* wheel_vertex_positions,
                        __global float* wheel_radii,
                        __global float* old_wheel_vertex_positions,
                        __global float* old_wheel_radii,
                        __global int* indexes,
                        __global int* crossover_magnitude_array,
                        __global int* crossover_angle_array,
                        __global int* crossover_wheel_array) {

    int gid = get_global_id(0);

    int ai = indexes[2*gid];
    int start_aiv = ai*NUMBER_OF_VERTICES_PER_CAR;

    int bi = indexes[2*gid+1];
    int start_biv = bi*NUMBER_OF_VERTICES_PER_CAR;

    int cross_ai = 2*gid;
    int start_cross_aiv = cross_ai*NUMBER_OF_VERTICES_PER_CAR;

    int cross_bi = 2*gid+1;
    int start_cross_biv = cross_bi*NUMBER_OF_VERTICES_PER_CAR;

    int crossover_id = gid*CROSSOVER_POINTS;

    // crossover angles
    for(int i=0; i<NUMBER_OF_VERTICES_PER_CAR; i++) {
        bool cross = false;
        for (int j=0; j<CROSSOVER_POINTS; j++) {
           if(crossover_angle_array[crossover_id+j] == i) {
              cross = true;
           }
        }

        if (cross) {
            angles[start_cross_biv+i] = old_angles[start_aiv+i];
            vertex_colors[start_cross_biv+i] = old_vertex_colors[start_aiv+i];

            angles[start_cross_aiv+i] = old_angles[start_biv+i];
            vertex_colors[start_cross_aiv+i] = old_vertex_colors[start_biv+i];
        } else {
            angles[start_cross_aiv+i] = old_angles[start_aiv+i];
            vertex_colors[start_cross_aiv+i] = old_vertex_colors[start_aiv+i];

            angles[start_cross_biv+i] = old_angles[start_biv+i];
            vertex_colors[start_cross_biv+i] = old_vertex_colors[start_biv+i];
        }
    }

    // crossover magnitudes
    for(int i=0; i<NUMBER_OF_VERTICES_PER_CAR; i++) {
        bool cross = false;
        for (int j=0; j<CROSSOVER_POINTS; j++) {
           if(crossover_magnitude_array[crossover_id+j] == i) {
              cross = true;
           }
        }

        if (cross) {
            magnitudes[start_cross_biv+i] = old_magnitudes[start_aiv+i];
            vertex_colors[start_cross_biv+i] = old_vertex_colors[start_aiv+i];

            magnitudes[start_cross_aiv+i] = old_magnitudes[start_biv+i];
            vertex_colors[start_cross_aiv+i] = old_vertex_colors[start_biv+i];
        } else {
            magnitudes[start_cross_aiv+i] = old_magnitudes[start_aiv+i];
            vertex_colors[start_cross_aiv+i] = old_vertex_colors[start_aiv+i];

            magnitudes[start_cross_biv+i] = old_magnitudes[start_biv+i];
            vertex_colors[start_cross_biv+i] = old_vertex_colors[start_biv+i];
        }
    }

    int start_aiw = ai*NUMBER_OF_WHEELS_PER_CAR;
    int start_biw = bi*NUMBER_OF_WHEELS_PER_CAR;
    int start_cross_aiw = cross_ai*NUMBER_OF_WHEELS_PER_CAR;
    int start_cross_biw = cross_bi*NUMBER_OF_WHEELS_PER_CAR;

    // crossover wheel positions and wheel radii
    for(int i=0; i<NUMBER_OF_WHEELS_PER_CAR; i++) {
        bool cross = false;
        for (int j=0; j<CROSSOVER_POINTS; j++) {
           if(crossover_wheel_array[crossover_id+j] == i) {
              cross = true;
           }
        }
        if (cross) {
            wheel_vertex_positions[start_cross_biw+i] = old_wheel_vertex_positions[start_aiw+i];
            wheel_radii[start_cross_biw+i] = old_wheel_radii[start_aiw+i];

            wheel_vertex_positions[start_cross_aiw+i] = old_wheel_vertex_positions[start_biw+i];
            wheel_radii[start_cross_aiw+i] = old_wheel_radii[start_biw+i];
        } else {
            wheel_vertex_positions[start_cross_aiw+i] = old_wheel_vertex_positions[start_aiw+i];
            wheel_radii[start_cross_aiw+i] = old_wheel_radii[start_aiw+i];

            wheel_vertex_positions[start_cross_biw+i] = old_wheel_vertex_positions[start_biw+i];
            wheel_radii[start_cross_biw+i] = old_wheel_radii[start_biw+i];
        }
    }

}


__kernel void mutate(__global float* magnitudes, __global int* mutation_indexes, __global float* mutated_magnitudes, __global float4* mutation_colors, __global float4* vertex_colors, int index_offset, int offset_value) {
    int gid = get_global_id(0);
    int start_index = gid*index_offset;

    for(int i=0; i<index_offset; i++) {
        for(int j=0; j<POINT_MUTATIONS; j++) {
            if (mutation_indexes[gid*POINT_MUTATIONS+j] == i) {
                if (offset_value == 1)
                    magnitudes[start_index+i] += mutated_magnitudes[gid*POINT_MUTATIONS+j];
                else
                    magnitudes[start_index+i] = mutated_magnitudes[gid*POINT_MUTATIONS+j];
                vertex_colors[start_index+i] = mutation_colors[gid*POINT_MUTATIONS+j];
            }
        }
    }

}


__kernel void mutate_int(__global int* magnitudes, __global int* mutation_indexes, __global int* mutated_magnitudes, __global float4* mutation_colors, __global float4* vertex_colors, int index_offset) {
    int gid = get_global_id(0);
    int start_index = gid*index_offset;

    for(int i=0; i<index_offset; i++) {
        for(int j=0; j<POINT_MUTATIONS; j++) {
            if (mutation_indexes[gid*POINT_MUTATIONS+j] == i) {
                magnitudes[start_index+i] = mutated_magnitudes[gid*POINT_MUTATIONS+j];
                vertex_colors[start_index+i] = mutation_colors[gid*POINT_MUTATIONS+j];
            }
        }
    }

}