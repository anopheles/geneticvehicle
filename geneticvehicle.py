#!/usr/bin/env python
# encoding: utf-8

from __future__ import division
from IPython.lib.latextools import genelatex
from pyopencl import clrandom
from pyopencl import array
from pyopencl import scan
from functools import partial
from itertools import izip_longest
import pyopencl
import time
import numpy
import sys
import os

def load_file(filename):
    with open(filename) as file_handler:
        return "".join(file_handler.readlines())

def xgrouper(n, iterable, fillvalue=None):
    return zip(*[iter(iterable)]*n)

def grouper(n, iterable, fillvalue=None):
    "grouper(3, 'abcdefg', 'x') --> ('a','b','c'), ('d','e','f'), ('g','x','x')"
    return izip_longest(*[iter(iterable)]*n, fillvalue=fillvalue)

class  GeneticVehicle(object):

    def __init__(self, number_of_cars=1024, number_of_vertices_per_car=4, number_of_wheels_per_car=3):
        self.number_of_vertices_per_car = number_of_vertices_per_car
        self.number_of_cars = number_of_cars
        self.number_of_wheels_per_car = number_of_wheels_per_car
        self.number_of_contact_points = 4
        self.density = 0.5
        self.counter = 0
        self.steps = 1
        self.delta = 1/16
        self.satisfy_constraints = 4
        self.crossover_points = 2
        self.point_mutations = 1
        self.process = False
        self.generation = 1
        self.run = False
        self.island = SlopeXtreme()

        # vehicle parameters, used to build vehicle and for mutation
        self.min_radius = 5
        self.max_radius = 8
        self.angle_displacement=0.5
        self.min_magnitude = 0
        self.max_magnitude = 15

        self.build()

    def build(self):
        self.compile_source()
        self.generate_data_structures()

    def compile_source(self):
        self.context = pyopencl.create_some_context()
        self.queue = pyopencl.CommandQueue(self.context)
        self.mf = pyopencl.mem_flags

        opencl_source = load_file("geneticvehicle.cl") % {
                                "vertices_per_car" : self.number_of_vertices_per_car,
                                "number_of_cars" : self.number_of_cars,
                                "density" : self.density,
                                "number_of_wheels" : self.number_of_wheels_per_car,
                                "number_of_contact_points" : self.number_of_contact_points,
                                "island_start" : self.island.island_start,
                                "island_step" : self.island.island_step,
                                "island_end" : self.island.island_end,
                                "island_acceleration"  : int(self.island.island_acceleration),
                                "island_range" : self.island.range(),
                                "crossover_points" : self.crossover_points,
                                "point_mutations" : self.point_mutations}

        self.program = pyopencl.Program(self.context, opencl_source)

        try:
            self.program.build()
        except Exception as why:
            print why
            print(self.program.get_build_info(self.context.devices[0], pyopencl.program_build_info.LOG))


    def simulation_step(self, pre_callback=None, post_callback=None):
        if self.process or self.run:
            self.process = False

            if pre_callback:
                pre_callback()

            for round in range(self.steps):
                self.counter += 1
                self.program.calculate_loads(self.queue, (self.number_of_cars,), None, self.vehicle_forces.data,
                                                                                       self.vehicle_momenta.data,
                                                                                       self.vehicle_velocities.data,
                                                                                       self.vehicle_angular_velocities.data)

                self.program.integrate(self.queue, (self.number_of_cars,), None, self.vehicle_alive.data,
                                                                                 self.vehicle_positions.data,
                                                                                 self.vehicle_masses.data,
                                                                                 self.vehicle_forces.data,
                                                                                 self.vehicle_velocities.data,
                                                                                 self.vehicle_angular_velocities.data,
                                                                                 self.vehicle_orientations.data,
                                                                                 self.vehicle_momenta.data,
                                                                                 self.vehicle_inertias.data,
                                                                                 numpy.float32(self.delta),
                                                                                 self.wheel_angular_velocities.data,
                                                                                 self.wheel_radii.data,
                                                                                 self.wheel_momenta.data,
                                                                                 self.wheel_masses.data,
                                                                                 self.wheel_inertias.data,
                                                                                 self.wheel_orientations.data,
                                                                                 self.vehicle_vertices.data,
                                                                                 self.wheel_vertex_positions.data)

                for _ in range(self.satisfy_constraints):
                    self.program.collision(self.queue, (self.number_of_cars,), None, self.vehicle_alive.data,
                                                                                self.vehicle_positions.data,
                                                                                self.vehicle_masses.data,
                                                                                self.vehicle_velocities.data,
                                                                                self.vehicle_orientations.data,
                                                                                self.geometry.data,
                                                                                numpy.int32(len(self.geometry)),
                                                                                self.vehicle_vertices.data,
                                                                                self.vehicle_inertias.data,
                                                                                self.vehicle_angular_velocities.data,
                                                                                self.vehicle_contact_points.data,
                                                                                self.vehicle_contact_normals.data,
                                                                                self.vehicle_center_masses.data,
                                                                                numpy.float32(self.delta),
                                                                                self.wheel_vertex_positions.data,
                                                                                self.wheel_radii.data,
                                                                                self.vehicle_bounding_volumes.data,
                                                                                self.wheel_momenta.data,
                                                                                self.wheel_masses.data)

            self.program.assign_score(self.queue, (self.number_of_cars,), None, self.vehicle_positions.data, self.vehicle_velocities.data, self.vehicle_score.data)
            if not self.counter % 500:
                self.program.evaluate_score(self.queue, (self.number_of_cars,), None, self.vehicle_score.data, self.vehicle_old_score.data, self.vehicle_alive.data)
                self.vehicle_score, self.vehicle_old_score = self.vehicle_old_score, self.vehicle_score

            if post_callback:
                post_callback()

    def generate_data_structures(self, wheel_vertex_positions=None, wheel_radii=None, magnitudes=None, angles=None, vertex_colors=None):
        self.work_items = 64*self.queue.device.max_compute_units

        self.vehicle_positions = pyopencl.array.Array(self.queue, (self.number_of_cars, 2), dtype=numpy.float32)
        self.vehicle_score = pyopencl.array.Array(self.queue, self.number_of_cars, dtype=numpy.float32)
        self.vehicle_old_score = pyopencl.array.Array(self.queue, self.number_of_cars, dtype=numpy.float32)

        # for testing purposes
        centers = numpy.zeros((self.number_of_cars, 2), dtype=numpy.float32)
        for index, x in enumerate(numpy.nditer(centers, op_flags=['readwrite'])):
            if index % 2 == 0:
                #x[...] = (index-2) * 10
                x[...] = 0
            else:
                x[...] = -20
                #x[...] = -20-2*index
                #x[...] = -0.5*(index-2)**2
        self.vehicle_positions.set(centers)

        self.generate_vertices(magnitudes=magnitudes, angles=angles, vertex_colors=vertex_colors)
        self.generate_vehicle_properties()
        self.generate_wheel_properties(wheel_vertex_positions=wheel_vertex_positions, wheel_radii=wheel_radii)
        self.generate_bounding_volumes()

        # for testing purposes
        #pyopencl.clrandom.RanluxGenerator(queue, 64*queue.device.max_compute_units, seed=time.time()).fill_uniform(self.vehicle_alive, a=0.75, b=1.25)

        self.geometry_points = self.island.generate_geometry()
        self.geometry = pyopencl.array.zeros(self.queue, (len(self.geometry_points), 2), dtype=numpy.float32)
        self.geometry.set(numpy.array(self.geometry_points, dtype=numpy.float32))


    def generate_bounding_volumes(self):
        self.vehicle_bounding_volumes = pyopencl.array.Array(self.queue, self.number_of_cars, dtype=numpy.float32)
        self.program.generate_bounding_volumes(self.queue, (self.number_of_cars,), None, self.vehicle_vertices.data, self.wheel_vertex_positions.data, self.wheel_radii.data, self.vehicle_bounding_volumes.data)

    def generate_wheel_properties(self, wheel_vertex_positions=None, wheel_radii=None):
        if not wheel_vertex_positions:
            self.wheel_vertex_positions = pyopencl.array.zeros(self.queue, (self.number_of_cars*self.number_of_wheels_per_car), dtype=numpy.int32)
            pyopencl.clrandom.RanluxGenerator(self.queue, self.work_items, seed=time.time()).fill_uniform(self.wheel_vertex_positions, a=0, b=self.number_of_vertices_per_car)
        else:
            self.wheel_vertex_positions = wheel_vertex_positions

        if not wheel_radii:
            self.wheel_radii = pyopencl.array.zeros(self.queue, (self.number_of_cars*self.number_of_wheels_per_car), dtype=numpy.float32)
            pyopencl.clrandom.RanluxGenerator(self.queue, self.work_items, seed=time.time()).fill_uniform(self.wheel_radii, a=self.min_radius, b=self.max_radius)
        else:
            self.wheel_radii = wheel_radii

        self.wheel_masses = pyopencl.array.Array(self.queue, (self.number_of_cars*self.number_of_wheels_per_car), dtype=numpy.float32)
        self.wheel_inertias = pyopencl.array.Array(self.queue, (self.number_of_cars*self.number_of_wheels_per_car), dtype=numpy.float32)
        self.wheel_angular_velocities = pyopencl.array.zeros(self.queue, (self.number_of_cars*self.number_of_wheels_per_car), dtype=numpy.float32)
        self.wheel_momenta = pyopencl.array.zeros(self.queue, (self.number_of_cars*self.number_of_wheels_per_car), dtype=numpy.float32)
        self.wheel_orientations = pyopencl.array.zeros(self.queue, (self.number_of_cars*self.number_of_wheels_per_car), dtype=numpy.float32)
        self.program.generate_wheel_properties(self.queue, (self.number_of_cars*self.number_of_wheels_per_car,), None, self.wheel_masses.data, self.wheel_inertias.data, self.wheel_radii.data)

    def generate_vehicle_properties(self):
        self.vehicle_masses = pyopencl.array.Array(self.queue, self.number_of_cars, dtype=numpy.float32)
        self.vehicle_alive = pyopencl.array.Array(self.queue, self.number_of_cars, dtype=numpy.float32)
        self.vehicle_alive.fill(1)
        self.vehicle_inertias = pyopencl.array.Array(self.queue, self.number_of_cars, dtype=numpy.float32)
        self.vehicle_center_masses = pyopencl.array.Array(self.queue, (self.number_of_cars, 2), dtype=numpy.float32)
        self.vehicle_velocities = pyopencl.array.zeros(self.queue, (self.number_of_cars, 2), dtype=numpy.float32)
        self.vehicle_contact_points = pyopencl.array.zeros(self.queue, (self.number_of_cars*self.number_of_contact_points, 2), dtype=numpy.float32)
        self.vehicle_contact_normals = pyopencl.array.zeros(self.queue, (self.number_of_cars*self.number_of_contact_points, 2), dtype=numpy.float32)
        self.vehicle_angular_velocities = pyopencl.array.zeros(self.queue, self.number_of_cars, dtype=numpy.float32)
        self.vehicle_orientations = pyopencl.array.zeros(self.queue, self.number_of_cars, dtype=numpy.float32)
        self.vehicle_forces = pyopencl.array.zeros(self.queue, (self.number_of_cars,2), dtype=numpy.float32)
        self.vehicle_momenta = pyopencl.array.zeros(self.queue, self.number_of_cars, dtype=numpy.float32)
        pyopencl.clrandom.RanluxGenerator(self.queue, self.work_items, seed=time.time()).fill_uniform(self.vehicle_momenta, a=-25000, b=25000)
        self.vehicle_vertices.reshape((self.number_of_cars*self.number_of_vertices_per_car, 2))
        self.program.generate_vehicle_properties(self.queue, (self.number_of_cars,), None, self.vehicle_masses.data, self.vehicle_center_masses.data, self.vehicle_inertias.data, self.vehicle_vertices.data)

    def generate_vertices(self, chaos=False, magnitudes=None, angles=None, vertex_colors=None):
        if not magnitudes:
            self.magnitudes = pyopencl.array.Array(self.queue, (self.number_of_cars*self.number_of_vertices_per_car),dtype=numpy.float32)
            pyopencl.clrandom.RanluxGenerator(self.queue, self.work_items, seed=time.time()).fill_uniform(self.magnitudes, a=self.min_magnitude, b=self.max_magnitude)
        else:
            self.magnitudes = magnitudes

        if not angles:
            offset_angles = pyopencl.array.zeros(self.queue, (self.number_of_cars*self.number_of_vertices_per_car, 1), dtype=numpy.float32)
            pyopencl.clrandom.RanluxGenerator(self.queue, self.work_items, seed=time.time()).fill_uniform(offset_angles, a=-self.angle_displacement, b=self.angle_displacement)
            self.angles = pyopencl.array.zeros(self.queue, (self.number_of_cars*self.number_of_vertices_per_car, 1), dtype=numpy.float32)
            if chaos:
                pyopencl.clrandom.RanluxGenerator(self.queue, self.work_items, seed=time.time()).fill_uniform(self.angles, a=0, b=2*numpy.pi)
            else:
                ordered_angles = pyopencl.array.arange(self.queue, 0, 2*numpy.pi, (2*numpy.pi)/self.number_of_vertices_per_car, dtype=numpy.float32)
                self.program.generate_angles(self.queue, self.angles.shape, None, self.angles.data, ordered_angles.data)
            self.angles += offset_angles
        else:
            self.angles = angles

        if not vertex_colors:
            vehicle_colors = pyopencl.array.Array(self.queue, (self.number_of_cars, 4) , dtype=numpy.float32)
            pyopencl.clrandom.RanluxGenerator(self.queue, self.work_items, seed=time.time()).fill_uniform(vehicle_colors, a=0, b=255)
            self.vertex_colors = pyopencl.array.Array(self.queue, (self.number_of_cars*self.number_of_vertices_per_car, 4) , dtype=numpy.float32)
            self.program.generate_colors(self.queue, (self.number_of_cars, 1), None, self.vertex_colors.data, vehicle_colors.data)
        else:
            self.vertex_colors = vertex_colors

        self.vehicle_vertices = pyopencl.array.Array(self.queue, (self.number_of_cars*self.number_of_vertices_per_car, 2), dtype=numpy.float32)
        self.program.generate_vertices(self.queue, self.vehicle_vertices.shape, None, self.magnitudes.data, self.angles.data, self.vehicle_vertices.data)

    def get_sorted_score_ids(self):
        return self.vehicle_score.get().argsort()[::-1]

    def get_vehicle_information(self, vehicle_id):
        center = self.vehicle_positions.get()[vehicle_id]
        vertices = xgrouper(self.number_of_vertices_per_car, self.vehicle_vertices.get())[vehicle_id]
        bounding_radius = self.vehicle_bounding_volumes.get()[vehicle_id]
        vertex_colors = xgrouper(self.number_of_vertices_per_car, self.vertex_colors.get())[vehicle_id]
        wheel_radii = xgrouper(self.number_of_wheels_per_car, self.wheel_radii.get())[vehicle_id]
        wheel_vertex = xgrouper(self.number_of_wheels_per_car, self.wheel_vertex_positions.get())[vehicle_id]

        return center, vertices, bounding_radius, vertex_colors, wheel_radii, wheel_vertex

    def get_chromosome(self, vehicle_id):
        columns = ""
        values = ""

        magnitudes = xgrouper(self.number_of_vertices_per_car, self.magnitudes.get())[vehicle_id]
        angles = xgrouper(self.number_of_vertices_per_car, self.angles.get())[vehicle_id]
        wheel_radii = xgrouper(self.number_of_wheels_per_car, self.wheel_radii.get())[vehicle_id]
        wheel_vertex = xgrouper(self.number_of_wheels_per_car, self.wheel_vertex_positions.get())[vehicle_id]

        data = {"magnitudes": magnitudes,
                "angles": angles,
                "radii": wheel_radii,
                "wheel vertex": wheel_vertex}

        for column_name, values in data.items():
            for index, value in enumerate(values):
                columns += column_name+str(index) + "\t"
                values += str(value) + "\t"
            #columns += "\t"
            #valuess += "\t"

        return columns + "\n" + values

    def evolve(self):
        # TODO dependancies are clearly wrong here
        from visualization import display_vehicles

        def generate(wheel_vertex_positions, wheel_radii, magnitudes, angles, vertex_colors):
            self.generate_data_structures(wheel_vertex_positions=wheel_vertex_positions,
                                          wheel_radii=wheel_radii,
                                          magnitudes=magnitudes,
                                          angles=angles,
                                          vertex_colors=vertex_colors)

        if self.number_of_cars > 2:
            display_vehicles(genetic_vehicle, range(self.number_of_cars), filename="%s_1_pool" % genetic_vehicle.generation)
            display_vehicles(genetic_vehicle, genetic_vehicle.get_sorted_score_ids()[0:10], filename="%s_2_top" % genetic_vehicle.generation)
            #selection = self.roulette_wheel_selection(population_size=int(self.number_of_cars/2))
            selection = self.roulette_wheel_selection(population_size=int(self.number_of_cars))
            print "Chromosome\n", genetic_vehicle.get_chromosome(genetic_vehicle.get_sorted_score_ids()[0])
            if selection:
                display_vehicles(genetic_vehicle, selection.get(), filename="%s_3_selection" % genetic_vehicle.generation)
                crossover = self.crossover(selection)
                generate(*crossover)
                display_vehicles(genetic_vehicle, range(self.number_of_cars), filename="%s_4_crossover" % genetic_vehicle.generation)
                mutation = self.mutate(*crossover)
                generate(*mutation)
                display_vehicles(genetic_vehicle, range(self.number_of_cars), filename="%s_5_mutation" % genetic_vehicle.generation)
                self.generation += 1
                return True

    def mutate(self, wheel_vertex_positions, wheel_radii, magnitudes, angles, vertex_colors):
        # TODO the code below needs some serious refactorization
        if self.point_mutations > 0:
            # mutate magnitude, angle, vehicle_radii and wheel vertex position separately
            mutation_colors = pyopencl.array.Array(self.queue, ((self.number_of_cars*self.point_mutations), 4), dtype=numpy.float32)
            pyopencl.clrandom.RanluxGenerator(self.queue, self.work_items, seed=time.time()).fill_uniform(mutation_colors, a=0, b=255)
            mutated_magnitudes = pyopencl.array.Array(self.queue, (self.number_of_cars*self.point_mutations), dtype=numpy.float32)
            pyopencl.clrandom.RanluxGenerator(self.queue, self.work_items, seed=time.time()).fill_uniform(mutated_magnitudes, a=self.min_magnitude, b=self.max_magnitude)
            mutation_indexes = pyopencl.array.Array(self.queue, (self.number_of_cars*self.point_mutations), dtype=numpy.int32)
            pyopencl.clrandom.RanluxGenerator(self.queue, self.work_items, seed=time.time()).fill_uniform(mutation_indexes, a=0, b=self.number_of_vertices_per_car)
            self.program.mutate(self.queue, (self.number_of_cars, 1), None, magnitudes.data, mutation_indexes.data, mutated_magnitudes.data, mutation_colors.data, vertex_colors.data, numpy.int32(self.number_of_vertices_per_car), numpy.int32(0))

            mutation_colors = pyopencl.array.Array(self.queue, ((self.number_of_cars*self.point_mutations), 4), dtype=numpy.float32)
            pyopencl.clrandom.RanluxGenerator(self.queue, self.work_items, seed=time.time()).fill_uniform(mutation_colors, a=0, b=255)
            mutated_angles = pyopencl.array.Array(self.queue, (self.number_of_cars*self.point_mutations), dtype=numpy.float32)
            pyopencl.clrandom.RanluxGenerator(self.queue, self.work_items, seed=time.time()).fill_uniform(mutated_angles, a=-self.angle_displacement, b=self.angle_displacement)
            mutation_indexes = pyopencl.array.Array(self.queue, (self.number_of_cars*self.point_mutations), dtype=numpy.int32)
            pyopencl.clrandom.RanluxGenerator(self.queue, self.work_items, seed=time.time()).fill_uniform(mutation_indexes, a=0, b=self.number_of_vertices_per_car)
            self.program.mutate(self.queue, (self.number_of_cars, 1), None, angles.data, mutation_indexes.data, mutated_angles.data, mutation_colors.data, vertex_colors.data, numpy.int32(self.number_of_vertices_per_car), numpy.int32(1))

            mutation_colors = pyopencl.array.Array(self.queue, ((self.number_of_cars*self.point_mutations), 4), dtype=numpy.float32)
            pyopencl.clrandom.RanluxGenerator(self.queue, self.work_items, seed=time.time()).fill_uniform(mutation_colors, a=0, b=255)
            mutated_radii = pyopencl.array.Array(self.queue, (self.number_of_cars*self.point_mutations), dtype=numpy.float32)
            pyopencl.clrandom.RanluxGenerator(self.queue, self.work_items, seed=time.time()).fill_uniform(mutated_radii, a=self.min_radius, b=self.max_radius)
            mutation_indexes = pyopencl.array.Array(self.queue, (self.number_of_cars*self.point_mutations), dtype=numpy.int32)
            pyopencl.clrandom.RanluxGenerator(self.queue, self.work_items, seed=time.time()).fill_uniform(mutation_indexes, a=0, b=self.number_of_wheels_per_car)
            self.program.mutate(self.queue, (self.number_of_cars, 1), None, wheel_radii.data, mutation_indexes.data, mutated_radii.data, mutation_colors.data, vertex_colors.data, numpy.int32(self.number_of_wheels_per_car), numpy.int32(0))

            mutation_colors = pyopencl.array.Array(self.queue, ((self.number_of_cars*self.point_mutations), 4), dtype=numpy.float32)
            pyopencl.clrandom.RanluxGenerator(self.queue, self.work_items, seed=time.time()).fill_uniform(mutation_colors, a=0, b=255)
            mutated_wheel_vertex_positions = pyopencl.array.Array(self.queue, (self.number_of_cars*self.point_mutations), dtype=numpy.int32)
            pyopencl.clrandom.RanluxGenerator(self.queue, self.work_items, seed=time.time()).fill_uniform(mutated_wheel_vertex_positions, a=0, b=self.number_of_vertices_per_car)
            mutation_indexes = pyopencl.array.Array(self.queue, (self.number_of_cars*self.point_mutations), dtype=numpy.int32)
            pyopencl.clrandom.RanluxGenerator(self.queue, self.work_items, seed=time.time()).fill_uniform(mutation_indexes, a=0, b=self.number_of_wheels_per_car)
            self.program.mutate(self.queue, (self.number_of_cars, 1), None, wheel_vertex_positions.data, mutation_indexes.data, mutated_wheel_vertex_positions.data, mutation_colors.data, vertex_colors.data, numpy.int32(self.number_of_wheels_per_car), numpy.int32(0))

        return wheel_vertex_positions, wheel_radii, magnitudes, angles, vertex_colors

    def roulette_wheel_selection(self, population_size=32):
        # calculate sum over all scores
        total_score = pyopencl.array.sum(self.vehicle_score).get()

        if total_score > 0:
            from pyopencl.elementwise import ElementwiseKernel
            roulette_wheel_probabilities = ElementwiseKernel(self.context,
                    "float total_score, float *scores, "
                    "float *probabilities",
                    "probabilities[i] = scores[i]/total_score",
                    "roulette_wheel_probabilities")
            probabilities = pyopencl.array.empty_like(self.vehicle_score)
            roulette_wheel_probabilities(total_score, self.vehicle_score, probabilities)

            accumulated_probabilities_kernel = pyopencl.scan.InclusiveScanKernel(self.context, numpy.float32, "a+b")
            accumulated_probabilities_kernel(probabilities)

            selection_probabilities = pyopencl.array.Array(self.queue, population_size, dtype=numpy.float32)
            pyopencl.clrandom.RanluxGenerator(self.queue, self.work_items, seed=time.time()).fill_uniform(selection_probabilities)
            population_indexes = pyopencl.array.Array(self.queue, population_size, dtype=numpy.uint32)

            self.program.roulette_wheel_selection(self.queue, (population_size,), None, selection_probabilities.data, probabilities.data, population_indexes.data, numpy.uint32(self.number_of_cars))
            associated_scores = array.take(self.vehicle_score, population_indexes, queue=self.queue)

            return population_indexes

    def crossover(self, indexes):
        # remember old magnitudes and angles
        old_magnitudes = self.magnitudes
        old_angles = self.angles
        old_number_of_cars = self.number_of_cars
        old_wheel_vertex_positions = self.wheel_vertex_positions
        old_wheel_radii = self.wheel_radii
        old_vertex_colors = self.vertex_colors

        self.number_of_cars = len(indexes)

        # create magnitude and angle arrays for offspring
        magnitudes = pyopencl.array.Array(self.queue, (self.number_of_cars*self.number_of_vertices_per_car), dtype=numpy.float32)
        angles = pyopencl.array.Array(self.queue, (self.number_of_cars*self.number_of_vertices_per_car), dtype=numpy.float32)
        vertex_colors = pyopencl.array.Array(self.queue, (self.number_of_cars*self.number_of_vertices_per_car, 4), dtype=numpy.float32)

        # create wheel vertex position and wheel radii for offspring
        wheel_vertex_positions = pyopencl.array.Array(self.queue, (self.number_of_cars*self.number_of_wheels_per_car), dtype=numpy.int32)
        wheel_radii = pyopencl.array.Array(self.queue, (self.number_of_cars*self.number_of_wheels_per_car), dtype=numpy.float32)

        crossover_magnitude_array = pyopencl.array.Array(self.queue, (self.number_of_cars*self.crossover_points), dtype=numpy.int32)
        pyopencl.clrandom.RanluxGenerator(self.queue, self.work_items, seed=time.time()).fill_uniform(crossover_magnitude_array, a=0, b=self.number_of_vertices_per_car)

        crossover_angle_array = pyopencl.array.Array(self.queue, (self.number_of_cars*self.crossover_points), dtype=numpy.int32)
        pyopencl.clrandom.RanluxGenerator(self.queue, self.work_items, seed=time.time()).fill_uniform(crossover_angle_array, a=0, b=self.number_of_vertices_per_car)

        crossover_wheel_array = pyopencl.array.Array(self.queue, (self.number_of_cars*self.crossover_points), dtype=numpy.int32)
        pyopencl.clrandom.RanluxGenerator(self.queue, self.work_items, seed=time.time()).fill_uniform(crossover_wheel_array, a=0, b=self.number_of_wheels_per_car)

        self.program.crossover(self.queue, (int(self.number_of_cars/2),), None, magnitudes.data,
                                                                                angles.data,
                                                                                vertex_colors.data,
                                                                                old_magnitudes.data,
                                                                                old_angles.data,
                                                                                old_vertex_colors.data,
                                                                                wheel_vertex_positions.data,
                                                                                wheel_radii.data,
                                                                                old_wheel_vertex_positions.data,
                                                                                old_wheel_radii.data,
                                                                                indexes.data,
                                                                                crossover_magnitude_array.data,
                                                                                crossover_angle_array.data,
                                                                                crossover_wheel_array.data)

        return wheel_vertex_positions, wheel_radii, magnitudes, angles, vertex_colors


class Island(object):

    def __init__(self, start=-20000, end=20000, step=50):
        self.island_acceleration = True
        self.island_start = start
        self.island_end = end
        self.island_step = step

        self.x_values  = numpy.linspace(self.island_start, self.island_end, self.range()/self.island_step)
        self.y_values  = numpy.random.random_sample(self.range()/self.island_step)*10

    def range(self):
        return numpy.abs(self.island_start)+numpy.abs(self.island_end)

    def generate_geometry(self):
        return numpy.array(zip(self.x_values, self.y_values), dtype=numpy.float32)


class Rocky(Island):

    def __init__(self):
        Island.__init__(self, step=5)
        self.y_values = numpy.random.random_sample(self.range()/self.island_step)*90


class Slope(Island):

    def __init__(self):
        Island.__init__(self, step=20)
        right_slope = -5*numpy.arange(self.range()/self.island_step/2)
        self.y_values = 15*numpy.random.random_sample(self.range()/self.island_step)+(numpy.concatenate((right_slope[::-1], right_slope)))


class SlopeXtreme(Island):

    def __init__(self):
        Island.__init__(self, step=10)
        right_slope = -5*numpy.arange(self.range()/self.island_step/2)
        right_multiplier = 0.0025*numpy.arange(self.range()/self.island_step/2)**2
        left_multiplier = numpy.zeros(self.range()/self.island_step/2)+0
        self.y_values = numpy.concatenate((left_multiplier, right_multiplier))*numpy.random.random_sample(self.range()/self.island_step)+(numpy.concatenate((-right_slope[::-1], right_slope)))


class Parabola(Island):

    def __init__(self):
        Island.__init__(self)
        right_side = -0.05*numpy.arange(self.range()/self.island_step/2)**2
        self.y_values = numpy.concatenate((right_side[::-1], right_side))


def show_histogram(genetic_vehicle):
    import matplotlib.pyplot as plt
    bins = 64
    plt.clf()
    plt.hist(genetic_vehicle.vehicle_score.get(), bins)
    plt.title("Score Distribution (population size: %s, generation: %s)" % (genetic_vehicle.number_of_cars, genetic_vehicle.generation))
    plt.xlabel("Scores (%s bins)" % bins)
    plt.ylabel("Count")
    plt.ion()
    plt.draw()
    plt.show()
    return plt


def save_histogram(plt):
    results_directory = "results"
    if not os.path.isdir(results_directory):
        os.mkdir(results_directory)
    plt.savefig(os.path.join(results_directory, "figure%s.png" % genetic_vehicle.generation))

if __name__ == '__main__':
    genetic_vehicle = GeneticVehicle()
    genetic_vehicle.steps = 50
    genetic_vehicle.run = True
    genetic_vehicle.number_of_cars = 1024

    def callback(counter):
        if genetic_vehicle.number_of_cars < 4:
            print "number of cars too low"
            sys.exit()
        if counter % 4 == 0:
            alive_total_ratio = pyopencl.array.sum(genetic_vehicle.vehicle_alive).get()/genetic_vehicle.number_of_cars*100
            print "alive/total ratio", alive_total_ratio
            if alive_total_ratio < genetic_vehicle.number_of_cars/64 or counter > 10**5:
                print "evolving"
                save_histogram(show_histogram(genetic_vehicle))
                if not genetic_vehicle.evolve():
                    print "evolving failed"
                    sys.exit()

    counter = 0
    while True:
        genetic_vehicle.simulation_step(post_callback=partial(callback, counter))
        counter += 1



