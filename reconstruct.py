#!/usr/bin/env python3
"""Reconstructs an obj file from its json surface-depth parameterization.

Usage:
  reconstruct.py json <file>
"""
from docopt import docopt
import numpy as np
import json
import os

class CubeMesh:
    def __init__ (self, corner_vertices):
        enforce(corner_vertices.shape == (8, 3), 
            "invalid vertex shape: expected (8, 3), got %s",
            corner_vertices.shape)

        self.verts = corner_vertices

        # planes on [0 .. 6), corresponds to axis directions
        #    [ -x, +x, -y, +y, -z, +z ]
        #
        # conceptually, plane = axis + direction.
        # numerically, indices can be related as follows:
        #   axis on [0 .. 3), axis = plane / 2    (x, y, z)
        #   dir  on [0 .. 1), dir  = plane % 2    (-1, +1)
        #
        # how does this map to vertices? 
        #   the vertices for a given plane are: all x / y / z values on [-1, +1],
        #   where the plane's axis = that direction
        #
        # so the +x plane is given by the 4 vertices
        #    (+x, -y, -z), (+x, +y, -z), (+x, -y, +z), (+x, +y, +z)
        # with x, y, z = 1, 1, 1
        #
        # assuming that vertices are given by the indices [0 .. 8), ie
        #   000, 001, 010, 011, 100, 101, 110, 111
        #

        def gen_planar_vertex_coords (plane):
            coords = [ -1.0, 1.0 ]
            axis, direction = plane // 2, plane % 2
            i_axis, j_axis = (axis + 1) % 3, (axis + 2) % 3
            for i in range(2):
                for j in range(2):
                    vertex = [0, 0, 0]
                    vertex[axis] = coords[direction]
                    vertex[i_axis] = coords[i]
                    vertex[j_axis] = coords[j]
                    yield vertex

        def graycode (n):
            enforce(n == 8, "graycode tbd, for now just hardcoded. %s != 8", n)
            return [
                0b000,
                0b001,
                0b010,
                0b011,
                0b010,
                0b110,
                0b111,
                0b101,
                0b100,
            ]

        def gen_verts ():
            def gen_coords ():
                for i in range(8):
                    bit = 1
                    while (bit & 7) != 0:
                        yield +1.0 if i & bit else -1.0
                        bit <<= 1
            return np.array(list(gen_coords())).reshape((8, 3))

        def gen_edges ():
            def gen_edge_pairs ():
                n = 8
                for i in range(8):
                    bit = 1
                    while (bit & 7):
                        if i & bit == 0:
                            yield i, i | bit
                        bit <<= 1
            return np.array(list(gen_edge_pairs())).reshape((12, 2))

        edges = gen_edges()
        print(edges.shape)
        print(edges)

        face_indices = [[] for i in range(6) ]
        for a, b in edges:
            for axis in range(3):
                if (a & (1 << axis)) == (b & (1 << axis)):
                    plane = axis * 2 + ((a >> axis) & 1)
                    if a not in face_indices[plane]:
                        face_indices[plane].append(a)
                    if b not in face_indices[plane]:
                        face_indices[plane].append(b)

        face_indices = np.array(face_indices)
        for i in range(face_indices.shape[0]):
            face_indices[i][3], face_indices[i][2] = face_indices[i][2], face_indices[i][3]

        print(face_indices.shape)
        print(face_indices)

        # def gen_verts ():
        #     coords = [ -1.0, 1.0 ]
        #     return [
        #         [ coords[i & 1], coords[(i >> 1) & 1], coords[(i >> 2) & 1] ]
        #         for i in graycode(8)
        #     ]

        def gen_planar_vertex_indices (plane):
            axis, direction = plane // 2, plane % 2
            if axis == 0:
                i_axis, j_axis = (axis + 1), (axis + 2)
            elif axis == 1:
                i_axis, j_axis = (axis - 1), (axis + 2)
            elif axis == 2:
                i_axis, j_axis = (axis - 2), (axis - 1)

            # i_axis, j_axis = (axis + 1) % 3, (axis + 2) % 3
            for i in range(2):
                for j in range(2):
                    yield (direction << axis) | (i << i_axis) | (j << j_axis)

        def gen_planar_normals (plane):
            direction_values = [ -1.0, 1.0 ]
            axis, direction = plane // 2, plane % 2
            vertex = [ 0, 0, 0 ]
            vertex[axis] = direction_values[direction]
            return vertex

        print("Cube Vertices:")
        for plane in range(6):
            print("\t%s"%list(gen_planar_vertex_coords(plane)))

        print("Cube Vertex coords:")
        for plane in range(6):
            print("\t%s"%list(gen_planar_vertex_indices(plane)))

        self.quad_verts = np.array(gen_verts())
        self.quad_faces = face_indices
        # self.quad_faces = np.array(list(map(list, map(gen_planar_vertex_indices, range(6)))))
        # self.quad_verts = np.array(list(map(list, map(gen_planar_vertex_coords, range(6))))).reshape(12, 3)
        self.quad_faces = np.array([
            [0, 4, 6, 2],
            [1, 3, 7, 5],
            [0, 1, 5, 4],
            [6, 7, 3, 2],
            [2, 3, 1, 0],
            [4, 5, 7, 6],
        ])
        self.quad_normals = np.array(list(map(list, map(gen_planar_normals, range(6)))))
        print(self.quad_faces.shape, self.quad_faces)
        print(self.quad_verts.shape, self.quad_verts)
        print(self.quad_normals.shape, self.quad_normals)
        self.vertex_normals = np.array([
            [-1, -1, -1],
            [1, -1, -1],
            [-1, 1, -1],
            [1, 1, -1],
            [-1, -1, 1],
            [1, -1, 1],
            [-1, 1, 1],
            [1, 1, 1],
        ])
        print(self.verts.shape)
        # print(self.faces.shape)
        print(self.vertex_normals.shape)

    def __str__ (self):
        def stringify (fmt, array, f):
            try:
                return [ fmt%tuple(f(array[i])) for i in range(array.shape[0]) ]
            except TypeError:
                print("Failed to stringify with fmt '%s', values with shape %s, first element shape = %s, values = %s"%(
                    fmt, array.shape, array[0].shape, array[0]
                ))
            return []

        obj_lines = stringify('v %f %f %f', self.quad_verts, lambda x: x)
        # obj_lines += stringify('vn %f %f %f', self.quad_normals)
        # obj_lines += stringify('f %d// %d// %d// %d//', self.quad_faces, lambda i: i + 1)
        # return '\n'.join(obj_lines)

        # obj_lines += [ 'v %f %f %f'%tuple(self.quad_verts[i]) for i in range(self.verts) ]
        self.generate_triangle_faces()
        obj_lines += stringify('f %d// %d// %d//', self.faces, lambda i: i + 1)
        # obj_lines += [ 'vn %f %f %f'%tuple(self.vertex_normals[i]) for i in range(self.vertex_normals.shape[0]) ]
        return '\n'.join(obj_lines)

    def generate_triangle_faces (self):
        faces = []
        for i in range(self.quad_faces.shape[0]):
            a, b, c, d = self.quad_faces[i]
            faces.append([ a, b, c ])
            faces.append([ c, d, a ]) 
        self.faces = np.array(faces)

    def subdivide (self, vertex_offsets):
        new_faces, new_normals = [], []
        new_verts = []
        for i in range(self.quad_faces.shape[0]):
            a, b, c, d = self.quad_faces[i]

            #     a - e - b
            #    /   /   /
            #   h - k - f
            #  /   /   /
            # d - g - c

            # subdivde
            e = len(self.quad_verts) + len(new_verts)
            new_verts.append((self.quad_verts[a] + self.quad_verts[b]) / 2)
            f = len(self.quad_verts) + len(new_verts)
            new_verts.append((self.quad_verts[b] + self.quad_verts[c]) / 2)
            g = len(self.quad_verts) + len(new_verts)
            new_verts.append((self.quad_verts[c] + self.quad_verts[d]) / 2)
            h = len(self.quad_verts) + len(new_verts)
            new_verts.append((self.quad_verts[d] + self.quad_verts[a]) / 2)
            k = len(self.quad_verts) + len(new_verts)
            new_verts.append((self.quad_verts[a] + self.quad_verts[b] + self.quad_verts[c] + self.quad_verts[d]) / 4)

            # apply offsets
            # TBD
            # self.quad_verts[k] += self.face_normals[i] * vertex_offsets.pop()

            # interpolate w/ normal from adjacent faces...
            # TBD
            # self.quad_verts[e] += 
            new_faces += [
                [ a, e, k, h ],
                [ e, b, f, k ],
                [ h, k, g, d ],
                [ k, f, c, g ],
            ]
            new_normals += [
                self.quad_normals[i],
                self.quad_normals[i],
                self.quad_normals[i],
                self.quad_normals[i],
            ]
        self.quad_verts = np.array(list(self.quad_verts) + new_verts)
        self.quad_faces, self.quad_normals = np.array(new_faces), np.array(new_normals)

    def subdivide_reconstruct (self, vertex_offsets):
        pass

def reconstruct_mesh (data):
    print(data.shape)
    enforce(type(data) == np.ndarray, "expected a numpy array, not %s", type(data))
    enforce(len(data.shape) == 1, "expected a 1d array, not shape %s"%(data.shape,))
    cube_mesh = CubeMesh(data[:8 * 3].reshape(8, 3))
    print(cube_mesh)
    cube_mesh.subdivide(None)
    cube_mesh.subdivide(None)
    cube_mesh.subdivide(None)
    cube_mesh.subdivide(None)
    print(cube_mesh)
    return str(cube_mesh)

def write_file (result, output_file):
    basedir, file = os.path.split(output_file)
    if not os.path.exists(basedir):
        os.makedirs(basedir)

    with open(output_file, 'w') as f:
        f.write(result)

def process_json_file (input_file, output_file):
    with open(input_file, 'r') as f:
        data = np.array(json.loads(f.read()))
        print(data.shape)
        result = reconstruct_mesh(data)
        write_file(result, output_file)

def enforce(condition, fmt, *args, exception=Exception):
    if not condition:
        raise exception(fmt % args)

if __name__ == '__main__':
    args = docopt(__doc__)

    # validate arguments
    class ArgumentParsingException(Exception):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

    enforce_arg = lambda cond, fmt, *args: enforce(cond, fmt, *args, exception=ArgumentParsingException)

    try:
        if args['<file>']:
            input_file = args['<file>']
            if args['json']:
                enforce(input_file.endswith(".json"), "%s is not a json file", input_file)
            enforce(os.path.exists(input_file), "%s does not exist", input_file)

    except ArgumentParsingException as e:
        print("Invalid argument: %s" % e)
        sys.exit(-1)

    if args['json']:
        basedir, file = os.path.split(input_file)
        process_json_file(input_file, os.path.join('reconstructed', file.replace('.json', '.obj')))
