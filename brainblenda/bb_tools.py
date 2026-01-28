#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 14 17:20:00 2026

@author: wst


visualisation tools for Snudda & Blender
requires Blender 4.0 > with a functional python installation

"""

import sys
import os
import bpy
import bmesh
import h5py

import mathutils
import math
import random
import numpy as np

from snudda.utils.snudda_path import snudda_parse_path
from snudda import SnuddaLoad


# default_colours = [(0.168, 0.416, 0.8), (0.8, 0.294, 0.134)]
# snr_colour = [52 / 255, 168 / 255, 224 / 255, 1]
# midline = 5691.66

### Allen import convention: axis_forward = 'Y', axis_up = 'Z'
### coord_conv = {'Mouselight': {'X':4, 'Y':3, 'Z':2}, 'Peng': {'X':2, 'Y':3, 'Z':4}}


def clear_scene(bg_colour=(1, 1, 1, 1), clipping=1e6, raw_colours=True):
    """
    clears all objects from active blender scene, and (optionallly) sets background colour.

    bg_colour (RGBA, optional): colour to set blender background to. Default: (1,1,1,1).
    clipping (float, opional): distance at which objects will be clipped in the default viewer. For Allen meshes a value of 1e6 is reasonable to keep all objects in view. Default: 1e6.
    raw_colours (bool, optional): If True, sets blender view transform to 'Raw', such that colours are not modified by blender (ie., RGB values will be preserved in render). Default: True.
    """

    if bpy.context.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)

    if bg_colour:
        assert len(bg_colour) == 4, "colour must RGBA"
        bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[
            0
        ].default_value = bg_colour

    for area in bpy.context.screen.areas:
        if area.type == "VIEW_3D":
            for space in area.spaces:
                if space.type == "VIEW_3D":
                    space.clip_end = clipping

    if raw_colours:
        bpy.context.scene.view_settings.view_transform = "Raw"

    return


def frame_selected():
    """
    frames selected objects in blender viewer.

    """

    for area in bpy.context.screen.areas:
        if area.type == "VIEW_3D":
            for region in area.regions:
                if region.type == "WINDOW":
                    override = {
                        "area": area,
                        "region": region,
                        "edit_object": bpy.context.edit_object,
                    }
                    with bpy.context.temp_override(**override):
                        bpy.ops.view3d.view_selected(use_all_regions=False)
                    break
            break

    return


def frame_all():
    """
    frames all objects in blender viewer.

    """

    for area in bpy.context.screen.areas:
        if area.type == "VIEW_3D":
            for region in area.regions:
                if region.type == "WINDOW":
                    override = {
                        "area": area,
                        "region": region,
                        "edit_object": bpy.context.edit_object,
                    }
                    with bpy.context.temp_override(**override):
                        bpy.ops.view3d.view_axis(type="TOP")
                        bpy.ops.view3d.view_all()
                    break
            break

    return


def set_coronal_view():
    """
    sets 3D view to coronal (following Allen mesh conventions)

    """
    for area in bpy.context.screen.areas:
        if area.type == "VIEW_3D":
            space = area.spaces.active
            region_3d = space.region_3d

            base_quat = mathutils.Quaternion((0.5, 0.5, 0.5, 0.5))
            roll_quat = mathutils.Quaternion((1.0, 0.0, 0.0), math.radians(90))
            region_3d.view_rotation = roll_quat @ base_quat
            region_3d.view_perspective = "ORTHO"
            break
    return


def bulletproof_name(name):
    """

    returns a new name if the name is already in use. This avoids conflicts with blender object operations.

    name (str): desired name.

    returns:
        name (str): desired name with a numeric suffix.

    """

    base = name
    i = 1

    while name in bpy.data.objects:
        name = f"{base}.{i:03d}"
        i += 1
    return name


def collect_meshes(name):
    """

    returns a Blender collection by name, or creates a new one if necessary.
    name (str): name of collection.

    returns:
        col (bpy.types.Collection)

    """
    if name in bpy.data.collections:
        return bpy.data.collections[name]
    col = bpy.data.collections.new(name)
    bpy.context.scene.collection.children.link(col)
    return col


def is_inside(p, obj):
    """
    checks if a point is inside a blender object.

    p (list of floats): 3D coordinate
    obj (bpy.types.Object): Blender mesh object

    returns:

        True if p is inside obj.

    """
    p = mathutils.Vector(p)

    result, closest, normal, _ = obj.closest_point_on_mesh(p)
    direction = p - closest

    return direction.dot(normal) < 0


def distance_to_mesh(p, obj):
    """
    calculates distance from a given point to the surface of a blender object.

    p (list of floats): 3D coordinate.
    obj (bpy.types.Object): blender mesh object.

    returns:

        distance (float): distance to mesh surface.
        nearest_point (mathutils.Vector): point on mesh surface nearest to specified point.

    """

    p = mathutils.Vector(p)

    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bm.transform(obj.matrix_world)

    bvh = mathutils.bvhtree.BVHTree.FromBMesh(bm)
    nearest_point, normal, index, distance = bvh.find_nearest(p)

    bm.free()

    return distance, nearest_point


##### not necessary but can provide significant speed ups for large arrays of coordinates
# from numba import jit
# @jit(nopython=True)
# def distance_numba(point1, point2):
#     dx = point2[0] - point1[0]
#     dy = point2[1] - point1[1]
#     dz = point2[2] - point1[2]
#     return math.sqrt(dx*dx + dy*dy + dz*dz)


def get_mesh_volume(obj, scale_f=1e-9):
    """
    calculates volume of blender object.

    obj (bpy.types.Object): blender mesh object
    scale_f (float): scaling factor for conversion to desired unit. Default: 1e-9 (converts Allen mesh volumes to mm^3).

    returns:
        volume (float): volume, mutlitplied by optional scaling factor.
    """

    me = obj.data

    bm = bmesh.new()
    bm.from_mesh(me)
    bm.transform(obj.matrix_world)
    bmesh.ops.triangulate(bm, faces=bm.faces)

    volume = 0
    for f in bm.faces:
        v1, v2, v3 = [v.co for v in f.verts[:3]]
        volume += v1.dot(v2.cross(v3)) / 6
    bm.free()

    return volume * scale_f


def random_RGBA(low=50, high=255, alpha=1):
    """
    generates a random RGBA. Use low/high limits to control colour darkness for contrast with scene (i.e., 0-100 for dark objects on white backgrounds).

    """

    return tuple(np.random.randint(low, high, size=3) / 255) + (alpha,)


def flip_mesh_across_midline(obj, midline=5691.66):
    """
    flip blender object across midline
    obj (bpy.types.Object): blender mesh object.
    midline: midline of brain. Default: 5691.66.

    """
    obj.scale.z *= -1
    if obj.parent:
        obj.parent.location = [
            obj.parent.location[0],
            obj.parent.location[1],
            2 * midline - obj.parent.location[2],
        ]

    else:
        obj.location = [obj.location[0], obj.location[1], 2 * midline - obj.location[2]]

    return


def bisect_object(obj, plane="sagittal", location=5691.66):
    """
    splits mesh objects at specified plane
    obj (bpy.types.Object): blender mesh object.
    plane (str): desired plane, accepts 'sagittal', 'coronal' or 'horizontal'. Follows conventions of Allen meshes.
    location (float): location along axis to bisect.

    """

    assert plane.lower() in [
        "sagittal",
        "coronal",
        "horizontal",
    ], f"Ensure plane is one of sagittal, coronal, or horizontal. Received: {plane}."

    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")

    plane_key = {"sagittal": 2, "coronal": 0, "horizontal": 1}

    plane_co = [0, 0, 0]
    plane_no = [0, 0, 0]

    plane_co[plane_key[plane.lower()]] = location
    plane_no[plane_key[plane.lower()]] = 1

    bpy.ops.mesh.bisect(
        plane_co=plane_co,
        plane_no=plane_no,
        use_fill=True,
        clear_inner=False,
        clear_outer=True,
        threshold=0.0001,
    )

    bpy.ops.object.mode_set(mode="OBJECT")

    return


def set_origin_to_center(obj):
    """
    sets origin to center of volume of object
    obj (bpy.types.Object): blender mesh object.

    returns:
        obj.location: (XYZ)
    """

    if bpy.context.active_object and bpy.context.active_object.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.origin_set(type="ORIGIN_CENTER_OF_VOLUME")

    return obj.location


def add_tracked_camera(
    target_name,
    rotate=True,
    coronal=True,
    x_res=2160,
    y_res=2160,
    altitude=0,
    cam_clip_end=1e6,
    cam_type="ORTHO",
    cam_scale=2e4,
):
    """
    adds a camera to the scene, tracked to the center of a specified object.

    target_name (str): name of object to track to.
    rotate (bool): whether the camera should rotate about the object.
    coronal (bool): if True, will set a coronal view (Allen convention).
    x_res, y_res (int): resolution of x and y axes of camera (px).
    altitude (float): altitude of camera angle w.r.t. horizontal plane, in degrees. Default: 0.
    cam_clip_end (float): clipping threshold of camera. Default: 1e6. 
    cam_type (str): camera type. Default: 'ORTHO'. 
    cam_scale (float): camera scaling (i.e., zoom) for orthogonal cameras. Default: 2e4. 
    """

    assert cam_type in ['PERSP', 'ORTHO', 'PANO', 'CUSTOM'], 'Check camera type!'
    
    x_res = round(x_res)
    y_res = round(y_res)

    target = bpy.data.objects[target_name]
    set_origin_to_center(target)
    frame_selected()

    if coronal:
        set_coronal_view()

    bpy.ops.object.empty_add(type="PLAIN_AXES", location=target.location)
    bpy.ops.object.camera_add()

    cam = bpy.data.objects["Camera"]
    bpy.context.scene.camera = cam

    for area in bpy.context.window.screen.areas:
        if area.type == "VIEW_3D":
            for region in area.regions:
                if region.type == "WINDOW":
                    with bpy.context.temp_override(
                        window=bpy.context.window,
                        screen=bpy.context.window.screen,
                        area=area,
                        region=region,
                        space_data=area.spaces.active,
                        scene=bpy.context.scene,
                    ):
                        bpy.ops.view3d.camera_to_view()
                    break
            break

    cam.data.clip_end = cam_clip_end
    cam.data.type = cam_type
    if cam_type == "ORTHO":
        cam.data.ortho_scale = cam_scale

    em = bpy.data.objects["Empty"]

    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.select_all(action="DESELECT")
    cam.select_set(True)
    em.select_set(True)
    bpy.context.view_layer.objects.active = em
    bpy.ops.object.parent_set(type="OBJECT", keep_transform=True)

    if rotate:
        em.rotation_euler = [0, 0, math.radians(altitude)]
        em.keyframe_insert(data_path="rotation_euler", frame=1)
        em.rotation_euler = [0, math.radians(360), math.radians(altitude)]
        em.keyframe_insert(data_path="rotation_euler", frame=360)

        bpy.data.scenes["Scene"].frame_end = 360

        for fcurve in em.animation_data.action.fcurves:
            for keyframe in fcurve.keyframe_points:
                keyframe.interpolation = "LINEAR"
    bpy.data.scenes["Scene"].render.resolution_x = x_res
    bpy.data.scenes["Scene"].render.resolution_y = y_res
    return


def animate_visibility(obj, frame, on=True):
    """
    animates the visibility of an object in both render and viewport.

    obj (bpy.types.Object): blender mesh object.
    frame (int): keyframe at which the object should change visibility
    on (bool): if True, the object will start invisibile and become visibile at frame. If False, the inverse will occur. Default: True.

    """

    bpy.context.scene.frame_set(1)
    obj.hide_render = on
    obj.keyframe_insert(data_path="hide_render", frame=1)
    obj.hide_viewport = on
    obj.keyframe_insert(data_path="hide_viewport", frame=1)

    bpy.context.scene.frame_set(frame)
    obj.hide_render = ~on
    obj.keyframe_insert(data_path="hide_render", frame=frame)
    obj.hide_viewport = ~on
    obj.keyframe_insert(data_path="hide_viewport", frame=frame)

    for fcurve in obj.animation_data.action.fcurves:
        for keyframe in fcurve.keyframe_points:
            keyframe.interpolation = "LINEAR"
    return


def build_from_swc(
    filepath,
    name=None,
    lx=0,
    ly=0,
    lz=0,
    coord_space=None,
    flip=False,
    midline=5691.66,
    rotate=False,
    rotating=False,
    draw_axon=False,
    fill_process_tips=False,
    merge_threshold=0.001,
    scale_f=1,
    colour=None,
    alpha=1,
):
    """
    renders a neuron in blender from an .swc file, using bezier curves.
    
    Inspired by https://github.com/Hjorthmedh/Snudda/blob/master/snudda/plotting/Blender/io_mesh_swc/operator_swc_import.py

    filepath (str): path to SWC file of neuron to be rendered
    name (str, optional): neuron name, will be passed to object name in Blender. Default: None.
    lx, ly, lz (float, optional): soma offsets in XYZ. Default: 0.
    coord_space (dict, optional): specified coordinate space if different from default. Default: {'X':2, 'Y':3,'Z':4}.
    flip (bool, optional): if True, flips the neuron about the midline.
    midline (float, optional): midline of brain (Allen CCF). Default 5691.66.
    rotate (bool, optional): Applies random rotatation if true. Will be overriden by 'rotating'.
    rotating (bool, optional): Animates rotation about Z, at 1° per frame. Will override 'rotate'.
    draw_axon (bool, optional): If False, axons (ie., SWC type 2) will not be rendered.
    fill_process_tips (bool optional): If True, fills neurite tips with spheres for aesthetics. Warning: slow for large numbers of neurons.
    merge_threshold (float, optional): Distance within which vertices will be merged. Higher values reduce the complexity of the resulting object. Default: 0.001.
    scale_f (float, optional): Scaling factor for neuron size. Default: 1 (ie., 1:1).
    colour (RGBA, optional): Colour to render neuron in. If not specfied a random colour will be generated.
    alpha (float, optional): Alpha value (transparency) for neuron material. Must be in the interval [0,1]. Default: 1.

    """

    f = open(filepath)
    lines = f.readlines()
    f.close()

    x = 0
    while lines[x][0] == "#":
        x += 1
    if coord_space:
        coord_space = coord_space
    else:
        coord_space = {"X": 2, "Y": 3, "Z": 4}

    data = lines[x].strip().split()
    somaID = int(data[0])
    somaType = float(data[1])
    somaX = float(data[coord_space["X"]]) + lx
    somaY = float(data[coord_space["Y"]]) + ly
    somaZ = float(data[coord_space["Z"]]) + lz
    somaR = float(data[5])
    somaParent = int(data[6])

    neuron = {somaID: [somaType, somaX, somaY, somaZ, somaR, somaParent]}

    x += 1

    for l in lines[x:]:
        data = l.strip().split()
        compID = int(data[0])
        compType = float(data[1])
        compX = float(data[coord_space["X"]]) + lx
        compY = float(data[coord_space["Y"]]) + ly
        compZ = float(data[coord_space["Z"]]) + lz
        compR = float(data[5])
        compParent = int(data[6])
        neuron[compID] = [
            compType,
            compX - somaX,
            compY - somaY,
            compZ - somaZ,
            compR,
            compParent,
        ]

    if bpy.context.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")

    bpy.ops.object.empty_add(
        type="ARROWS",
        location=(
            neuron[1][1] / scale_f,
            neuron[1][2] / scale_f,
            neuron[1][3] / scale_f,
        ),
        rotation=(0, 0, 0),
    )
    em = bpy.context.selected_objects[0]

    if not name:
        name = "neuron"

    em.name = bulletproof_name(name)

    if rotating:
        em.rotation_euler = [0, 0, 0]
        em.keyframe_insert(data_path="rotation_euler", frame=1)
        em.rotation_euler = [0, 0, math.radians(360)]
        em.keyframe_insert(data_path="rotation_euler", frame=360)

        bpy.data.scenes["Scene"].frame_end = 360

        for fcurve in em.animation_data.action.fcurves:
            for keyframe in fcurve.keyframe_points:
                keyframe.interpolation = "LINEAR"

    elif rotate:

        em.rotation_euler = [0, random.randrange(0, 30), random.randrange(0, 360)]

    if not colour:
        colour = random_RGBA()
    else:
        assert len(colour) == 4, "colour must RGBA"

    material = bpy.data.materials.new(name=str(name) + "_mat")
    material.use_nodes = True
    pbsdf_node = material.node_tree.nodes["Principled BSDF"]
    pbsdf_node.inputs["Base Color"].default_value = colour
    pbsdf_node.inputs["Metallic"].default_value = 0
    pbsdf_node.inputs["Roughness"].default_value = 1
    pbsdf_node.inputs["Specular IOR Level"].default_value = 0.05
    pbsdf_node.inputs["Sheen Weight"].default_value = 0
    pbsdf_node.inputs["Alpha"].default_value = alpha
    material.blend_method = "BLEND"

    last = 0

    for key, value in neuron.items():
        if value[0] == 1:  # soma
            somaRadie = somaR
            bpy.ops.mesh.primitive_uv_sphere_add(
                location=(0 / scale_f, 0 / scale_f, 0 / scale_f),
                radius=somaRadie / scale_f,
                segments=64,
                ring_count=64,
            )
            somaObj = bpy.context.selected_objects[0]
            somaObj.name = "Soma"
            somaObj.parent = em
            somaObj.data.materials.append(material)
            last = -10

        if value[-1] == -1:
            continue
        if value[0] == 10:
            continue
        if value[0] == 5:
            continue
        if draw_axon == False:
            if value[0] == 2:
                continue

        if value[-1] != last:
            if fill_process_tips:
                if last != -10:
                    bpy.ops.mesh.primitive_uv_sphere_add(
                        radius=p.radius,
                        location=p.co,
                        scale=(1, 1, 1),
                        segments=32,
                        ring_count=16,
                    )
                    obj = bpy.context.selected_objects[0]
                    obj.data.polygons.foreach_set(
                        "use_smooth", [True] * len(obj.data.polygons)
                    )
                    obj.active_material = material
                    obj.parent = em

            # trace the origins
            tracer = bpy.data.curves.new("tracer", "CURVE")
            tracer.dimensions = "3D"
            spline = tracer.splines.new("BEZIER")
            curve = bpy.data.objects.new("curve", tracer)
            curve.data.use_fill_caps = False
            curve.data.materials.append(material)

            bpy.context.scene.collection.objects.link(curve)

            # render ready curve
            tracer.resolution_u = 12
            tracer.bevel_resolution = 12
            tracer.fill_mode = "FULL"
            tracer.bevel_depth = 1.0

            # move nodes to objects
            p = spline.bezier_points[0]
            if neuron[value[-1]][1] == somaX:
                xco = 0
            else:
                xco = neuron[value[-1]][1]
            if neuron[value[-1]][2] == somaY:
                yco = 0
            else:
                yco = neuron[value[-1]][2]
            if neuron[value[-1]][3] == somaZ:
                zco = 0
            else:
                zco = neuron[value[-1]][3]

            p.co = [xco, yco, zco]
            p.radius = neuron[value[-1]][4]
            p.handle_right_type = "VECTOR"
            p.handle_left_type = "VECTOR"

            if last > 0:
                spline.bezier_points.add(1)
                p = spline.bezier_points[-1]
                p.co = [value[1] / scale_f, value[2] / scale_f, value[3] / scale_f]
                p.radius = value[4] / scale_f
                p.handle_right_type = "VECTOR"
                p.handle_left_type = "VECTOR"

            curve.parent = em

        # continue the last bezier curve
        if value[-1] == last:
            spline.bezier_points.add(1)
            p = spline.bezier_points[-1]
            p.co = [value[1] / scale_f, value[2] / scale_f, value[3] / scale_f]
            p.radius = value[4] / scale_f
            p.handle_right_type = "VECTOR"
            p.handle_left_type = "VECTOR"

        last = key

    ##fill in end of processes, can be slow
    if fill_process_tips:
        bpy.ops.mesh.primitive_uv_sphere_add(
            radius=p.radius, location=p.co, scale=(1, 1, 1), segments=32, ring_count=16
        )
        obj = bpy.context.selected_objects[0]
        obj.data.polygons.foreach_set("use_smooth", [True] * len(obj.data.polygons))
        obj.active_material = material
        obj.parent = em

    ##merge all objects into single object
    for obj in bpy.data.objects[name].children:
        if "curve" in obj.name:
            obj.select_set(True)
            bpy.context.view_layer.objects.active = obj
            bpy.ops.object.convert(target="MESH", keep_original=False)
        elif "Sphere" in obj.name:
            obj.select_set(True)
            bpy.context.view_layer.objects.active = obj
        elif "Soma" in obj.name:
            obj.select_set(True)
            bpy.context.view_layer.objects.active = obj

    soma = next(
        (obj for obj in bpy.data.objects[name].children if "Soma" in obj.name), None
    )

    bpy.ops.object.select_all(action="DESELECT")

    for obj in bpy.data.objects[name].children:
        obj.select_set(True)

    bpy.context.view_layer.objects.active = soma

    if bpy.context.active_object.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")

    bpy.ops.object.join()

    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")

    bpy.ops.mesh.remove_doubles(
        threshold=merge_threshold,
        use_unselected=False,
        use_sharp_edge_from_normals=False,
    )
    bpy.ops.object.mode_set(mode="OBJECT")
    joined_obj = bpy.context.active_object
    joined_obj.name = name
    joined_obj.data.use_auto_smooth = True
    joined_obj.data.auto_smooth_angle = math.pi / 2

    if flip:
        flip_mesh_across_midline(joined_obj, midline)

    return joined_obj


def import_allen_mesh(
    filepath, forward_axis="Y", up_axis="Z", colour=None, alpha=1, bf_cull=False
):
    """
    imports .obj files from Allen SDK to blender.

    filepath (str): path to .obj file
    forward_axis, up_axis (str, optional): 'X', 'Y', 'Z', axes for blender orientation
    colour (RGBA, optional): colour to render mesh in. If not specfied a random colour will be generated.
    alpha (float, optional): alpha value (transparency) for neuron material. Must be in the interval [0,1]. Default: 1.

    returns:
        obj (bpy.types.Object): blender mesh object of Allen region.
    """

    assert filepath.endswith(".obj"), "Expected .obj file."

    if not colour:
        colour = random_RGBA()
    else:
        assert len(colour) == 4, "colour must RGBA"

    bpy.ops.wm.obj_import(filepath=filepath, forward_axis=forward_axis, up_axis=up_axis)
    material = bpy.data.materials.new(name="mat")
    material.use_nodes = True
    pbsdf_node = material.node_tree.nodes["Principled BSDF"]
    pbsdf_node.inputs["Base Color"].default_value = colour
    pbsdf_node.inputs["Metallic"].default_value = 0
    pbsdf_node.inputs["Roughness"].default_value = 0
    pbsdf_node.inputs["Specular IOR Level"].default_value = 0
    pbsdf_node.inputs["Alpha"].default_value = alpha
    material.blend_method = "BLEND"
    material.use_backface_culling = bf_cull
    mesh = bpy.context.selected_objects[0]
    mesh.active_material = material

    return mesh


def spherical_sample(n, ndim=3):
    """
    returns n randdom points on the surface of a unit sphere.

    n (int): number of points
    """

    vec = np.random.randn(ndim, n)
    vec /= np.linalg.norm(vec, axis=0)
    return vec.T


def mesh_from_coordinates(coordinates, name=None, reference_object=None, scale_f=1):
    """
    creates a blender object from a set of coordinates, placing a reference object at each coordinate.
    Helpful for mapping out soma or synapse positions.

    coordinates (Nx3 array): coordinates of interest.
    reference_object (blender object, optional): specific blender object to place at each coordinate. Object should be centered at [0,0,0]. If None, a sphere will be generated.
    scale_f (float, optional): scaling factor for coordinates to blender space. Default: 1e6.

    returns:
        obj (bpy.types.Object): blender mesh object.
    """

    if not name:
        name = "coordinates"

    mesh = bpy.data.meshes.new("coordinates")
    mesh_object = bpy.data.objects.new("coordinates", mesh)
    mesh_object.name = f"{name}_temp"

    # create mesh from coords
    mesh.from_pydata(coordinates * scale_f, [], [])
    mesh.update(calc_edges=True)

    bpy.context.scene.cursor.location = [0, 0, 0]
    mesh_object.location = bpy.context.scene.cursor.location
    bpy.data.collections["Collection"].objects.link(mesh_object)

    if not reference_object:
        material = bpy.data.materials.new(name="mat")
        material.use_nodes = True
        pbsdf_node = material.node_tree.nodes["Principled BSDF"]
        pbsdf_node.inputs["Base Color"].default_value = (0, 0, 0, 1)

        bpy.ops.mesh.primitive_uv_sphere_add(
            location=[0, 0, 0], radius=10, segments=16, ring_count=16
        )
        reference_object = bpy.context.selected_objects[0]
        reference_object.data.materials.append(material)

    reference_object.parent = mesh_object
    mesh_object.instance_type = "VERTS"
    bpy.ops.object.select_all(action="DESELECT")

    mesh_object.select_set(True)
    before = set(bpy.data.objects)
    bpy.ops.object.duplicates_make_real()
    after = set(bpy.data.objects)
    realised = list(after - before)

    for r in realised:
        r.select_set(True)
    bpy.context.view_layer.objects.active = realised[0]
    bpy.ops.object.join()
    joined_obj = bpy.context.selected_objects[0]
    joined_obj.name = bulletproof_name(name)

    bpy.data.objects.remove(mesh_object, do_unlink=True)
    bpy.data.objects.remove(reference_object, do_unlink=True)

    frame_selected()

    return joined_obj


def draw_somas(
    soma_coordinates, name=None, colour=None, radius=7, res=1, alpha=1, scale_f=1e6
):
    """
    draws somas as spheres.

    soma_coordinates (Nx3 list of floats): coordinates of soma locations.
    name (str, optional): name of soma group for blender.
    colour (RGBA, optional): colour to draw somas in.
    radius (float, optional): radius of somas. Default: 7.
    res (float, optional): resolution of soma spheres. Higher resolution can be slow. Default: 1.
    alpha (float, optional): alpha value (transparency) for material. Must be in the interval [0,1]. Default: 1.
    scale_f (float, optional): scaling factor for coordinates to blender space. Default: 1e6.

    returns:
        obj (bpy.types.Object): blender mesh object of somas.
    """

    if not colour:
        colour = random_RGBA()
    else:
        assert len(colour) == 4, "colour must RGBA"

    material = bpy.data.materials.new(name="soma_mat")
    material.use_nodes = True
    pbsdf_node = material.node_tree.nodes["Principled BSDF"]
    pbsdf_node.inputs["Base Color"].default_value = colour
    pbsdf_node.inputs["Alpha"].default_value = alpha
    bpy.ops.mesh.primitive_uv_sphere_add(
        location=[0, 0, 0],
        radius=radius,
        segments=int(8 * res),
        ring_count=int(8 * res),
    )
    reference_sphere = bpy.context.selected_objects[0]
    reference_sphere.data.materials.append(material)

    mesh_object = mesh_from_coordinates(
        soma_coordinates, name=name, reference_object=reference_sphere, scale_f=scale_f
    )

    return mesh_object


def draw_hypervoxels(
    hypervoxel_coords,
    hypervoxel_side_length=300.0,
    colour=(0, 0, 0, 1),
    alpha=1,
    thickness=30,
    scale_f=1e6,
):
    """
    draw hypervoxels as skeleton cubes.
    OBS: Snudda hypervoxel coordinates are available through SnuddaDetect, but, by default, are not written to disk.

    hypervoxel_coords (Nx3 list of floats): coordinates of hypervoxel centers.
    hypervoxel_side_length (float): size of hypervoxels.
    colour (RGBA, optional): Colour to render hypervoxels in. If not specfied a random colour will be generated.
    alpha (float, optional): alpha value (transparency) for material. Must be in the interval [0,1]. Default: 1.
    thickness (float, optional): Thickness of wireframe.
    scale_f (float, optional): scaling factor for coordinates to blender space. Default: 1e6.

    returns:
        obj (bpy.types.Object): blender mesh object of hypervoxels.
    """

    center = [hypervoxel_side_length / 2] * 3
    bpy.ops.mesh.primitive_cube_add(location=center, size=hypervoxel_side_length)

    material = bpy.data.materials.new(name="vox_mat")
    material.use_nodes = True
    pbsdf_node = material.node_tree.nodes["Principled BSDF"]
    pbsdf_node.inputs["Base Color"].default_value = colour
    pbsdf_node.inputs["Metallic"].default_value = 0
    pbsdf_node.inputs["Roughness"].default_value = 1
    pbsdf_node.inputs["Specular IOR Level"].default_value = 0
    pbsdf_node.inputs["Alpha"].default_value = alpha
    material.blend_method = "BLEND"

    reference_cube = bpy.context.selected_objects[0]
    reference_cube.data.materials.append(material)
    reference_cube.name = "hypervoxel"

    mod = reference_cube.modifiers.new(name="wf", type="WIREFRAME")
    mod.thickness = thickness
    mod.use_replace = True
    bpy.ops.object.modifier_apply(modifier="wf")

    hypervoxel_coords = np.array(hypervoxel_coords) * scale_f
    mesh = mesh_from_coordinates(
        coordinates=hypervoxel_coords,
        name="hypervoxels",
        reference_object=reference_cube,
    )

    return mesh


##################################
#### Snudda specfic functions ####
##################################

### Note: tested with Snudda SNr branch only. For virtual synapses: synapse location must be saved in input hdf5 file.
### OBS: this is not default snudda behaviour.
### Snudda uses microns at some points, and meters at others. Adjust scale_f depending on use case.


def draw_neuron_from_snudda(
    neurons, neuron_id, snudda_data, name=None, colour=None, scale_f=1e6
):
    """ "
    draws specified neuron from snudda network.
    neurons (list of dicts): data from snudda network: ie., sl.data['neurons'].
    neuron_id (int): id of neuron to draw.
    name (str, optional): name of neuron for blender.
    colour (RGBA, optional): Colour to render neuron in. If not specfied a random colour will be generated.
    scale_f (float, optional): Scaling factor for coordinates to blender space. Default: 1e6.

    returns:
        obj (bpy.types.Object): blender mesh object of neuron.
    """

    if not colour:
        colour = random_RGBA()
    else:
        assert len(colour) == 4, "colour must RGBA"

    position = neurons[neuron_id]["position"] * scale_f
    e_rot = mathutils.Matrix(neurons[neuron_id]["rotation"].reshape(3, 3)).to_euler()

    build_from_swc(
        snudda_parse_path(neurons[neuron_id]["morphology"], snudda_data),
        name=name,
        lx=position[0],
        ly=position[1],
        lz=position[2],
        colour=colour,
    )

    obj = bpy.context.selected_objects[0]
    obj.rotation_euler = e_rot

    return obj


def draw_postsynaptic(
    network_path,
    post_id,
    snudda_data,
    draw_presynaptic_partners=True,
    draw_synapses=True,
    scale_f=1e6,
    post_colour=(239 / 255, 42 / 255, 126 / 255, 1.0),
    pre_colour=None,
    synapse_colour=None,
    synapse_radius=5,
    match_synapses=False,
):
    """
    draws postsynaptic neuron with (optionally) presynaptic partners and/or synapse locations.

    network_path (str): path to snudda network-synapses.hdf5 file.
    post_id (int): id of neuron to analyse and draw.
    snudda_data (str): path to snudda data.
    draw_presynaptic_partners (bool, optional): If True, presynaptic neurons will also be drawn.
    draw_synapses (bool, optional): If True, afferent synapses of post_id will be drawn.
    scale_f (float, optional): Scaling factor for coordinates to blender space. Default: 1e6.
    pre_colour (RGBA, optional): Colour to draw postsynaptic neuron in. Default: (239/255, 42/255, 126/255, 1.0) - kind of a red-pink colour. I am colourblind though, so don't trust that...
    post_colour (RGBA, optional): Colour to draw presynaptic neurons in.
    synapse_colour (RGBA, optional): Colour to draw synapses in.
    synapse_radius (float, optional): Radius of synapses. Default: 5.


    returns:

    pre_ids (list): list of presynaptic partner ids.
    post_synapse_coords (list): list of afferent synapse coordinates.

    """

    sl = SnuddaLoad(os.path.join(network_path, "network-synapses.hdf5"))
    neurons = sl.data["neurons"]
    synapses = sl.data["synapses"]
    synapse_coords = sl.data["synapse_coords"]
    post_synapses = (synapses[:, 1] == post_id).astype(bool)
    pre_ids = list(set(synapses[post_synapses, 0]))
    pre_synapse_coords = synapse_coords[post_synapses]

    draw_neuron_from_snudda(
        neurons,
        post_id,
        snudda_data,
        name="postsynaptic" + str(post_id),
        colour=post_colour,
    )

    if not match_synapses:

        if draw_presynaptic_partners:
            for pre_id in pre_ids:
                draw_neuron_from_snudda(
                    neurons,
                    pre_id,
                    snudda_data,
                    name="presynaptic" + str(pre_id),
                    colour=pre_colour,
                )

        if draw_synapses:

            if not synapse_colour:
                synapse_colour = random_RGBA()
            else:
                assert len(synapse_colour) == 4, "colour must RGBA"

            material = bpy.data.materials.new(name="syn_mat")
            material.use_nodes = True
            pbsdf_node = material.node_tree.nodes["Principled BSDF"]
            pbsdf_node.inputs["Base Color"].default_value = synapse_colour
            bpy.ops.mesh.primitive_uv_sphere_add(
                location=[0, 0, 0], radius=synapse_radius, segments=16, ring_count=16
            )
            reference_sphere = bpy.context.selected_objects[0]
            reference_sphere.data.materials.append(material)

            mesh_from_coordinates(
                pre_synapse_coords,
                name="synapses",
                reference_object=reference_sphere,
                scale_f=scale_f,
            )

    else:

        for pre_id in pre_ids:

            matched_synapses = (
                (synapses[:, 0] == pre_id) & (synapses[:, 1] == post_id)
            ).astype(bool)

            colour = random_RGBA(low=0, high=255)
            draw_neuron_from_snudda(
                neurons,
                pre_id,
                snudda_data,
                name="presynaptic" + str(pre_id),
                colour=colour,
            )

            material = bpy.data.materials.new(name="syn_mat")
            material.use_nodes = True
            pbsdf_node = material.node_tree.nodes["Principled BSDF"]
            pbsdf_node.inputs["Base Color"].default_value = colour
            bpy.ops.mesh.primitive_uv_sphere_add(
                location=[0, 0, 0], radius=synapse_radius, segments=16, ring_count=16
            )
            reference_sphere = bpy.context.selected_objects[0]
            reference_sphere.data.materials.append(material)

            mesh_from_coordinates(
                synapse_coords[matched_synapses],
                name="synapses" + str(pre_id),
                reference_object=reference_sphere,
                scale_f=scale_f,
            )

    return pre_ids, pre_synapse_coords


def draw_presynaptic(
    network_path,
    pre_id,
    snudda_data,
    draw_postsynaptic_partners=True,
    draw_synapses=True,
    scale_f=1e6,
    pre_colour=(239 / 255, 42 / 255, 126 / 255, 1.0),
    post_colour=None,
    synapse_colour=None,
    synapse_radius=5,
):
    """
    draws presynaptic neuron with (optionally) postsynaptic partners and/or synapse locations.

    network_path (str): path to snudda network-synapses.hdf5 file.
    pre_id (int): id of neuron to draw.
    snudda_data (str): path to snudda data.
    draw_postsynaptic_partners (bool, optional): If True, postsynaptic neurons will also be drawn.
    draw_synapses (bool, optional): If True, efferent synapses of pre_id will be drawn.
    scale_f (float, optional): Scaling factor for coordinates to blender space. Default: 1e6.
    pre_colour (RGBA, optional): Colour to draw presynaptic neuron in. Default: (239/255, 42/255, 126/255, 1.0) - kind of a red-pink colour. I am colourblind though, so don't trust that...
    post_colour (RGBA, optional): Colour to draw postsynaptic neurons in.
    synapse_colour (RGBA, optional): Colour to draw synapses in.
    synapse_radius (float, optional): Radius of synapses. Default: 5.

    returns:

    post_ids (list): list of postsynaptic partner ids.
    pre_synapse_coords (list): list of efferent synapse coordinates.

    """

    sl = SnuddaLoad(os.path.join(network_path, "network-synapses.hdf5"))
    neurons = sl.data["neurons"]
    synapses = sl.data["synapses"]
    synapse_coords = sl.data["synapse_coords"]

    post_synapses = (synapses[:, 0] == pre_id).astype(bool)
    post_ids = list(set(synapses[post_synapses, 1]))
    pre_synapse_coords = synapse_coords[post_synapses]

    draw_neuron_from_snudda(
        neurons,
        pre_id,
        snudda_data,
        name="presynaptic" + str(pre_id),
        colour=pre_colour,
    )

    if draw_postsynaptic_partners:
        for post_id in post_ids:
            draw_neuron_from_snudda(
                neurons,
                post_id,
                snudda_data,
                name="postynaptic" + str(post_id),
                colour=post_colour,
            )

    if draw_synapses:
        if not synapse_colour:
            synapse_colour = pre_colour
        else:
            assert len(synapse_colour) == 4, "colour must RGBA"

        material = bpy.data.materials.new(name="syn_mat")
        material.use_nodes = True
        pbsdf_node = material.node_tree.nodes["Principled BSDF"]
        pbsdf_node.inputs["Base Color"].default_value = synapse_colour
        bpy.ops.mesh.primitive_uv_sphere_add(
            location=[0, 0, 0], radius=synapse_radius, segments=16, ring_count=16
        )
        reference_sphere = bpy.context.selected_objects[0]
        reference_sphere.data.materials.append(material)

        mesh_from_coordinates(
            pre_synapse_coords,
            name="synapses",
            reference_object=reference_sphere,
            scale_f=scale_f,
        )

    return post_ids, pre_synapse_coords


def add_virtual_input_locations(
    coords,
    neuron,
    location=[0, 0, 0],
    rotation=[0, 0, 0],
    name=None,
    scale_f=1,
    soma_radius=10,
    colour=None,
    radius=4,
    res=1,
    jitter_amount=None,
    timing=None,
):
    """
    renders synapses at input locations defined in snudda input file (ie., virtual inputs).

    coords (Nx3 array): input locations.
    neuron (snudda neuron class, optional): postsynaptic neuron. Overrides location and rotation if passed.
    location (1x3 array, optional): location to center coordinates.
    rotation (1x3 array, optional): rotation of coordinates in Euler format (XYZ).
    name (str, optional): name of synapse group for blender.
    scale_f (float, optional): Scaling factor for coordinates to blender space. Default: 1 (ie., 1:1).
    soma_radius (float, optional): Radius of soma. Synapses within this distance are randomly offset to the soma surface.
    colour (RGBA, optional): Colour to render synapses in. If not specfied a random colour will be generated.
    radius (float, optional): Radius for synapse representaiton.
    res (float, optional): Resolution of synapse spheres. Higher resolution can be slow. Default: 1.
    jitter_amount (float, optional): Jitter to be applied to synapse coordinates. Offsets synapses from dendrites for aesthetic purposes.
    timing (int, optional): If specified, synapses will be animated to appear (ie., rain in) at the given time (keyframe). Can become very slow, as each synapse will be rendered individually.

    """

    coords = np.array(coords) * scale_f

    distances = np.linalg.norm(coords, axis=1)
    inside_soma = distances < soma_radius
    coords[inside_soma] = soma_radius * spherical_sample(np.sum(inside_soma))

    if jitter_amount:
        jitter = np.random.uniform(-jitter_amount, jitter_amount, coords.shape)
        coords += jitter

    if not colour:
        colour = random_RGBA()
    else:
        assert len(colour) == 4, "colour must RGBA"

    material = bpy.data.materials.new(name=str(name) + "_mat")
    material.use_nodes = True
    pbsdf_node = material.node_tree.nodes["Principled BSDF"]
    pbsdf_node.inputs["Metallic"].default_value = 0
    pbsdf_node.inputs["Roughness"].default_value = 0.5
    pbsdf_node.inputs["Base Color"].default_value = colour

    if neuron:
        rotation = mathutils.Matrix(neuron["rotation"].reshape(3, 3)).to_euler()
        location = neuron["position"] * scale_f

    if timing:
        for c in coords:
            bpy.ops.mesh.primitive_uv_sphere_add(
                location=[0, 0, 0],
                radius=radius,
                segments=int(8 * res),
                ring_count=int(8 * res),
            )
            sphere = bpy.context.selected_objects[0]
            sphere.data.materials.append(material)
            rotated_coord = rotation.to_matrix() @ mathutils.Vector(c)
            sphere.location = [0, 0, 0]
            sphere.keyframe_insert(
                data_path="location", frame=1, options={"INSERTKEY_NEEDED"}
            )
            sphere.location = rotated_coord + mathutils.Vector(location)
            sphere.keyframe_insert(
                data_path="location",
                frame=random.randint(timing - 5, timing + 5),
                options={"INSERTKEY_NEEDED"},
            )

            if sphere.animation_data and sphere.animation_data.action:
                for fcurve in sphere.animation_data.action.fcurves:
                    if fcurve.data_path == "location":
                        for keyframe in fcurve.keyframe_points:
                            keyframe.interpolation = "BEZIER"
                            keyframe.easing = "EASE_OUT"
    else:
        bpy.ops.mesh.primitive_uv_sphere_add(
            location=[0, 0, 0],
            radius=radius,
            segments=int(8 * res),
            ring_count=int(8 * res),
        )
        reference_sphere = bpy.context.selected_objects[0]
        reference_sphere.data.materials.append(material)

        mesh = mesh_from_coordinates(coords, reference_object=reference_sphere)

        if neuron:
            rotation = mathutils.Matrix(neuron["rotation"].reshape(3, 3)).to_euler()
            location = neuron["position"] * scale_f

        mesh.rotation_euler = rotation
        mesh.location = location

    return


def virtual_synapse_coordinates(input_file, neuron_id, input_prefix, snudda_data):
    """
    returns coordinates of synapses for specified postsynapic neuron_ids. OBS: make sure you are saving locations when you generate input, otherwise there will be no locations to find...

    input_file (str): path to snudda input file (.hdf5).
    neuron_id (int or list of ints): postsynaptic neuron id(s).
    input_prefix (str or list of strs): prefix for input types of interest ('STN' for example).
    snudda_data (str): path to snudda data folder.

    returns:
    syn_dict (dict): nested dictionary. Top level keys: neuron ids. Lower level keys: input_prefix. Values: lists of coordinates.

    """

    if isinstance(neuron_id, int):
        neuron_id = [neuron_id]
    if isinstance(input_prefix, str):
        input_prefix = [input_prefix]

    input_data = h5py.File(snudda_parse_path(input_file, snudda_data), "r")
    syn_dict = {}
    for n_id in neuron_id:
        neuron_dict = {}
        n_id = str(n_id)
        if n_id in input_data["input"].keys():
            for i_p in input_prefix:
                syns = []
                for k in input_data["input"][n_id].keys():
                    if i_p in k:
                        syns.append(input_data["input"][n_id][k]["location"][()])
                neuron_dict[i_p] = syns
        syn_dict[n_id] = neuron_dict

    return syn_dict


##########################################################################################################################################################################

if __name__ == "__main__":
    print(
        "Do not run this file directly. Call functions from python within blender please!"
    )
    sys.exit(-1)
