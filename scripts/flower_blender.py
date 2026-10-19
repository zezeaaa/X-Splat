"""Render a Flower Blender scene to RGB images plus a NeRF-style transforms.json.

Driven by scripts/prepare_flower.py, which passes the render settings from
configs/preprocess/flower.yaml:

    blender <scene>.blend --background --python scripts/flower_blender.py -- <arguments>
"""
import argparse
import glob
import json
import math
import os
import re
import sys

try:
    import bpy
    from mathutils import Matrix, Vector
except ImportError:
    sys.exit("This script runs inside Blender: blender <scene>.blend --background --python flower_blender.py -- ...")

UP_AXES = ("+x", "-x", "+y", "-y", "+z", "-z")


def remove_backdrop():
    """The library scenes stand the flower on a 'Cube', which is neither rendered nor measured."""
    for obj in list(bpy.data.objects):
        if obj.type == "MESH" and obj.name.startswith("Cube"):
            print(f"Removed backdrop object: {obj.name}")
            bpy.data.objects.remove(obj, do_unlink=True)


def mtl_textures(mtl_path):
    """Material name -> map_Kd texture file, from a Wavefront .mtl."""
    textures, current = {}, None
    for line in open(mtl_path, encoding="utf-8", errors="ignore"):
        parts = line.split()
        if len(parts) >= 2 and parts[0] == "newmtl":
            current = parts[1]
        elif len(parts) >= 2 and parts[0] == "map_Kd" and current is not None:
            textures[current] = parts[1]
            current = None
    return textures


def apply_textures(mtl_path):
    """Feed each material's .mtl texture into the base colour of its Principled BSDF:
    the library's .blend files carry no textures themselves."""
    textures = mtl_textures(mtl_path)
    applied = 0
    for mat in bpy.data.materials:
        key = mat.name if mat.name in textures else re.sub(r"\.\d+$", "", mat.name)  # Blender's .001 suffix
        if key not in textures:
            continue
        mat.use_nodes = True  # gives the material a Principled BSDF
        principled = next(n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
        tex_node = mat.node_tree.nodes.new("ShaderNodeTexImage")
        tex_node.image = bpy.data.images.load(os.path.join(os.path.dirname(mtl_path), textures[key]), check_existing=True)
        mat.node_tree.links.new(tex_node.outputs["Color"], principled.inputs["Base Color"])
        applied += 1
    print(f"[MTL] Applied textures from {os.path.basename(mtl_path)} to {applied} materials.")


def scene_bounds():
    """Centre and orbit radius of the meshes, from their world-space bounding boxes."""
    corners = [obj.matrix_world @ Vector(v) for obj in bpy.data.objects if obj.type == "MESH" for v in obj.bound_box]
    lo = [min(c[i] for c in corners) for i in range(3)]
    hi = [max(c[i] for c in corners) for i in range(3)]
    center = [(lo[i] + hi[i]) / 2 for i in range(3)]
    radius = 0.5 * math.sqrt((hi[0] - lo[0]) ** 2 + (hi[1] - lo[1]) ** 2 + (hi[2] - lo[2]) ** 2)
    return center, max(radius * 2.0, 2.0)


def view_angles(num_views, elev_max_deg, elev_min_deg, n_rotations):
    """(azimuth, elevation) pairs on a spiral from (0, elev_max) to (2*pi*n_rotations, elev_min);
    sin(elevation) is stepped linearly, which keeps the cameras evenly spaced along it."""
    sin_max, sin_min = math.sin(math.radians(elev_max_deg)), math.sin(math.radians(elev_min_deg))
    views = []
    for i in range(num_views):
        t = i / float(num_views - 1)  # 0 at the top, 1 at the bottom
        sin_el = max(-1.0 + 1e-3, min(1.0 - 1e-3, sin_max + t * (sin_min - sin_max)))
        views.append(((2.0 * math.pi * n_rotations * t) % (2.0 * math.pi), math.asin(sin_el)))
    return views


def main():
    parser = argparse.ArgumentParser(description="Render the RGB views of a Flower Blender scene")
    parser.add_argument("--out_dir", required=True, help="Gets images/*.png and transforms.json.")
    parser.add_argument("--num_views", type=int, required=True)
    parser.add_argument("--n_rotations", type=float, required=True, help="Full azimuth turns of the spiral from top to bottom.")
    parser.add_argument("--elev_max", type=float, required=True, help="Elevation (degrees) the spiral starts at.")
    parser.add_argument("--elev_min", type=float, required=True, help="Elevation (degrees) the spiral ends at.")
    parser.add_argument("--up_axis", choices=UP_AXES, required=True, help="World direction of the bloom; elev > 0 is on that side.")
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--samples", type=int, required=True, help="Cycles samples per pixel.")
    parser.add_argument("--seed", type=int, required=True, help="Cycles sampling seed; fixes the render.")
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])

    blend_dir = os.path.dirname(os.path.abspath(bpy.data.filepath))
    remove_backdrop()
    apply_textures(sorted(glob.glob(os.path.join(blend_dir, "*.mtl")))[0])  # the library ships one per scene

    images_dir = os.path.join(os.path.abspath(args.out_dir), "images")
    os.makedirs(images_dir, exist_ok=True)

    scene = bpy.context.scene
    cam = next(o for o in bpy.data.objects if o.type == "CAMERA")
    scene.camera = cam
    scene.render.resolution_x, scene.render.resolution_y = args.width, args.height
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = "PNG"
    scene.render.engine = "CYCLES"
    scene.cycles.samples = args.samples
    scene.cycles.seed = args.seed  # fixed, or Cycles varies its sampling pattern per frame
    scene.cycles.use_animated_seed = False

    center, radius = scene_bounds()
    center_vec = Vector(center)
    up_sign = 1.0 if args.up_axis[0] == "+" else -1.0
    up_idx = "xyz".index(args.up_axis[1])
    orbit = [k for k in range(3) if k != up_idx]  # the two axes the azimuth sweeps
    up = Vector([0.0, 0.0, 0.0])
    up[up_idx] = up_sign

    frames = []
    for i, (azimuth, elevation) in enumerate(view_angles(args.num_views, args.elev_max, args.elev_min, args.n_rotations)):
        c, s = math.cos(elevation), math.sin(elevation)
        eye = [0.0, 0.0, 0.0]
        eye[orbit[0]] = center[orbit[0]] + radius * c * math.sin(azimuth)
        eye[orbit[1]] = center[orbit[1]] + radius * c * math.cos(azimuth)
        eye[up_idx] = center[up_idx] + radius * up_sign * s
        eye = Vector(eye)
        forward = (center_vec - eye).normalized()
        right = forward.cross(up).normalized()
        up_cam = right.cross(forward)
        c2w = Matrix((
            (right[0], up_cam[0], -forward[0], eye[0]),
            (right[1], up_cam[1], -forward[1], eye[1]),
            (right[2], up_cam[2], -forward[2], eye[2]),
            (0, 0, 0, 1),
        ))
        cam.matrix_world = c2w
        scene.render.filepath = os.path.join(images_dir, f"{i:04d}.png")
        bpy.ops.render.render(write_still=True)
        print(f"Rendered {scene.render.filepath}", flush=True)
        frames.append({"file_path": os.path.join("images", f"{i:04d}"), "transform_matrix": [list(row) for row in c2w]})

    transforms_path = os.path.join(os.path.dirname(images_dir), "transforms.json")
    with open(transforms_path, "w") as f:
        json.dump({"camera_angle_x": cam.data.angle_x, "frames": frames}, f, indent=2)
    print(f"Wrote {transforms_path}")


if __name__ == "__main__":
    main()
