import argparse
import random
import math
import os

import bpy
import cv2
from mathutils import Vector
import matplotlib.pyplot as plt
import numpy as np
import skimage


class BlenderRendering:
    def __init__(self, file_path: str, distance_radius: float, image_size: int):
        bpy.ops.wm.open_mainfile(filepath=file_path)
        self.armature = bpy.data.objects['Armature']
        self.camera = bpy.data.objects['Camera']
        self.light = bpy.data.objects['Light']
        self.target = bpy.data.objects['Target']
        self.radius = distance_radius
        self.image_size = image_size
        bpy.context.scene.render.resolution_x = image_size
        bpy.context.scene.render.resolution_y = image_size

    # Function to randomize finger rotations
    def randomize_fingers(self, mini=-0.25, maxi=0.05):
        bpy.context.view_layer.objects.active = self.armature
        bpy.ops.object.mode_set(mode='POSE')
        for bone in self.armature.pose.bones:
            if "controller" in bone.name.lower():  # Adjust this condition to match your finger bones naming
                new_location = Vector((random.uniform(mini, maxi), random.uniform(mini, maxi), random.uniform(mini, maxi)))
                bone.location = new_location

    # Function to randomize camera position
    def randomize_camera(self):
        self.camera.location = self.target.location + self.random_point_on_sphere(self.radius)
        # Add a track_to constraint to the camera
        if "Track To" not in self.camera.constraints:
            constraint = self.camera.constraints.new(type='TRACK_TO')
            constraint.target = self.target
            constraint.track_axis = 'TRACK_NEGATIVE_Z'
            constraint.up_axis = 'UP_Y'
        else:
            self.camera.constraints["Track To"].target = self.target

    # Function to synchronize light position and rotation with the camera
    def synchronize_light_with_camera(self):
        # Add or update Copy Location constraint
        if "Copy Location" not in self.light.constraints:
            loc_constraint = self.light.constraints.new(type='COPY_LOCATION')
            loc_constraint.target = self.camera
        else:
            self.light.constraints["Copy Location"].target = self.camera

        # Add or update Copy Rotation constraint
        if "Copy Rotation" not in self.light.constraints:
            rot_constraint = self.light.constraints.new(type='COPY_ROTATION')
            rot_constraint.target = self.camera
        else:
            self.light.constraints["Copy Rotation"].target = self.camera

    @staticmethod
    def random_point_on_sphere(radius: float):
        theta = random.uniform(0, 2*math.pi)
        phi = random.uniform(0, 0.7*math.pi)
        x = radius * math.sin(phi) * math.cos(theta)
        y = radius * math.sin(phi) * math.sin(theta)
        z = radius * math.cos(phi)
        return Vector((x, y, z))

    # Function to convert hex color to RGB
    @staticmethod
    def hex_to_rgb(hexa: str):
        hexa = hexa.lstrip('#')
        lv = len(hexa)
        return tuple(int(hexa[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))

    @staticmethod
    def change_color(image_path, target_color):
        desired_color = np.asarray(target_color, dtype=np.float64)
        # read image
        img = plt.imread(image_path)
        img = (img * 255).astype(np.uint8)
        # read face mask as grayscale and threshold to binary
        mask = img[:, :, 3]
        rgb_img = img[:, :, :3]
        # get average bgr color of face
        average_color = cv2.mean(rgb_img, mask=mask)[:3]
        # compute difference colors and make into an image the same size as input
        diff_color = desired_color - average_color
        diff_color = np.full_like(rgb_img, diff_color, dtype=np.uint8)
        # shift input image color
        output_img = (rgb_img + diff_color).clip(0, 255)
        # antialias mask, convert to float in range 0 to 1 and make 3-channels
        facemask = cv2.GaussianBlur(mask, (0, 0), sigmaX=3, sigmaY=3, borderType=cv2.BORDER_DEFAULT)
        facemask = skimage.exposure.rescale_intensity(
            facemask, in_range=(100, 150), out_range=(0, 1)
        ).astype(np.float32)
        facemask = cv2.merge([facemask, facemask, facemask])

        # combine img and new_img using mask
        result = (rgb_img * (1 - facemask) + output_img * facemask)
        result = result.clip(0, 255).astype(np.uint8)
        result = np.concatenate([result, np.expand_dims(mask, -1)], -1)

        plt.imsave(image_path, result)

    # Function to randomize and apply a new color to the hand model using hex values
    def randomize_hand_color_with_hex(self, filename):
        hex_colors = [
            "#4B3932", "#5A453C", "#695046", "#785C50",
            "#87675A", "#967264", "#A57E6E", "#B48A78",
        ]
        random_hex = random.choice(hex_colors)
        random_color = self.hex_to_rgb(random_hex)
        self.change_color(filename, random_color)

    # Function to set the background color
    @staticmethod
    def set_background_color(color=(0., 0., 0., 1.), strength=1):
        world = bpy.context.scene.world
        if not world:
            world = bpy.data.worlds.new("World")
            bpy.context.scene.world = world
        world.use_nodes = True
        bg_node = world.node_tree.nodes.get('Background')
        if not bg_node:
            bg_node = world.node_tree.nodes.new('ShaderNodeBackground')
        bg_node.inputs['Color'].default_value = color
        bg_node.inputs['Strength'].default_value = strength

    @staticmethod
    def use_gpu_rendering():
        bpy.data.scenes[0].render.engine = "CYCLES"

        # Set the device_type
        bpy.context.preferences.addons[
            "cycles"
        ].preferences.compute_device_type = "CUDA"  # or "OPENCL"

        # Set the device and feature set
        bpy.context.scene.cycles.device = "GPU"

        # get_devices() to let Blender detects GPU device
        bpy.context.preferences.addons["cycles"].preferences.get_devices()
        for d in bpy.context.preferences.addons["cycles"].preferences.devices:
            d["use"] = 1  # Using all devices, include GPU and CPU

    def run(
            self,
            num_views: int,
            output_csv: str = 'labels.csv',
            output_image_root: str = 'images/render',
    ):
        self.use_gpu_rendering()

        self.set_background_color()

        for i in range(num_views):
            self.randomize_fingers()
            self.randomize_camera()
            self.synchronize_light_with_camera()
            bpy.context.view_layer.update()
            # Log bone positions
            filename = output_image_root + f'_{i}.png'
            # Render the scene
            bpy.context.scene.render.filepath = filename
            bpy.ops.render.render(write_still=True, use_viewport=False)
            # Randomize color
            self.randomize_hand_color_with_hex(filename)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--blend_file', type=str, default="custom_hand.blend", help='Blender input file')
    parser.add_argument('--n_generation', type=int, default=10, help='Number of generation')
    parser.add_argument('--camera_dist', type=float, default=12, help='Camera distance to target')
    parser.add_argument('--img_size', type=int, default=512, help='square image pixel size')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    args = parser.parse_args()
    random.seed(args.seed)
    rendering = BlenderRendering(file_path=args.blend_file, distance_radius=args.camera_dist, image_size=args.img_size)

    output_folder = str(args.blend_file).split('/')[-1].split('.')[0]
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    output_image_root = output_folder + '/images/render'
    rendering.run(args.n_generation, output_image_root=output_image_root)
