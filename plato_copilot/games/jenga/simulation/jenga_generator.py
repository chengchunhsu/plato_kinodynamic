import xml.etree.ElementTree as ET
from xml.dom import minidom
from easydict import EasyDict
import numpy as np

    
def get_geom(name, type='box', size='.15 .05 .015',  quat="1 0 0 0", pos='0 0 0', color="0.8 0.6 .4 1"):
    geom = ET.Element('geom', name=name, type=type, size=size, quat=quat, pos=pos, rgba=color)
    return geom


def array_to_string(array):
    return ' '.join(map(str, array))

def free_block(pos, name, odd_layer=True, color="0.8 0.6 .4 1", length=0.15, width=0.05, height=0.03):
    body = ET.Element('body', name=name, pos=pos)
    body.append(ET.Element("freejoint", name=f"{name}_freejoint"))
    if odd_layer:
        size1 = array_to_string([length, width, height / 2])
        size2 = array_to_string([length, width, height / 2])
    else:
        size1 = array_to_string([width, length, height / 2])
        size2 = array_to_string([width, length, height / 2])
    geom1 = get_geom(size=size1, color=color, name=f"{name}_geom_1")
    geom2 = get_geom(size=size2, pos=array_to_string([0, 0, height-0.001]), color=color, name=f"{name}_geom_2")
    body.append(geom1)
    body.append(geom2)
    return body

class JengaXMLGenerator:
    def __init__(self, debug=False):
        # Initialize the root element, assuming 'xml' is the root
        self.root = ET.Element('mujoco')
        self.debug = debug
        self.block_names = []
        self.odd_layer_block_names = []
        self.even_layer_block_names = []
    
    def add_default(self, joint_limited='false', geom_attributes=None):
        default = ET.Element('default')
        default.append(ET.Element('joint', limited=joint_limited))
        if geom_attributes is None:
            geom_attributes = {'group': "2", 'friction': '0.3 0.005 0.0001', 'density': "1",
                               'solref': "0.02 1", 'solimp': ".99 .99 0.005 0.9 2", 'material': "geom"}
        geom = ET.Element('geom', **geom_attributes)
        default.append(geom)
        self.root.append(default)

    def add_global_visual(self):
        visual = ET.Element('visual')
        headlight = ET.Element('headlight', diffuse="0.6 0.6 0.6", ambient="0.3 0.3 0.3", specular="0 0 0")
        rgba = ET.Element('rgba', haze="0.15 0.25 0.35 1")
        global_ = ET.Element('global', azimuth="40", elevation="-20")
        znear_map = ET.Element('map', znear="0.001")
        visual.append(headlight)
        visual.append(rgba)
        visual.append(global_)
        visual.append(znear_map)
        self.root.append(visual)

    def add_assets(self, material_attributes=None):
        assets = ET.Element('asset')
        if material_attributes is None:
            material_attributes = {'name': "geom", 'rgba': "0.8 0.6 .4 1"}
        material = ET.Element('material', **material_attributes)
        assets.append(material)

        skybox_texture = ET.Element('texture', type="skybox", builtin="gradient", rgb1="0.3 0.5 0.7", rgb2="0 0 0", width="512", height="3072")
        groundplane_texture = ET.Element('texture', type="2d", name="groundplane", builtin="checker", mark="edge", rgb1="0.2 0.3 0.4", rgb2="0.1 0.2 0.3", markrgb="0.8 0.8 0.8", width="300", height="300")
        groundplane_material = ET.Element('material', name="groundplane", texture="groundplane", texuniform="true", texrepeat="5 5", reflectance="0.2")
        assets.append(skybox_texture)
        assets.append(groundplane_texture)
        assets.append(groundplane_material)
        self.root.append(assets)
    
    def add_options(self, **options_attributes):
        options = ET.Element('option', **options_attributes)
        self.root.append(options)
    
    def add_size(self, **size_attributes):
        size = ET.Element('size', **size_attributes)
        self.root.append(size)
    
    def get_xml_string(self):
        return ET.tostring(self.root, encoding='unicode')
    
    def save_xml_file(self, file_path):
        tree = ET.ElementTree(self.root)
        tree.write(file_path, encoding='unicode', xml_declaration=True)
    
    def prettify(self):
        rough_string = ET.tostring(self.root, 'utf-8')
        reparsed = minidom.parseString(rough_string)
        return reparsed.toprettyxml(indent="  ")
    
    def add_et_element(self, et_element):
        self.root.append(et_element)


    def mocap(self, pos, quat, size,density, rgba):
        pos = array_to_string(pos)
        quat = array_to_string(quat)
        size = array_to_string(size)
        rgba = array_to_string(rgba)

        mocap_body = ET.Element('body', name="mocap", mocap="true", pos=pos)
        geom = ET.Element('geom', type="capsule", quat=quat, size=size, density=density, rgba=rgba)
        mocap_body.append(geom)
        return mocap_body

    def apply_configuration(self, config, block_sizes=(0.05, 0.15, 0.03)):
        if block_sizes is None:
            block_sizes = (0.05, 0.15, 0.03)
        floor_pos= array_to_string(config.floor_pos)
        num_layer = config.num_layer
        world_body = ET.Element('worldbody')
        light = ET.Element('light',  name="top", pos="0 0 -1.5", dir="0 0 -1", directional="true")
        ground = ET.Element('geom', name="ground",type='plane', material="groundplane", group="2", pos=floor_pos, size='10 10 1', rgba='0.8 0.9 0.8 1')
        world_body.append(light)
        world_body.append(ground)
        side_camera = ET.Element('camera', name="side", pos="-1.323 -2.179 2.020", xyaxes="0.849 -0.529 -0.000 0.166 0.265 0.950" , fovy="60")

        world_body.append(side_camera)

        # Add jenga blocks
        for k in range(1, num_layer+1):
            odd_layer = k % 2 == 0
            height = config.floor_pos[2] + block_sizes[2] / 2.0 + k * 2 * block_sizes[2]
            noise = np.random.uniform(-0.001, 0.001, 3)
            middle_large = np.random.uniform(-1, 1, 1) > 0
            for i in range(3):
                block_name = self.get_block_name(k, i)

                color = "0.8 0.6 .4 1"
                if self.debug:
                    if odd_layer:
                        if (k+i) % 2 == 1:
                            color = "0.1 0.1 .1 1"
                    else:
                        if (k+i) % 2 == 1:
                            color = "0.9 0.9 .9 1"
                noise_mask = 0
                if (middle_large and i == 1) or (not middle_large and i != 1):
                    noise_mask = 1

                block_length = block_sizes[1]
                block_width = block_sizes[0]
                if noise_mask:
                    block_height = block_sizes[2]
                else:
                    block_height = block_sizes[2] * 0.95

                if odd_layer:
                    pos_str = array_to_string(np.array([block_length, block_width + 2 * block_width * i, height]) + noise)
                    world_body.append(free_block(name=block_name, pos=pos_str, odd_layer=odd_layer, color=color, length=block_length, width=block_width, height=block_height))
                    self.odd_layer_block_names.append(block_name)
                else:
                    pos_str = array_to_string(np.array([block_width + 2 * block_width * i, block_length, height]) + noise)
                    world_body.append(free_block(name=block_name, pos=pos_str, odd_layer=odd_layer, color=color, length=block_length, width=block_width, height=block_height))
                    self.even_layer_block_names.append(block_name)
                self.block_names.append(block_name)

        # add a viusalization placeholder body
        block_name = "vis_block"
        pos_str = array_to_string(np.array([10, 10, 0.3]))
        world_body.append(free_block(name=block_name, pos=pos_str, odd_layer=True, color="0.8 0.2 .2 0.5", length=block_length+0.005, width=block_width+0.005, height=block_height+0.005))
        self.block_names.append(block_name)
        # Add mocap body
        mocap_body = self.mocap(**config.mocap)
        world_body.append(mocap_body)

        # Add worldbody to root
        self.root.append(world_body)

    def get_block_name(self, layer, index):
        return f"block_{layer}_{index}"

    def generate_xml(self, config, geom_attributes=None, block_sizes=(0.05, 0.15, 0.03)):
        # clean the root element
        self.root.clear()
        self.add_default(geom_attributes=geom_attributes)
        self.add_global_visual()
        self.add_assets()
        self.add_options(timestep='0.002', solver="CG", cone="pyramidal", jacobian="sparse", tolerance="1e-7", iterations="100")
        self.add_size(njmax="4000", nconmax="1000", nstack="1000000")
        self.apply_configuration(config, block_sizes=block_sizes)
        return self.get_xml_string()


