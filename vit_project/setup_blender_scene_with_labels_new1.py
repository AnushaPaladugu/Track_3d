
import bpy
import bmesh

# Clear existing objects
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# Import the PLY file
bpy.ops.wm.ply_import(filepath="floorplan_3d_model5.ply")

# Create materials
materials = {
    "WallMaterial": (0.96, 0.87, 0.70, 1.0),
    "BathroomWallMaterial": (0.7, 0.9, 0.7, 1.0),
    "LivingRoomMaterial": (0.9, 0.8, 0.8, 1.0),
    "BedroomMaterial": (0.8, 0.8, 0.9, 1.0),
    "KitchenMaterial": (0.9, 0.9, 0.7, 1.0),
    "OtherRoomMaterial": (0.85, 0.85, 0.85, 1.0),
    "DoorFrameMaterial": (0.55, 0.27, 0.07, 1.0),
    "DoorPanelMaterial": (0.55, 0.27, 0.07, 1.0),
    "BathroomDoorMaterial": (0.4, 0.2, 0.1, 1.0),
    "WindowMaterial": (0.7, 0.9, 1.0, 0.5),
    "FloorMaterial": (0.95, 0.92, 0.88, 1.0),
    "BedMaterial": (0.6, 0.4, 0.2, 1.0),
    "SofaMaterial": (0.5, 0.5, 0.5, 1.0),
    "CounterMaterial": (0.7, 0.6, 0.5, 1.0),
    "SinkMaterial": (1.0, 1.0, 1.0, 1.0)
}

# Assign materials to the imported mesh
obj = bpy.context.scene.objects[0]
mesh = obj.data

# Create materials in Blender
bpy_materials = {}
for mat_name, color in materials.items():
    mat = bpy.data.materials.new(name=mat_name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    principled = nodes.get("Principled BSDF")
    principled.inputs["Base Color"].default_value = color
    if mat_name == "WindowMaterial":
        principled.inputs["Alpha"].default_value = 0.5
        mat.blend_method = 'BLEND'
    bpy_materials[mat_name] = mat

print("Materials created. Split the mesh in Blender and assign materials manually based on the following order:")
material_mapping = {i: name for i, name in enumerate([
    "WallMaterial", "BathroomWallMaterial", "LivingRoomMaterial", "BedroomMaterial",
    "KitchenMaterial", "OtherRoomMaterial", "DoorFrameMaterial", "DoorPanelMaterial",
    "BathroomDoorMaterial", "WindowMaterial", "BedMaterial", "SofaMaterial",
    "CounterMaterial", "SinkMaterial", "FloorMaterial"
])}
print(material_mapping)

# Add room labels as text objects
room_labels = [
    {"type": room["type"], "center": room["center"]} for room in [
        {"type": r["type"], "center": r["center"]} for r in rooms
    ]
]

for room in room_labels:
    bpy.ops.object.text_add(location=(room["center"][0], room["center"][1], room["center"][2] + 0.3))
    text_obj = bpy.context.object
    text_obj.data.body = room["type"]
    text_obj.scale = (0.02, 0.02, 0.02)
    text_obj.rotation_euler = (1.5708, 0, 0)  # Rotate to face up

# Set up the scene
bpy.context.scene.render.engine = 'CYCLES'
bpy.context.scene.world.node_tree.nodes["Background"].inputs[0].default_value = (0.8, 0.8, 0.8, 1)
bpy.ops.object.light_add(type='SUN', location=(5, 5, 5))
bpy.ops.object.camera_add(location=(2, -2, 2), rotation=(1.0, 0.0, 0.8))
bpy.context.scene.camera = bpy.context.object
