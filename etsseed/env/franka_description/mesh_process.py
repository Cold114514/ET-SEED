import trimesh
from shutil import copyfile


def as_mesh(scene_or_mesh):
    """
    Convert a possible scene to a mesh.

    If conversion occurs, the returned mesh has only vertex and face data.
    """
    if isinstance(scene_or_mesh, trimesh.Scene):
        if len(scene_or_mesh.geometry) == 0:
            mesh = None  # empty scene
        else:
            # we lose texture information here
            mesh = trimesh.util.concatenate(
                tuple(trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
                    for g in scene_or_mesh.geometry.values()))
    else:
        #assert(isinstance(mesh, trimesh.Trimesh))
        mesh = scene_or_mesh
    return mesh

def main():
    import sys
    sys.path.append("/data2/haoran/RL-Pose/PoseOrientedGym/assets/v-hacd/app")
    sys.path.append("/data2/haoran/RL-Pose/PoseOrientedGym/assets/v-hacd/app/TestVHACD")
    sys.path.append("/data2/haoran/RL-Pose/PoseOrientedGym/assets/v-hacd/app/CMakeFiles/TestVHACD.dir")
    
    for i in range(1):
        # st = "{:0>3d}".format(i+1)/data2/haoran/RL-Pose/PoseOrientedGym/assets/v-hacd/app
        # print(st)
        infile = "/data2/haoran/RL-Pose/PoseOrientedGym/assets/franka_description/finger.obj"
        mesh = trimesh.load(infile)
        mesh = as_mesh(mesh)
        output_dir = "/data2/haoran/RL-Pose/PoseOrientedGym/assets/franka_description"
        trimesh.exchange.urdf.export_urdf(mesh, output_dir, resolution=1000000, concavity=0.0025, maxNumVerticesPerCH=150)
        copyfile(infile,output_dir+"model.obj")

if __name__ == "__main__":
    main()
