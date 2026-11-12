from pxr import Usd, UsdGeom, UsdPhysics
import omni
from isaacsim.core.utils.extensions import get_extension_path_from_name
"""
def main():
    stage = omni.usd.get_context().get_stage()
    frankaPrim = stage.GetPrimAtPath("/World/Franka")

    for prim in Usd.PrimRange(frankaPrim):
        primName = prim.GetName()
        if prim.IsA(UsdPhysics.RevoluteJoint):
            print("Revolute Joint: " + primName)
        elif prim.IsA(UsdPhysics.FixedJoint):
            print("Fixed Joint: " + primName)
        elif prim.IsA(UsdPhysics.PrismaticJoint):
            print("Prismatic Joint: " + primName)
        elif prim.IsA(UsdPhysics.SphericalJoint):
            print("Spherical Joint: " + primName)
"""
def main():
    
    import os

    mg_path = get_extension_path_from_name("isaacsim.robot_motion.motion_generation")
    urdf_path = os.path.join(
        mg_path,
        "motion_policy_configs/franka/lula_franka_gen.urdf"
    )

    robot_description_path = os.path.join(
        mg_path,
        "motion_policy_configs/franka/rmpflow/robot_descriptor.yaml"
    )
    print(urdf_path)
    print(robot_description_path)

main()
