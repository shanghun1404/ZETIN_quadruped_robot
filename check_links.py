import os
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

from pxr import Usd, UsdGeom

usd_path = os.path.join(os.getcwd(), "for", "for.usd")
stage = Usd.Stage.Open(usd_path)

body_names = []
for prim in stage.Traverse():
    if prim.IsA(UsdGeom.Mesh) or prim.IsA(UsdGeom.Xform):
        # We only care about links, usually they have rigid body APIs or are children of Articulation
        pass
    if "PhysicsRigidBodyAPI" in prim.GetAppliedSchemas():
        body_names.append(prim.GetName())

print("Rigid Body Links:", body_names)

simulation_app.close()
