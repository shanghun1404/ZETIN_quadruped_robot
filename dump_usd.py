from pxr import Usd, UsdGeom
stage = Usd.Stage.Open("for/for.usd")
for prim in stage.Traverse():
    if prim.IsA(UsdGeom.Xformable):
        print(prim.GetPath())
