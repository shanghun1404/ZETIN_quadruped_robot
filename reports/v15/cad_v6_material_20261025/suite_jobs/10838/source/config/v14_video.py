"""Record observer cameras without changing or replaying robot physics."""

from pathlib import Path

from v11_video_geometry import tracking_camera_pose


class RolloutVideos:
    """Stream one view per selected environment; stop before its first auto-reset."""

    def __init__(self, sim, robot, output_dir, env_ids, *, view, fps):
        import imageio.v2 as imageio
        import omni.replicator.core as rep
        from pxr import Gf, UsdGeom

        self.sim = sim
        self.robot = robot
        self.view = view
        self.fps = fps
        self.streams = {}
        self.closed = False
        # Four benchmark robots share each terrain tile. Hide peers only during
        # observer renders, then restore before the next physics/depth-policy step.
        self.env_visibility = [
            UsdGeom.Imageable(sim.stage.GetPrimAtPath(f"/World/envs/env_{env_id}")).GetVisibilityAttr()
            for env_id in range(len(robot.data.root_pos_w))
        ]
        visuals = sim.stage.GetPrimAtPath("/Visuals")
        self.visual_visibility = UsdGeom.Imageable(visuals).GetVisibilityAttr() if visuals else None
        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=False)
        for env_id in env_ids:
            camera_path = f"/World/V14VideoObserver_{env_id}"
            camera = UsdGeom.Camera.Define(sim.stage, camera_path)
            camera.CreateFocalLengthAttr(24.0)
            camera.CreateHorizontalApertureAttr(24.0)
            camera.CreateClippingRangeAttr(Gf.Vec2f(0.05, 1000.0))
            product = rep.create.render_product(camera_path, (960, 640))
            annotator = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
            annotator.attach([product])
            path = directory / f"env_{env_id:02d}.mp4"
            self.streams[env_id] = {
                "camera": camera_path, "product": product, "annotator": annotator,
                "path": path, "frames": 0,
                "writer": imageio.get_writer(
                    str(path), fps=fps, codec="libx264", quality=8,
                    pixelformat="yuv420p", ffmpeg_params=["-movflags", "+faststart"],
                ),
            }
            self._track(env_id)
        # Initialize RGB products without advancing the physics or resetting robots.
        for _ in range(3):
            sim.render()

    def _track(self, env_id):
        root = tuple(self.robot.data.root_pos_w[env_id].detach().cpu().tolist())
        eye, target = tracking_camera_pose(root, self.view)
        self.sim.set_camera_view(eye, target, self.streams[env_id]["camera"])

    def capture(self, alive):
        active = [env_id for env_id in self.streams if bool(alive[env_id])]
        if not active:
            return
        for env_id in active:
            self._track(env_id)
        attributes = self.env_visibility + ([self.visual_visibility] if self.visual_visibility else [])
        previous = [(attr.HasAuthoredValueOpinion(), attr.Get()) for attr in attributes]
        try:
            for attr in attributes:
                attr.Set("invisible")
            for env_id in active:
                self.env_visibility[env_id].Set("inherited")
                self.sim.render()
                stream = self.streams[env_id]
                frame = stream["annotator"].get_data()
                if frame.size == 0:
                    raise RuntimeError(f"empty observer image for environment {env_id}")
                stream["writer"].append_data(frame[:, :, :3])
                stream["frames"] += 1
                self.env_visibility[env_id].Set("invisible")
        finally:
            for attr, (authored, value) in zip(attributes, previous):
                if authored:
                    attr.Set(value)
                else:
                    attr.Clear()

    def close(self):
        if self.closed:
            return
        for stream in self.streams.values():
            stream["writer"].close()
        for stream in self.streams.values():
            stream["annotator"].detach([stream["product"].path])
            stream["product"].destroy()
        self.closed = True

    def metadata(self):
        return [
            {
                "env_id": env_id, "path": str(stream["path"]),
                "frames": stream["frames"], "fps": self.fps,
                "duration_s": stream["frames"] / self.fps,
                "view": self.view, "resolution": [960, 640],
                "stops_before_first_auto_reset": True,
                "peers_hidden_only_during_observer_render": True,
            }
            for env_id, stream in self.streams.items()
        ]
