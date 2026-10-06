# iplanner

[iPlanner](https://github.com/leggedrobotics/iPlanner) (Yang, Wang, Cadena and Hutter, "iPlanner: Imperative Path
Planning", RSS 2023, arXiv 2302.11434) behind the arena_planners bridge: a ResNet-18 style encoder over one depth
frame and a decoder that maps the encoding plus a goal point in the robot frame to 5 keypoints and a "fear" collision
score. A cubic spline through the keypoints gives the path, returned as a 50-point waypoint chunk at 10 Hz. The
released weights were trained by imperative learning in simulation and the model has no language input.

```sh
arena feature planners add iplanner
arena launch sim:=gazebo robot:=jackal[camera=vln_rgbd] robot.planner:=iplanner
```

`robot.planner:=iplanner` selects `robot.mobile:=vla` with `goal_inputs: [pose]`. Each goto phase hands the planner
its goal pose, which the planner turns into a goal in the robot frame every tick from `robot_pose`. The robot
needs a `depth` sensor in `model_params.yaml`. The weights take 213 MB on disk.

## Inputs

| Datasource | Jackal topic (`camera=vln_rgbd`) | Model input |
|---|---|---|
| `depth` | `vln_camera/depth_image` (`32FC1` meters, `16UC1` millimeters also accepted) | `(1, 3, 360, 640)` depth in meters, resized by the bridge to 640x360, non-finite and beyond `depth_max` set to 0, the one channel repeated three times |
| `robot_pose` | TF `map -> base_link` | the subgoal `[x forward, y left, 0]` in meters in the robot frame, the goal pose for the `conv_dist` arrival check |
| `subgoal` | `plan` from the nav2 planner server, `global_costmap/costmap` | the farthest point up to 6 m along the plan whose straight chord from the robot stays clear of the costmap, used in place of the goal pose, the goal pose itself until a plan arrives |

`depends.global_plan: true` makes the adapter start a nav2 planner server (`robot.mobile.global_planner`, default
`nav2/navfn`) that replans to the goal every second, and the `SubgoalGenerator` datasource turns the plan into
`subgoal`. The model steers at that point, so it only ever sees a goal in line of sight. A goal pose without a route
left the robot stuck behind walls, see Notes.

Upstream trained on the CMU environment's depth camera: 640x360 at 2.0 rad (115 degrees) horizontal field of view,
0.1 to 10 m, mounted level 0.75 m above the ground. The jackal `vln_rgbd` depth camera is 256x256 at 90 degrees,
10 m far clip, 0.98 m above the floor. Its square frame is stretched to 16:9 on the way in, as upstream's node does
with any frame. The planner therefore sees a narrower, taller and higher view than in training. A dedicated camera
variant with upstream's 640x360, 2.0 rad field of view and 0.75 m mount would reproduce the training input exactly.
Pixels beyond 10 m read 0 in both cameras, as in the training frames (upstream's `dataloader.py` zeroes depth
beyond `max-camera-depth`).

## Actions

Each tick the network predicts 5 keypoints spaced about a fifth of the goal distance apart, and upstream's
`TrajGeneratorFromPFreeRot` samples the cubic spline through the origin and the keypoints at 51 points. The chunk is
those samples after the origin, `[x, y]` in meters in the robot frame of the observation. The bridge follows it with
pure pursuit (lookahead 0.25 m, 1.5 m/s, upstream's `iplanner_path_follow` tracking distance and speed cap, capped by
the robot's own limits) for one tick, then asks again.

The fear score is upstream's collision probability within 2 m along the path. With `is_fear_act: true`, the planner
mirrors upstream's `fearPathDetection`: a fear above 0.5 on a path that heads forward (the first waypoint past
`track_dist` lies within `angular_thred` of the heading, as a cosine) raises a counter, and once the counter exceeds
`buffer_size` the planner returns an empty chunk and the robot holds still, like upstream's path truncated to its first
point. The counter drains when fear drops or the path turns away. A stopped robot sees the same frame. The hold
therefore lasts until the view or the goal changes, as it does upstream.

Within `conv_dist` of the goal the planner stops planning and holds still, upstream's convergence check, and sends
`arrived` when the goto phase expects that signal.

## Config

Set in `planner.yaml > config`, override per launch with `robot.mobile.config.<key>:=<value>`. The keys and defaults
are upstream's `config/vehicle_sim.yaml`.

| Key | Default | Effect |
|---|---|---|
| `device` | `auto` | Torch device, `auto` picks CUDA when available, else CPU. |
| `depth_max` | `10.0` | Depth in meters above which a pixel reads as no return (0). |
| `image_flip` | `false` | Rotate the depth frame by 180 degrees, for a camera mounted upside down. |
| `conv_dist` | `0.25` | Distance to the goal in meters at which the planner stops. |
| `is_fear_act` | `false` | Hold still on a sustained high fear score, upstream's default. Off here, see Notes. |
| `buffer_size` | `3` | Ticks of high forward fear before the hold, and ticks of low fear before release. |
| `angular_thred` | `0.5` | One minus the cosine of the heading cone inside which a path counts as forward. |
| `track_dist` | `0.5` | Distance in meters past which a waypoint decides the path heading. |

## Weights and license

Code: MIT (`LICENSE`, Fan Yang, Robotic Systems Lab, ETH Zurich). Weights: upstream's `plannernet.pt` from the
Google Drive link in its README, which states "This code is released under the MIT License" and attaches no other
terms to the file. The MIT grant covers copying, publishing and distributing. The manifest therefore points at a mirror
under `arena-rosnav/iplanner` on Hugging Face that holds the original file and `plannernet_state_dict.pt`, the same
tensors re-saved as a plain `state_dict` that `torch.load(weights_only=True)` reads. Upstream's README notes
the network was trained in simulation and not adapted to real-world data. See LICENSE and ATTRIBUTION.md. Cite the
paper when using this planner:

```bibtex
@inproceedings{Yang-RSS-23,
  author    = {Fan Yang and Chen Wang and Cesar Cadena and Marco Hutter},
  title     = {{iPlanner: Imperative Path Planning}},
  booktitle = {Proceedings of Robotics: Science and Systems},
  year      = {2023},
  address   = {Daegu, Republic of Korea},
  month     = {July},
  doi       = {10.15607/RSS.2023.XIX.064}
}
```

## Notes

- Measured on an RTX PRO 6000 Blackwell with torch 2.14.1+cu130 on a 640x360 frame. CPU (AMD Ryzen 9 9950X3D): 0.24 s to
  load, 407 MiB of resident memory, 35.5 ms per step (min 33.0 ms). CUDA (RTX PRO 6000 Blackwell): 0.39 s to load, 204
  MiB allocated, 270 MiB peak, 1.6 ms per step.
- The route decides whether a goal behind walls is reached. On ten start and goal pairs of hospital_1 (the five
  `hospital_complex` stages and five more from shifted seeds, 4 to 24 m apart, 2 to 8 pedestrians, 100 s and one
  episode per pair) the planner reached 5 goals steering by the subgoal and 2 steering by the goal pose.
- On a flat floor with the goal 4 m ahead the released weights put the last keypoint at 4.0 m with fear 0.000, a
  wall across the path at 1 m gives fear 0.987, and goals 3 m to the left or right bend the last keypoint to y = 3.0
  or -3.0 (`tests/test_model.py`, which runs on CPU and on CUDA when present and skips without the weights).
- The fear reaction is off by default. On map_empty with the `vln_rgbd` camera and the released weights, the
  two-leg scenario (5 m east, then 5 m north) held the robot about 1 m short of the final goal on a sustained fear
  score until the 60 s timeout, in both episodes. With `is_fear_act: false` both episodes finished in 20 s. The view
  is narrower and higher than in training, see Inputs.
- The planner adds no goal clipping: upstream's waypoint mode passes the goal through, its joystick mode caps it at
  5 m.
