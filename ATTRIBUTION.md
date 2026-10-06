# Attribution

## Code

Model code adapted from leggedrobotics/iPlanner (commit 4a8d823, 2025-02-23).
Source: https://github.com/leggedrobotics/iPlanner

`iplanner_model/` holds these upstream files verbatim, except that one absolute import became relative:

- `iplanner/planner_net.py` (`PlannerNet`, `Decoder`), `from percept_net import PerceptNet` now
  `from .percept_net import PerceptNet`
- `iplanner/percept_net.py` (`PerceptNet`, `BasicBlock`)
- `iplanner/traj_opt.py` (`CubicSplineTorch`, `TrajOpt`)

`planner.py` ports `IPlannerAlgo.plan` of `iplanner/ip_algo.py` (the `Resize((360, 640))` plus `ToTensor` depth
transform, the three-channel expansion and the spline at step 0.1) and these parts of `src/iplanner_node.py`: the
`imageCallback` depth cleanup (non-finite to 0, millimeters to meters, beyond `depth_max` to 0, optional 180 degree
rotation), the goal point in the robot frame from `goalCallback`, the `conv_dist` convergence check of `spin`,
`isForwardTraking`, `fearPathDetection` and the path truncation of `pubPath` during a fear reaction, which here is an
empty chunk. The config keys and defaults are `config/vehicle_sim.yaml`. The sensor-offset keypoint shift of
`IPlannerAlgo` is not ported: the bridge expects the chunk in the robot frame and the jackal depth camera sits 3 cm
ahead of it, under upstream's 0.1 m threshold. Loading changed from `torch.load` of the pickled module to a strict
`load_state_dict` on a freshly built `PlannerNet(16, 5)`, the `in-channel` and `knodes` of
`config/training_config.json`.

## Weights

`plannernet.pt` (sha256 685f16cde28d05249d50d24ed79ab4bdc94b3fbbcb99c8dbaed31039d11633b9) from the Google
Drive file 1UD11sSlOZlZhzij2gG_OmxbBN4WxVsO_ linked in upstream's README, a pickled `(PlannerNet, best_loss)` tuple
with 53,326,864 parameters. `weights.yaml` fetches `plannernet_state_dict.pt` (sha256
beccb331e782130b4ebe7738020af89104a94e993f8a95f1b5dfec1a9ebf6cb4), its `state_dict()` re-saved as plain tensors, from
the mirror https://huggingface.co/arena-rosnav/iplanner, which also holds the original file and the conversion script.
Upstream's README is the only license statement that covers the weights ("This code is released under the MIT
License"), and the MIT grant in LICENSE permits copying, publishing and distributing. The mirror therefore redistributes
them under the same notice.

No runtime dependency is a fork: torch, torchvision, numpy and pillow from PyPI.

See LICENSE for upstream terms (MIT).

Cite:

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
