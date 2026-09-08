# PercepTrack3D — Claude Code Project Instructions

## 1. Project identity

- Project name: **PercepTrack3D**
- Goal: Build a reproducible 3D perception pipeline using public KITTI camera and LiDAR data.
- Career target: Robot Perception / 3D Vision / Sensor Fusion engineer.
- Primary environment: Windows WSL Ubuntu.
- Current dataset location: `/home/ingon/datasets/KITTI/raw`
- The user is learning the foundations. Do not hide important logic behind large generated code blocks.

## 2. Final system in one sentence

Detect road objects in KITTI RGB images, combine the detections with calibrated LiDAR points to estimate each object's 3D position and size, track the objects over time, evaluate the results quantitatively, and finally package the working pipeline as ROS2 nodes with selected C++ components.

## 3. Why this project exists

The completed project should provide evidence for job requirements such as:

- Experience processing camera and LiDAR data
- Understanding point clouds and 3D geometry
- Camera–LiDAR calibration and coordinate transformations
- 2D object detection and 3D position estimation
- Multi-object tracking and state estimation
- ROS2 message, TF, rosbag, and RViz integration
- Python prototyping followed by C++ implementation
- Quantitative evaluation, profiling, testing, and reproducibility

The goal is not merely to produce an attractive demo video. Every major result must be explainable, testable, and measurable.

## 4. Current checkpoint

The following archives have been downloaded and copied into WSL:

```text
/home/ingon/datasets/KITTI/raw/
├── 2011_09_26_calib.zip
├── 2011_09_26_drive_0015_sync.zip
└── 2011_09_26_drive_0015_tracklets.zip
```

The archives may not yet be extracted. Before changing anything, inspect the actual filesystem and report its state.

## 5. Working agreement with the user

### Communication

- Explain all commands and concepts in clear Korean.
- Assume the user can follow instructions but does not yet understand all Python, NumPy, calibration, and 3D geometry details.
- Define technical terms the first time they appear.
- Show the expected output before asking the user to run a command.
- When an error occurs, explain the cause before applying a fix.

### Scope control

- Work on **one phase at a time**.
- Do not implement later phases unless the current phase's exit criteria pass.
- Before editing files, briefly state what will change and why.
- Do not generate the entire project in one session.
- Prefer a small correct baseline before optimizations or advanced models.
- Never delete datasets, user files, or existing work without explicit confirmation.

### Learning-first rule

For each important module, the user should be able to answer:

1. What is the input and its shape/type?
2. What is the output and its shape/type?
3. Which coordinate frame is used?
4. What is the main formula or algorithm?
5. What failure cases exist?
6. How is the result evaluated?

At the end of every phase, provide three short self-check questions and wait before moving to the next phase.

## 6. Source and research policy

Before implementing an unfamiliar data format, API, or algorithm:

1. Check the official documentation first.
2. Check the original paper or official repository when applicable.
3. Use tutorials and blog posts only as supplementary explanations.
4. Record the useful URL and one-line takeaway in `docs/references.md`.
5. Do not copy a complete implementation without understanding and citing its source.

Preferred sources:

- KITTI official dataset pages and development kit
- NumPy, OpenCV, Open3D, PyTorch, and Ultralytics official documentation
- ROS2 official documentation
- Original algorithm papers and author repositories

Claude may explain, review, and debug code, but must not replace the user's learning process with unexplained generated code.

## 7. Environment policy

- Begin CPU-first. A GPU is not required for loading data, geometry, projection, or basic tracking.
- Prefer Python 3.10 or 3.11 unless the installed environment requires another compatible version.
- Use an isolated virtual environment inside the repository.
- Pin direct dependencies in `requirements.txt` after verifying the environment.
- Record Python and package versions in the README.
- Keep the KITTI dataset outside the Git repository.
- Do not install ROS2 during the initial Python phases.
- Add ROS2 only after the standalone Python pipeline works.
- If WSL is Ubuntu 22.04, use ROS2 Humble. If it is Ubuntu 24.04, use ROS2 Jazzy. Verify the OS version before installation.

## 8. Repository structure

Build the repository gradually toward this structure. Do not create empty boilerplate for every future phase on day one.

```text
PercepTrack3D/
├── CLAUDE.md
├── README.md
├── requirements.txt
├── .gitignore
├── configs/
│   └── kitti.yaml
├── notebooks/
│   ├── 01_kitti_data_check.ipynb
│   ├── 02_lidar_visualization.ipynb
│   ├── 03_lidar_camera_projection.ipynb
│   ├── 04_yolo_detection.ipynb
│   ├── 05_camera_lidar_fusion.ipynb
│   └── 06_multi_object_tracking.ipynb
├── src/perceptrack3d/
│   ├── __init__.py
│   ├── data/kitti_loader.py
│   ├── geometry/calibration.py
│   ├── geometry/projection.py
│   ├── detection/detector.py
│   ├── fusion/frustum_fusion.py
│   ├── tracking/kalman_tracker.py
│   └── visualization/
├── tests/
├── docs/
│   ├── references.md
│   ├── decisions.md
│   └── learning_notes/
├── outputs/
└── ros2_ws/                 # Create only in the ROS2 phase
```

The `.gitignore` must exclude at least:

```gitignore
.venv/
__pycache__/
*.pyc
.ipynb_checkpoints/
data/
outputs/*
!outputs/.gitkeep
```

## 9. Notebook-to-module workflow

Notebooks are for exploration and visible learning. They are not the final application architecture.

For every feature:

1. Explore the smallest example in an `.ipynb` file.
2. Print shapes, dtypes, ranges, and coordinate-frame assumptions.
3. Visualize the result.
4. Once correct, move reusable logic into `src/perceptrack3d/`.
5. Import the module back into the notebook.
6. Add a small automated test for deterministic geometry or parsing logic.

Do not leave large reusable functions duplicated inside notebooks.

## 10. Implementation roadmap

### Phase 0 — Dataset and environment

Tasks:

- Inspect the three ZIP archives without deleting them.
- Extract them under `/home/ingon/datasets/KITTI/raw`.
- Verify the expected files and frame counts.
- Verify WSL Ubuntu version, Python version, storage space, and Git availability.
- Create the repository and isolated Python environment.
- Install only the packages needed for Phase 1.
- Create a minimal README and `.gitignore`.

Expected dataset elements:

```text
2011_09_26/
├── calib_cam_to_cam.txt
├── calib_imu_to_velo.txt
├── calib_velo_to_cam.txt
└── 2011_09_26_drive_0015_sync/
    ├── image_00/
    ├── image_01/
    ├── image_02/          # Left color camera; primary RGB input
    ├── image_03/
    ├── velodyne_points/   # LiDAR .bin files
    ├── oxts/
    └── tracklet_labels.xml
```

Exit criteria:

- One RGB `.png`, one LiDAR `.bin`, and all calibration text files are found.
- The exact paths are documented in `configs/kitti.yaml` without hard-coding them throughout the code.

### Phase 1 — Understand and load KITTI data

Create `notebooks/01_kitti_data_check.ipynb`.

Tasks:

- Load one `image_02` PNG using OpenCV.
- Explain OpenCV's BGR channel order.
- Print image shape, dtype, and pixel range.
- Load one Velodyne `.bin` with `numpy.fromfile(dtype=np.float32)`.
- Reshape the values to `N x 4` and explain `x, y, z, reflectance`.
- Print the number of points and per-column ranges.
- Confirm that matching image and LiDAR filenames refer to the same synchronized index.
- Visualize the point cloud with Open3D.

Refactor reusable loading logic into `src/perceptrack3d/data/kitti_loader.py`.

Exit criteria:

- Frame `0000000000` loads without error.
- RGB and point-cloud visualizations are saved as evidence.
- The loader rejects malformed files with a clear error.
- Basic loader tests pass.

### Phase 2 — Calibration and coordinate frames

Tasks:

- Read and explain `calib_velo_to_cam.txt` and `calib_cam_to_cam.txt`.
- Clearly distinguish the Velodyne, camera, rectified camera, and image pixel frames.
- Parse rotation, translation, rectification, and projection matrices.
- Represent rigid transforms with homogeneous `4 x 4` matrices.
- Add shape and numerical sanity checks.
- Document the transformation chain before writing the final projection function.

Exit criteria:

- The user can explain the transformation order.
- Identity and round-trip transformation tests pass within tolerance.
- No unexplained magic matrix multiplication remains.

### Phase 3 — Project LiDAR points onto the RGB image

Create `notebooks/03_lidar_camera_projection.ipynb`.

Required transformation concept:

```text
Velodyne point
  -> camera coordinate
  -> rectified camera coordinate
  -> homogeneous image coordinate
  -> divide by depth
  -> pixel (u, v)
```

Tasks:

- Remove points behind the camera.
- Project valid points into `image_02`.
- Remove points outside the image boundary.
- Color projected points by depth.
- Save an overlay image.
- Test the projection with known shape, bounds, and positive-depth conditions.

Exit criteria:

- LiDAR points align visually with road, vehicles, and scene structure.
- Pixel-bound and positive-depth tests pass.
- The projection function is moved into `src/perceptrack3d/geometry/projection.py`.

### Phase 4 — 2D object detection baseline

Tasks:

- Use a pretrained lightweight YOLO model; do not train a model initially.
- Run inference on `image_02` frames.
- Retain relevant road-object classes such as car, truck, bus, person, bicycle, and motorcycle, while documenting label differences from KITTI.
- Store each detection as class, confidence, and `xyxy` bounding box.
- Measure per-frame inference time after warm-up.
- Save detection examples and failure cases.

Exit criteria:

- Deterministic detection output format is defined.
- At least one short sequence runs end-to-end.
- Runtime and qualitative failure examples are documented.

### Phase 5 — Camera–LiDAR fusion baseline

Baseline algorithm:

1. Project LiDAR points into the image.
2. Select points whose pixels fall inside each 2D detection box.
3. Reject invalid and clearly implausible points.
4. Estimate an initial 3D center using robust depth statistics.

Tasks:

- Visualize which LiDAR points were assigned to each 2D box.
- Report the number of assigned points.
- Use median or percentile-based depth instead of a raw mean baseline.
- Document the foreground/background contamination problem.
- Handle empty or sparse point sets explicitly.

Exit criteria:

- Each valid detection can produce a documented 3D position or a clear invalid status.
- Sparse, empty, and overlapping-box cases do not crash the pipeline.

### Phase 6 — Improve 3D localization and bounding boxes

Build improvements one at a time and compare them against the Phase 5 baseline:

- Region-of-interest filtering
- Optional ground removal
- Frustum-based point filtering
- Euclidean clustering or DBSCAN to separate foreground from background
- Robust 3D centroid estimation
- Axis-aligned 3D bounding-box baseline
- Optional oriented bounding box after the baseline works
- Bird's-eye-view visualization

Exit criteria:

- Baseline and improved methods are compared on the same frames.
- At least one quantitative localization or box-overlap metric is reported.
- Improvements are justified by measurements, not only screenshots.

### Phase 7 — Multi-object tracking

Start with a transparent classical baseline:

- Track state: position and velocity, e.g. `[x, y, z, vx, vy, vz]`
- Kalman filter prediction and correction
- Hungarian assignment
- Distance or Mahalanobis gating
- Track birth, confirmation, missed-frame handling, and deletion
- Stable track IDs in image and bird's-eye view

Do not use a black-box tracker until the classical baseline is understood.

Exit criteria:

- Tracks survive short detection misses without obvious ID explosion.
- Assignment cost, gate, and lifecycle thresholds are configurable.
- ID-switch and trajectory failure examples are documented.

### Phase 8 — Ground truth and quantitative evaluation

Tasks:

- Parse `tracklet_labels.xml` and verify its coordinate conventions.
- Match predictions and tracklets carefully; document assumptions.
- Evaluate meaningful subsets rather than reporting unsupported global claims.
- Report runtime for loading, detection, fusion, tracking, and total pipeline.
- Compare at least these variants:
  - A: 2D box + raw LiDAR statistics
  - B: 2D box + filtered/clustered LiDAR
  - C: Variant B + temporal tracking

Candidate metrics, subject to label compatibility:

- 3D center distance error
- Depth error
- Bird's-eye-view IoU or 3D IoU
- Precision/recall under a defined matching threshold
- Track continuity and ID switches
- Frames per second and stage latency

Exit criteria:

- Evaluation code is reproducible from a command.
- A table and plots summarize results.
- Limitations and invalid comparisons are stated honestly.

### Phase 9 — ROS2 integration

Only start after the standalone Python pipeline works.

Target interfaces:

- RGB: `sensor_msgs/msg/Image`
- LiDAR: `sensor_msgs/msg/PointCloud2`
- Camera calibration: `sensor_msgs/msg/CameraInfo`
- Coordinate frames: `tf2`
- Results: suitable detection/tracking messages and `visualization_msgs/msg/MarkerArray`
- Playback: KITTI sequence publisher or rosbag-based replay
- Visualization: RViz2

Suggested nodes:

```text
kitti_player_node
    -> detector_node
    -> lidar_projection_node
    -> fusion_node
    -> tracker_node
    -> visualization_node
```

Exit criteria:

- Timestamps and frame IDs are consistent.
- RViz displays point cloud, camera, transforms, 3D objects, IDs, and trajectories.
- Launch and configuration files reproduce the demo.

### Phase 10 — C++ and performance engineering

Do not rewrite everything in C++ blindly.

Tasks:

- Profile the Python/ROS2 pipeline first.
- Select one meaningful bottleneck, such as PointCloud2 conversion, projection, ROI filtering, or clustering.
- Implement that component in modern C++.
- Use RAII, const-correctness, smart pointers where ownership is needed, and clear value/reference semantics.
- Add unit tests and compare output against the Python reference.
- Measure latency before and after.

Exit criteria:

- Numerical output agrees with the Python baseline within tolerance.
- A measured performance improvement or an honest no-improvement result is reported.
- The user can explain why that component was selected for C++.

## 11. Reproducibility and quality rules

- No absolute dataset path scattered across source files; use one configuration file.
- Set random seeds where randomness exists.
- Save evaluation configuration with results.
- Use type hints for public Python functions.
- Add short docstrings explaining inputs, outputs, units, and frames.
- Validate array shapes at module boundaries.
- Prefer pure functions for geometry where possible.
- Add tests for parsers, transforms, projections, and association utilities.
- Do not commit large datasets, model weights, caches, or generated videos.
- Never report performance without recording hardware, sequence, frame count, and measurement method.

## 12. README evidence checklist

The final README should contain:

- One-sentence problem definition
- System diagram
- Dataset and coordinate-frame explanation
- Installation and reproduction commands
- Camera–LiDAR projection result
- 2D detection and associated LiDAR points
- 3D/BEV tracking visualization
- Baseline-vs-improvement evaluation table
- Runtime breakdown
- Failure cases and limitations
- ROS2 node/topic/TF diagram
- C++ optimization comparison
- Sources and license/usage notes

## 13. What Claude should do first

When the user begins the first Claude Code session:

1. Read this file.
2. Inspect the current directory and `/home/ingon/datasets/KITTI/raw` without modifying anything.
3. Report whether the ZIP archives are present and whether extraction has already happened.
4. Check `lsb_release -a`, `python3 --version`, `git --version`, and available disk space.
5. Propose only the Phase 0 commands.
6. Explain each command in Korean and show the expected result.
7. Wait for user approval before installing packages or creating multiple project files.

The first milestone is not YOLO or ROS2. It is:

> Load one synchronized KITTI RGB frame and one matching LiDAR frame, understand their data structures, and visualize both correctly.
