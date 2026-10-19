# VidVRD RelateAnything: Surveillance Video Visual Relationship Detection System

This repository provides an end-to-end framework for **Video Visual Relationship Detection (VidVRD)** in complex surveillance scenarios. The system integrates decoupled person/object tracking, scale-invariant **Spatio-Temporal Interaction Clustering (STIC)**, and closed-vocabulary relational reasoning powered by **RelateAnything** (`maelic/relsgg-vits16plus`).

---

## 1. System Architecture

The pipeline processes surveillance videos from raw frames to structured relational representations and visual outputs through four sequential stages:

```
[Raw Surveillance Video (MP4)]
               │
               ▼
┌─────────────────────────────────────────────────────────┐
│ 1. Decoupled Causal Tracking Engine                     │
│    - Persons: Standalone ByteTrack + OSNet MSMT17 Re-ID │
│    - Objects: Hungarian Spatial IoU Tracker             │
│    - Monotonic Causal Agnostic Entity ID Assignment     │
└──────────────────────────────┬──────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────┐
│ 2. Spatio-Temporal Interaction Clustering (STIC v2)     │
│    - Computational Proxemics (Edward T. Hall)           │
│    - Monocular Ground Plane 3D Depth Alignment          │
│    - Pedestrian Passing-by Co-direction Gating          │
│    - Rigid Kinematic Coupling & Sub-part Subordination  │
└──────────────────────────────┬──────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────┐
│ 3. Closed-Vocabulary Relational Reasoning                │
│    - RelateAnything Vision-Language Foundation Model    │
│    - VidVRD Ontology: 60 Object Classes & 26 Predicates │
│    - Dynamic ROI Zoom Crop with Adaptive Context Margin │
│    - Temporal Hysteresis Debouncing & Ballistic FSM     │
└──────────────────────────────┬──────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────┐
│ 4. Decoupled Multi-Modal Export                         │
│    - Structured Relations & Trajectories JSON           │
│    - Clean Streaming Surveillance Visualizer MP4        │
│    - Offline Pure OpenCV Renderer (No GPU Needed)       │
└─────────────────────────────────────────────────────────┘
```

### Key Components

- **Causal Decoupled Tracking:** Separates human tracking from inanimate object tracking to prevent cross-class identity switches. Human identities are reinforced across occlusions using appearance embeddings extracted via OSNet-x0.25 (trained on MSMT17).
- **Spatio-Temporal Interaction Clustering (STIC):** Instead of evaluating an all-pairs combinatorial explosion ($O(N^2)$), STIC groups spatio-temporally proximate entities into interaction clusters using anthropometric reference heights ($H_{\text{ref}}$) and contact line depth gating ($|y_{2, a} - y_{2, b}| \le 0.35 \cdot H_{\text{ref}}$).
- **Kinematic Gating & FSM State Machines:** Distinguishes dynamic transitional actions (`get_on`, `get_off`, `throw`) from continuous durative postures (`ride`, `hold`, `carry`) using velocity differentials and contact persistence.
- **Decoupled Architecture:** Predictions are serialized into a comprehensive JSON file containing per-frame bounding box trajectories, allowing full visualization reconstruction offline in pure OpenCV without re-running deep learning models.

---

## 2. Environment Setup & Requirements

### Hardware & OS
- **Operating System:** Windows 10/11 or Linux (Ubuntu 20.04/22.04).
- **GPU:** NVIDIA GPU with CUDA 11.8+ recommended (tested on RTX series with 8GB+ VRAM). CPU inference is supported via `--device cpu`.
- **Python:** Python 3.10 to 3.12 (virtual environment recommended).

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/higo-ai/RelateAnything.git
   cd RelateAnything
   ```

2. **Set up virtual environment:**
   ```bash
   python -m venv .venv
   # Windows PowerShell:
   .\.venv\Scripts\Activate.ps1
   # Linux/macOS:
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   pip install ultralytics opencv-python Pillow scipy streamlit pandas pyarrow
   pip install -e ".[hub]"
   ```

4. **Verify model weights:**
   Ensure the detector weights are placed under `models/`:
   - `models/yoloe-26m-seg.pt` (YOLO segmentation weights)
   - `models/osnet_x0_25_msmt17.pth` (Optional local OSNet weights; falls back to automatic download)

---

## 3. Quickstart & Command-Line Interface (CLI)

The repository provides a single unified entry point `run.py` to analyze any surveillance video.

### Run End-to-End Analysis
```powershell
.venv\Scripts\python.exe run.py --video data/videos/video2.mp4 --device cuda
```

### CLI Arguments
| Argument | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `--video` | `str` | *Required* | Path to input video file (e.g., `data/videos/video1.mp4`). |
| `--mode` | `str` | `roi` | Inference mode: `roi` (STIC Dynamic ROI Zoom) or `full` (Full-Frame Baseline). |
| `--conf` | `float` | `0.35` | YOLO detector confidence threshold. |
| `--device` | `str` | `auto` | Compute device: `auto`, `cuda`, or `cpu`. |
| `--output_subdir` | `str` | `None` | Optional custom output subdirectory under `data/output/`. |

Execution outputs are automatically generated in `data/output/full_video/<video_name>/`:
- `<video_name>_relations.json`: Structured entity tracks, frame trajectories, and detected relationships.
- `<video_name>_visualizer.mp4`: Surveillance monitor video rendered with Set-of-Marks labels and interaction badges.

---

## 4. Offline Video Renderer from JSON

A notable feature of the system is complete decoupling between model inference and visualization rendering. Once a video is analyzed and its JSON output is generated, the visualizer MP4 can be reproduced in approximately 2–3 seconds on CPU without loading PyTorch or YOLO:

```powershell
.venv\Scripts\python.exe render_from_json.py `
    --video data/videos/video2.mp4 `
    --json data/output/full_video/video2/video2_relations.json `
    --output data/output/full_video/video2/video2_reconstructed.mp4
```

---

## 5. Web Application Interface (Streamlit)

For interactive demonstration and inspection, a local Streamlit web application is included:

```powershell
.venv\Scripts\streamlit.exe run app.py
```

### Application Features
- **Video Source Selection:** Select directly from 15 benchmark videos in `data/videos/` or upload custom MP4 files.
- **Dual Video Player:** Side-by-side comparison between raw surveillance source and rendered RelateAnything visualizer.
- **Relational Data Table:** Filterable summary of detected relationships, confidence scores, and active time intervals.
- **Entity Trajectory Explorer:** Detailed inspection of tracked entities, classes, and frame lifespans.
- **Instant JSON Export:** In-browser JSON viewer with a direct download button.
- **Offline Fast Re-render Tab:** Trigger fast CPU reconstruction directly from the user interface.

---

## 6. Output JSON Schema Specification

The generated JSON file adheres to the following specification:

```json
{
  "video_metadata": {
    "video_name": "video2",
    "fps": 25.0,
    "total_frames": 2818,
    "resolution": [1920, 1080],
    "duration_sec": 112.72,
    "mode": "roi_zoom",
    "description": "VidVRD Spatio-Temporal Interaction Clustering (STIC) with Causal Decoupled Tracking"
  },
  "tracks": {
    "[7]": {
      "track_id": "[7]",
      "type": "person",
      "class": "person",
      "color": [255, 128, 0],
      "total_frames_tracked": 321,
      "trajectories": {
        "1000": [450, 210, 520, 480],
        "1001": [452, 211, 521, 481]
      }
    }
  },
  "relations": [
    {
      "subject_id": "[7]",
      "subject_class": "person",
      "predicate": "carry",
      "object_id": "[9]",
      "object_class": "handbag",
      "confidence": 0.9812,
      "temporal_consistency": 1.0,
      "time_range": {
        "start_sec": 40.0,
        "end_sec": 52.84
      },
      "active_frames": [1000, 1001, 1002]
    }
  ],
  "clusters": [ ... ],
  "singletons": [ "[1]", "[2]", "[3]", "[5]" ]
}
```

### Schema Field Descriptions
- `video_metadata`: Core video properties including frame rate, resolution, frame count, and processing mode.
- `tracks`: Dictionary mapping each causal entity ID (`[id]`) to its semantic class, entity type (`person` / `object`), display color, and per-frame bounding box coordinates (`[x1, y1, x2, y2]`).
- `relations`: List of validated visual relationship triplets. Each record includes subject/object IDs and classes, predicted predicate, mean confidence score, temporal consistency ratio, time interval, and the precise list of active frame indices.
- `clusters` & `singletons`: Auxiliary grouping metadata documenting spatial clusters and isolated unengaged entities.

---

## 7. Experimental Evaluation on Benchmark Surveillance Videos

The system has been evaluated across 15 real-world surveillance video clips covering varied outdoor, indoor, and thermal infrared environments:

| Video | Scenario & Technical Challenge | Clusters | Primary Detected Relationships | Assessment |
| :--- | :--- | :---: | :--- | :--- |
| **`video1`** | Building corridor (73s). Doorway entry at 00:24s; two conversing people; backpack and fire extinguisher. | 1 | `[6] carry [8]` (0.96)<br>`[6] hold [8]` (0.87)<br>`[7] hold [6]` (0.53) | Doorway strangers isolated; non-interacting fixtures excluded. |
| **`video2`** | Crowded walkway (94s). 22 concurrent pedestrians passing each other; luggage items. | 6 | `[7] carry [9]` (0.98)<br>`[11] carry [12]` (0.96)<br>`[17] hold [16]` (0.91)<br>`[19] carry [20]` (0.95)<br>`[21] carry [22]` (0.95) | 6 independent clusters preserved; ground plane depth alignment prevents corridor cross-depth merging. |
| **`video3`** | Train hall (87s). Stroller pushing, luggage pulling. | 3 | `[2] carry [3]` (0.97)<br>`[4] carry [5]` (0.95)<br>`[7] carry [6]` (0.89)<br>`[7] pull [8]` (0.71) | Captures continuous wheeled luggage pulling (`pull`). |
| **`video4`** | Pedestrian corridor (39s). Walking and carrying personal bag. | 1 | `[4] carry [5]` (0.96)<br>`[4] hold [5]` (0.96) | Stable tracking without spurious relational artifacts. |
| **`video5`** | Thermal Infrared (110s). Low contrast, monochrome thermal camera. | 6 | `[10] carry [16]` (0.88)<br>`[14] ride [8]` (0.60)<br>`[14] get_off [17]` (0.59) | Functions under thermal IR conditions; captures mounting/dismounting episodes. |
| **`video7`** | Classroom (160s). Person placing bag on desk and leaving room. | 1 | `[2] hold [3]` (0.95)<br>`[2] carry [3]` (0.95) | Settled stationary clamping maintains settled bag at 0.00px jitter after release. |
| **`video10`** | Parking lot (84s). Walking, mounting motorcycle, riding, dismounting. | 2 | `[5] hold [4]` (0.97)<br>`[5] ride [4]` (0.86)<br>`[5] get_off [4]` (0.88)<br>`[7] ride [4]` (0.97) | Ground plane alignment distinguishes walking pedestrians from mounted riders. |
| **`video11`** | Outdoor combat scene (54s). Two persons striking in courtyard near parked vehicles. | 1 | `[6] hit [5]` (0.80)<br>`[5] hit [6]` (0.85)<br>`[5] kick [6]` (0.84)<br>`[6] kick [5]` (0.81) | Fast strike actions detected with 4-frame persistence; parked vehicles isolated. |
| **`video12`** | Single pedestrian crossing parking area (50s). | 0 | *(No interaction)* | Isolated singletons correctly yield zero false relations. |
| **`video13`** | Physical altercation in parking lot (52s). Striking and grappling. | 1 | `[5] hit [6]` (0.68)<br>`[6] hit [5]` (0.71)<br>`[5] kick [6]` (0.70) | Captures physical impact episodes; surrounding vehicles remain isolated. |
| **`video14`** | Solitary pedestrian walking in open yard (48s). | 0 | *(No interaction)* | Correctly identifies absence of interaction. |
| **`video15`** | Bag handling with motion blur (50s). Fast motion. | 1 | `[4] hold [5]` (0.98)<br>`[4] carry [5]` (0.95) | Maintained tracking through motion blur without trajectory fragmentation. |
| **`video16`** | Bag throwing action (51s). Impulsive release. | 1 | `[5] hold [6]` (0.94)<br>`[5] carry [6]` (0.93)<br>`[5] throw [6]` (0.88) | Ballistic transition FSM isolates rapid separation phase and confirms `throw`. |
| **`video17`** | Walking companions in courtyard (50s). Conversing pair. | 1 | `[4] hold [7]` (0.75)<br>`[7] hold [4]` (0.69) | Identifies close walking companion proximity; parked vehicles isolated. |
| **`video18`** | Walking couple in motorcycle parking area (54s). Overlapping vehicle perspective. | 1 | `[5] hold [6]` (0.91)<br>`[6] hold [5]` (0.77) | Couple clustered cleanly; perspective overlap with parked motorcycles does not swallow vehicles. |

---

## 8. Technical Limitations & Discussion

In the spirit of scientific transparency, several practical limitations of the current implementation are noted:

1. **Monocular 2D Projection Ambiguities:** In surveillance cameras with steep pitch angles, ground-plane depth approximation ($|y_{2, a} - y_{2, b}|$) assumes a flat horizontal ground plane. In scenes with stairs, ramps, or multi-level floors, this assumption can be violated.
2. **Detector Sensitivity on Small Objects:** Small handheld objects (e.g., phones, small wallets, or keys) remain challenging for YOLOE-26m at standard surveillance resolutions (e.g., 640x480 or 720p at distances $>15$m).
3. **Severe Mutual Occlusion:** In dense crowds where multiple individuals overlap for extended durations ($>3$s) with similar clothing appearance, ByteTrack identity continuity may degrade, occasionally causing tracklet ID renumbering upon separation.
4. **Domain Transfer:** While the 15 benchmark videos span diverse surveillance environments (indoors, corridors, open courtyards, parking facilities, and thermal IR), unconstrained real-world deployment across unseen weather conditions (heavy rain, fog, nighttime glare) requires continued calibration of detector thresholds.

---

## 9. Citation & Acknowledgments

This research was conducted within the **VinFast Computer Vision Center AI Trainee Research Program**.

```bibtex
@article{relateanything2026,
  title   = {Real-Time Visual Relationship Detection in Surveillance Video via Spatio-Temporal Interaction Clustering},
  author  = {VinFast CV Center AI Trainee Program},
  year    = {2026}
}
```
