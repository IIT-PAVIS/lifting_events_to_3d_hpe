[![Documentation
Status](https://readthedocs.org/projects/event-camera/badge/?version=latest)](https://event-camera.readthedocs.io/en/latest/?badge=latest)

# Lifting Monocular Events to 3D Human Poses
[Gianluca Scarpellini](https://scarpellini.dev/) [Pietro Morerio](https://scholar.google.com/citations?user=lPV9rbkAAAAJ&hl=it&oi=ao) [Alessio Del Bue](https://scholar.google.com/citations?user=LUzvbGIAAAAJ&hl=it&oi=ao)

![Method](./resources/figures/abstr_mono.png)

## Table of contents
- [Abstract](#Abstract)
- [Results](#Results)
- [References](#References)

## Abstract

This paper presents a novel 3D human pose estimation approach using a single
stream of asynchronous events as input. Most of the state-of-the-art approaches
solve this task with RGB cameras, however struggling when subjects are moving
fast. On the other hand, event-based 3D pose estimation benefits from the
advantages of event-cameras [[2]](#2), especially their efficiency and robustness to
appearance changes. Yet, finding human poses in asynchronous events is in
general more challenging than standard RGB pose estimation, since little or no
events are triggered in static scenes. Here we propose the first learning-based
method for 3D human pose from a single stream of events. Our method consists of
two steps. First, we process the event-camera stream to predict three orthogonal
heatmaps per joint; each heatmap is the projection of of the joint onto one
orthogonal plane. Next, we fuse the sets of heatmaps to estimate 3D localization
of the body joints. As a further contribution, we make available a new,
challenging dataset for event-based human pose estimation by simulating events
from the RGB Human3.6m dataset. Experiments demonstrate that our method achieves
solid accuracy, narrowing the performance gap between standard RGB and
event-based vision. 

- Code available at https://github.com/IIT-PAVIS/lifting_events_to_3d_hpe
  (implemented with pytorch-lightning [[1]](#1))
- Documentation at https://event-camera.readthedocs.io/en/latest/
- Paper (coming soon)

## Event-Human3.6m
![Example of simulated events from Human3.6m](./resources/figures/h3m.png)
The generation pipeline details and code are released at [https://github.com/IIT-PAVIS/lifting_events_to_3d_hpe/tree/master/scripts/h3m](https://github.com/IIT-PAVIS/lifting_events_to_3d_hpe/tree/master/scripts/h3m).

## Results
![Results](./resources/figures/results.png)

### DHP19

| Method                    | input     | MPJPE(mm) |
|---------------------------|-----------|-----------|
| Calabrese et al.          | stereo    |     79.63 |
| Constant-count -- stage 3 | monocular |     92.09 |
| Voxel-grid -- stage 3     | monocular |     95.51 |
| Constant-count -- stage 1 | monocular |     96.69 |
| Voxel-grid -- stage 1     | monocular |    105.24 |

### H3m
| Method                                                            | input  | MPJPE(mm) |
|-------------------------------------------------------------------|--------|-----------|
| Metha \etal (ResNet-50)                                           | RGB    |     80.50 |
| Kanazawa \etal  \cite{Kanazawa_2018}                              | RGB    |     88.00 |
| Nibali \etal \cite{Nibali20183DHeatmaps}                          | RGB    |     57.00 |
| Pavlakos \etal \cite{pavlakos17_coars_fine_volum_predic_singl}    | RGB    |     71.90 |
| Luvizon \etal \cite{Luvizon_2D3D_Pose_Estimation_CVPR_2018_paper} | RGB    |     53.20 |
| Cheng \etal \cite{Cheng_2020}                                     | RGB    |     40.10 |
|-------------------------------------------------------------------|--------|-----------|
| Spatio-temporal voxel-grid \textbf{(Ours)}                        | Events |    119.18 |
| Constant-count \textbf{(Ours)}                                    | Events |    116.40 |


## BibTex Citation

## References
<a id="1">[1]</a> Falcon, WA and .al (2019). PyTorch Lightning GitHub. Note:
https://github.com/PyTorchLightning/pytorch-lightning

<a id="2">[2]</a> Gallego, Guillermo, Tobi Delbruck, Garrick Michael Orchard, Chiara Bartolozzi, Brian Taba, Andrea Censi, Stefan Leutenegger, et al. 2020. “Event-Based Vision: A Survey.” _IEEE Transactions on Pattern Analysis and Machine Intelligence_. Institute of Electrical and Electronics Engineers (IEEE), 1. <http://dx.doi.org/10.1109/TPAMI.2020.3008413>
