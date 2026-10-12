<div align="center">
  <img src="assets/leafmot-banner.svg" alt="LeafMOT - Structural Multi-Object Tracking of Individual Leaves in Rosette Plants" width="100%">
</div>

<div align="center">
  <img alt="Manuscript under review" src="https://img.shields.io/badge/Manuscript-Under_review-b23a48?style=flat-square">
  <img alt="README-first release; code preparation ongoing" src="https://img.shields.io/badge/Release-README--first-d6a243?style=flat-square">
  <img alt="Task: leaf multi-object tracking" src="https://img.shields.io/badge/Task-Leaf_tracking-087f6b?style=flat-square">
</div>

<p align="center">
  <strong>Structural Multi-Object Tracking of Individual Leaves in Rosette Plants</strong>
</p>

<p align="center">
  <a href="#overview">Overview</a> · <a href="#benchmarks">Benchmarks</a> · <a href="#method">Method</a> · <a href="#release-status">Release status</a> · <a href="#citation">Citation</a>
</p>

## Overview

LeafMOT is a structure-aware multi-object tracking framework for individual leaves in longitudinal, top-view images of rosette plants. Rather than associating leaves by appearance alone, it combines RGB features with monocular depth, plant-relative geometry, developmental order, and online track memory to maintain identities through growth, occlusion, and in-plane rotation.

> This repository accompanies the manuscript under review. The overview and benchmark figure are available now; the complete code release and reproduction instructions are being finalized and will be published shortly.

## Benchmarks

![HOTA benchmark comparison across CanolaTrack, CanolaTrack-R, KOMATSUNA, and combined MSU-PID + CVPPP; seven models ranked highest to lowest in each panel](assets/hota-comparison.svg)

LeafMOT achieves the highest reported mean HOTA on all four evaluated benchmarks, compared with six competing methods. Higher HOTA is better.

## Method

| Component | Role in tracking |
|:--|:--|
| Depth-guided encoding | Monocular depth from Depth Anything 3 and DFormer RGB-depth fusion enrich leaf proposal features. |
| Rosette topology | Plant-relative geometry and sparse neighborhood relations place each leaf in the context of the whole plant. |
| Developmental order | An ordinal cue provides temporal context for leaf emergence and identity. |
| Structure-aware association | Online track memory and structural cues guide matching across frames. |


## Release status

This is a **README-first research release** so readers can examine the method and its reported results before the code package is complete.

The reproducibility release is being prepared to include the implementation, environment setup, data and detector-proposal preparation, depth preprocessing, training and inference commands, evaluation, and ablations. Please refer to the submitted manuscript for the experimental protocol.


## Citation

If you use CanolaTrack, CanolaTrack-R, or LeafMOT in your research, please cite the following papers. 

```bibtex
@article{liu2026leafmot,
  title  = {LeafMOT: Structural Multi-Object Tracking of Individual Leaves in Rosette Plants},
  author = {Liu, Shanghua and Babor, Majharulislam and H{\"o}hne, Marina M.-C.},
  year   = {2026},
  note   = {Manuscript under review}
}
```

```bibtex
@article{liu2026leaftracknet,
title = {LeafTrackNet: A deep learning framework for robust leaf tracking in top-down plant phenotyping},
journal = {Computers and Electronics in Agriculture},
volume = {256},
pages = {112417},
year = {2027},
issn = {0168-1699},
doi = {https://doi.org/10.1016/j.compag.2026.112417},
author = {Shanghua Liu and Majharulislam Babor and Christoph Verduyn and Breght Vandenberghe and Bruno Betoni Parodi and Cornelia Weltzien and Marina M.-C. Höhne},
}
```


## Acknowledgements

LeafMOT builds on [Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3), [DFormer](https://github.com/VCIP-RGBD/DFormer), and the broader multi-object tracking ecosystem. Please follow the licenses and citation requirements of the third-party models, datasets, and benchmarks used in your experiments.
