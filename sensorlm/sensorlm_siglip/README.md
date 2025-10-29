# SensorLM-SigLIP
This folder contains the code for training the [SigLIP](https://arxiv.org/abs/2303.15343) variant of [SensorLM](https://arxiv.org/abs/2506.09108).

This link works as an extension to the [Big Vision](https://github.com/google-research/big_vision) library, adding support for SensorLM-based models.

It contains the necessary configuration files, model definitions, and preprocessing utilities.


Code structure:

```
sensorlm_siglip/
│
├── configs/proj/image_text/
│   └── siglip_sensorlm.py         # Config for training SensorLM-SigLIP model
│
├── models/
│   └── vit.py                     # Modified Vision Transformer implementation (patch)
│
├── pp/
│   └── ops_image.py               # Custom sensor preprocessing operations
│
└── README.md
```

## Getting Started

For running the code, clone the official Big Vision repo, then copy the provided patch files into the corresponding directories.

```
git clone https://github.com/google-research/big_vision.git
cd big_vision

# Copy your patch files into the existing structure
cp -r ../sensorlm_siglip/* .
```

To train or evaluate the SigLIP–SensorLM model, use Big Vision’s standard launcher with the provided config.


## Citing SensorLM

If you use SensorLM datasets or code in your research, please cite the manuscript using:

```bib
@inproceedings{zhang2025sensorlm,
    title={SensorLM: Learning the Language of Wearable Sensors},
    author={Yuwei Zhang and Kumar Ayush and Siyuan Qiao and A. Ali Heydari and Girish Narayanswamy and Maxwell A. Xu and Ahmed A. Metwally and Shawn Xu and Jake Garrison and Xuhai Xu and Tim Althoff and Yun Liu and Pushmeet Kohli and Jiening Zhan and Mark Malhotra and Shwetak Patel and Cecilia Mascolo and Xin Liu and Daniel McDuff and Yuzhe Yang},
    booktitle={Conference on Neural Information Processing Systems (NeurIPS)},
    year={2025}
}
```

## Contributing

For details on contributing to this repository, please see [CONTRIBUTING.md](https://github.com/Google-Health/consumer-health-research/blob/main/CONTRIBUTING.md).

## License

Copyright 2025 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

[https://www.apache.org/licenses/LICENSE-2.0](https://www.apache.org/licenses/LICENSE-2.0)

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

## Disclaimers

This is not an officially supported Google product. This project is not eligible for the [Google Open Source Software Vulnerability Rewards Program](https://bughunters.google.com/open-source-security). This project is intended for demonstration purposes only. It is not intended for use in a production environment.

NOTE: the content of this research code repository (i) is not intended to be a medical device; and (ii) is not intended for clinical use of any kind, including but not limited to diagnosis or prognosis.
