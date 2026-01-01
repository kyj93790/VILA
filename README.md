# Improving Fisher Information Estimation and Efficiency for LoRA-based LLM Unlearning

This repository provides the official PyTorch implementation of the following paper:
> [**Improving Fisher Information Estimation and Efficiency for LoRA-based LLM Unlearning**](https://arxiv.org/abs/2508.21300) <br>
> [Yejin Kim](https://sites.google.com/view/yejin-c-kim/home?authuser=0)^, [Eunwon Kim](https://sites.google.com/view/eunwon-kim)^, [Buru Chang](https://sites.google.com/view/buru-chang)†, [Junsuk Choe](https://sites.google.com/site/junsukchoe/)† <br>
> ^Co-first Authors, †Co-corresponding Authors

[![arXiv](https://img.shields.io/badge/arXiv-2508.21300-9acd32.svg)](https://arxiv.org/abs/2508.21300)
[![Accepted @ COLM 2025](https://img.shields.io/badge/Accepted%20%40-COLM%202025-orange.svg)](https://colmweb.org/)


## Abstract
> LLMs have demonstrated remarkable performance across various tasks but face challenges related to unintentionally generating outputs containing sensitive information. A straightforward approach to address this issue is to retrain the model after excluding the problematic data. However, this approach incurs prohibitively high computational costs. To overcome this limitation, machine unlearning has emerged as a promising solution that can effectively remove sensitive information without the need to retrain the model from scratch. Recently, FILA has been proposed as a parameter-efficient unlearning method by integrating LoRA adapters. Specifically, it calculates the Fisher information to identify parameters associated with the forget set and assigns them to LoRA adapters for updates. Despite its innovative approach, FILA still requires access to all model parameters and does not adequately account for fundamental assumptions underlying Fisher information, leading to inaccuracies in importance estimation. To address these limitations, we propose VILA, a novel unlearning framework that explicitly considers the assumptions overlooked in FILA, thereby enhancing the accuracy of parameter identification for the forget set. Moreover, VILA significantly reduces computational costs by enabling parameter identification without accessing the entire model. Our method achieves up to 100x higher parameter efficiency and 40x faster training speed compared to FILA, and sets new state-of-the-art performance on benchmarks including TOFU, WMDP, and MUSE.

## Ours Contributions
- 💡 We refine the Fisher information extraction process to account for **the distributional mismatch** between the forget set and the full data distribution
- 💡 We improve computational efficiency by **performing Fisher information extraction within the LoRA subspace**.

## Cite
```
@inproceedings{
kim2025improving,
title={Improving Fisher Information Estimation and Efficiency for LoRA-based LLM Unlearning},
author={Yejin Kim and Eunwon Kim and Buru Chang and Junsuk Choe},
booktitle={Second Conference on Language Modeling},
year={2025},
}
```
