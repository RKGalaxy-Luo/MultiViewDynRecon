# Non-Volumetric Multi-view Dynamic Reconstruction with Adaptive Correction and Cross-View Smoothing

Multi-view reconstruction of the surfel model, smoothing and compression of the surfel model cross-view junction, adaptive correction of the tracking failure of the surfel model caused by major topology changes.

![](docs/rawImage.gif) ![](docs/Phong.gif) ![](docs/Normal.gif)



Whether to use cross-view smooth (upper: with cross-view term, bottom: without cross-view term)

![](docs/CrossViewComparison.gif)

Whether to use adaptive correction. (upper: with adaptive correction, bottom: without adaptive correction)

![](docs/AdaptiveCorrection.gif)

Our method is compared with SurfelWarp under single view condition (upper: our method, bottom: SurfelWarp).

![](docs/SingleViewComparison.gif)

Our Dataset can be found at: https://pan.baidu.com/s/1LDGWGoAHXkRNgB_auW2YeA?pwd=pv1s  download code: pv1s 



## Attention

Our datasets are not for commercial use. If you would like to use our datasets or obtain additional data, please contact: [maxiaolin0615@whut.edu.cn](mailto:maxiaolin0615@whut.edu.cn).

## Acknowledgements

This project would not have been possible without relying on some awesome repos : [SurfelWarp](https://github.com/weigao95/surfelwarp), [PoissonRecon](https://github.com/DavidXu-JJ/PoissonRecon_GPU), [PoissonReconGPU](https://github.com/RKGalaxy-Luo/Poisson-Surface-Reconstruction-GPU)

