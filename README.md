This an implementation of Soft Poroposal (SP) layer in caffe.

The layer is implemetated both in CPU and GPU mode, which can be used as
```
layer
{
  type: SoftProposal
  name: sp
  bottom: bottom
  top: sp
}
```

The pre-trained model of Soft Proposal Network (SPN) is coming soon ...

If you use this code, please cite
```
@INPROCEEDINGS{Zhu2017SPN,
    author = {Zhu, Yi and Zhou, Yanzhao and Ye, Qixiang and Qiu, Qiang and Jiao, Jianbin},
    title = {Soft Proposal Networks for Weakly Supervised Object Localization},
    booktitle = {ICCV},
    year = {2017}
}
```
