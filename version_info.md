# v0.0
Original ssd.pytorch.

# v1.0
Add one more conv layer at the finest feature map used for detection.
The purpose is to make the prediction of small objects more independent from the following layers.
In other word, we want to reduce the interference of following layers to the prediction of small objects.

Usage: Run 'train.py'

Change list:  
- [Modify_function] ssd.py -> multibox: define conf/cls layers according to branch4_3
- [Mddify_function] ssd.py -> SSD.__init__: generate ModuleList of added layer
- [Modify_file] train.py: init added conv layer
- [Modify_function] ssd.py -> SSD.forward: tile conf/cls layers onto branch4_3
- [Add_function] ssd.py -> add_branch4_3