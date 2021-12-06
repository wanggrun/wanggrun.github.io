在我们在使用
torchvision.transforms.RandomResizedCrop(size, scale=(0.08, 1.0), ratio=(0.75, 1.3333333333333333), interpolation=<InterpolationMode.BILINEAR: 'bilinear'>)
或者
torchvision.transforms.Resize(size, interpolation=<InterpolationMode.BILINEAR: 'bilinear'>, max_size=None, antialias=None)
时，默认情况下，size都是输入int，如224。
如果我们想要H和W不同，如，我们需要长为H=384，W为512，本来按照官方api，也是可以的，只需要size=(384, 512)即可。但是在torch=1.9.0, torchvision=0.10.0的情况下，会出现bug。例如你的代码如下：
from torchvision import transforms
[
        transforms.Resize(img_size, _pil_interp(interpolation)),
        transforms.CenterCrop(img_size)
    ]
则会报以下错误：

 File "/home/wangguangrun/pytorch-image-model/AE/timm/data/transforms.py", line 90, in __call__
    return F.resized_crop(img, 0, 0, img.size[1], img.size[0], self.size, self.interpolation)
  File "/home/wangguangrun/anaconda2/envs/torch2021/lib/python3.8/site-packages/torchvision/transforms/functional.py", line 548, in resized_crop
    img = resize(img, size, interpolation)
  File "/home/wangguangrun/anaconda2/envs/torch2021/lib/python3.8/site-packages/torchvision/transforms/functional.py", line 401, in resize
    return F_pil.resize(img, size=size, interpolation=pil_interpolation, max_size=max_size)
  File "/home/wangguangrun/anaconda2/envs/torch2021/lib/python3.8/site-packages/torchvision/transforms/functional_pil.py", line 241, in resize
    return img.resize(size[::-1], interpolation)
  File "/home/wangguangrun/anaconda2/envs/torch2021/lib/python3.8/site-packages/PIL/Image.py", line 1888, in resize
    return self._new(self.im.resize(size, resample, box))
TypeError: an integer is required (got type tuple)

这是pytorch的一个bug。原因是pytorch本身不管你size是int还是tuple，它都做了一个self.size=(size, size)处理了。你如果size=224，则self.size=(224,224)，无问题。如果你size =(384,512)，则它就把你变成self.size=((384,512), (384, 512))。这就必然会出错了。

Solution:
为了解决这个问题，你需要重写一个transforms.py：

import torch
import torchvision.transforms.functional as F
class FixedResize(object):
    """Crop the given PIL Image to random size and aspect ratio with random interpolation.
    A crop of random size (default: of 0.08 to 1.0) of the original size and a random
    aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio is made. This crop
    is finally resized to given size.
    This is popularly used to train the Inception networks.
    Args:
        size: expected output size of each edge
        scale: range of size of the origin size cropped
        ratio: range of aspect ratio of the origin aspect ratio cropped
        interpolation: Default: PIL.Image.BILINEAR
    """
    def __init__(self, size, interpolation='bilinear'):
        size = size[0]
        if isinstance(size, (list, tuple)):
            self.size = size
        else:
            self.size = (size, size)
        if interpolation == 'random':
            self.interpolation = _RANDOM_INTERPOLATION
        else:
            self.interpolation = _pil_interp(interpolation)
    def __call__(self, img): """
        Args:
            img (PIL Image): Image to be cropped and resized.
        Returns:
            PIL Image: Randomly cropped and resized image.
        """
        return F.resized_crop(img, 0, 0, img.size[1], img.size[0], self.size, self.interpolation)
    def __repr__(self):
        if isinstance(self.interpolation, (tuple, list)):
            interpolate_str = ' '.join([_pil_interpolation_to_str[x] for x in self.interpolation])
        else:
            interpolate_str = _pil_interpolation_to_str[self.interpolation]
        format_string = self.__class__.__name__ + '(size={0}'.format(self.size)
        format_string += ', interpolation={0})'.format(interpolate_str)
        return format_string
