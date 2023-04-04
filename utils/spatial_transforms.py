import random
import math
import numbers
import collections
import numpy as np
import torch
from PIL import Image, ImageOps
# import functions as F

try:
	import accimage
except ImportError:
	accimage = None
import types


class Compose(object):
	"""Composes several transforms together.
	Args:
		transforms (list of ``Transform`` objects): list of transforms to compose.
	"""
	
	def __init__(self, transforms):
		self.transforms = transforms
	
	def __call__(self, img):
		for t in self.transforms:
			img = t(img)
		return img
	
	def randomize_parameters(self):
		for t in self.transforms:
			t.randomize_parameters()


class Lambda(object):
	"""Apply a user-defined lambda as a transform.
	Args:
		lambd (function): Lambda/function to be used for transform.
	"""
	
	def __init__(self, lambd):
		assert isinstance(lambd, types.LambdaType)
		self.lambd = lambd
	
	def __call__(self, img):
		return self.lambd(img)
	
	def randomize_parameters(self):
		pass


class ToTensor(object):
	"""Convert a ``PIL.Image`` or ``numpy.ndarray`` to tensor.
	Converts a PIL.Image or numpy.ndarray (H x W x C) in the range
	[0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
	"""
	
	def __init__(self, norm_value=255):
		self.norm_value = norm_value
	
	def __call__(self, pic):
		"""
		Args:
			pic (PIL.Image or numpy.ndarray): Image to be converted to tensor.
		Returns:
			Tensor: Converted image.
		"""
		if isinstance(pic, np.ndarray):
			# handle numpy array
			img = torch.from_numpy(pic.transpose((2, 0, 1)))
			# backward compatibility
			return img.float().div(self.norm_value)
		
		if accimage is not None and isinstance(pic, accimage.Image):
			nppic = np.zeros([pic.channels, pic.height, pic.width], dtype=np.float32)
			pic.copyto(nppic)
			return torch.from_numpy(nppic)
		
		# handle PIL Image
		if pic.mode == 'I':
			img = torch.from_numpy(np.array(pic, np.int32, copy=False))
		elif pic.mode == 'I;16':
			img = torch.from_numpy(np.array(pic, np.int16, copy=False))
		else:
			img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
		# PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
		if pic.mode == 'YCbCr':
			nchannel = 3
		elif pic.mode == 'I;16':
			nchannel = 1
		else:
			nchannel = len(pic.mode)
		img = img.view(pic.size[1], pic.size[0], nchannel)
		# put it from HWC to CHW format
		# yikes, this transpose takes 80% of the loading time/CPU
		img = img.transpose(0, 1).transpose(0, 2).contiguous()
		if isinstance(img, torch.ByteTensor):
			return img.float().div(self.norm_value)
		else:
			return img
	
	def randomize_parameters(self):
		pass


class Normalize(object):
	"""Normalize an tensor image with mean and standard deviation.
	Given mean: (R, G, B) and std: (R, G, B),
	will normalize each channel of the torch.*Tensor, i.e.
	channel = (channel - mean) / std
	Args:
		mean (sequence): Sequence of means for R, G, B channels respecitvely.
		std (sequence): Sequence of standard deviations for R, G, B channels
			respecitvely.
	"""
	
	def __init__(self, mean, std):
		self.mean = mean
		self.std = std
	
	def __call__(self, tensor):
		"""
		Args:
			tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
		Returns:
			Tensor: Normalized image.
		"""
		# TODO: make efficient
		for t, m, s in zip(tensor, self.mean, self.std):
			t.sub_(m).div_(s)
		return tensor
	
	def randomize_parameters(self):
		pass


class Scale(object):
	"""Rescale the input PIL.Image to the given size.
	Args:
		size (sequence or int): Desired output size. If size is a sequence like
			(w, h), output size will be matched to this. If size is an int,
			smaller edge of the image will be matched to this number.
			i.e, if height > width, then image will be rescaled to
			(size * height / width, size)
		interpolation (int, optional): Desired interpolation. Default is
			``PIL.Image.BILINEAR``
	"""
	
	def __init__(self, size, interpolation=Image.BILINEAR):
		assert isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)
		self.size = size
		self.interpolation = interpolation
	
	def __call__(self, img):
		"""
		Args:
			img (PIL.Image): Image to be scaled.
		Returns:
			PIL.Image: Rescaled image.
		"""
		if isinstance(self.size, int):
			w, h = img.size
			if (w <= h and w == self.size) or (h <= w and h == self.size):
				return img
			if w < h:
				ow = self.size
				oh = int(self.size * h / w)
				return img.resize((ow, oh), self.interpolation)
			else:
				oh = self.size
				ow = int(self.size * w / h)
				return img.resize((ow, oh), self.interpolation)
		else:
			return img.resize(self.size, self.interpolation)
	
	def randomize_parameters(self):
		pass


class CenterCrop(object):
	"""Crops the given PIL.Image at the center.
	Args:
		size (sequence or int): Desired output size of the crop. If size is an
			int instead of sequence like (h, w), a square crop (size, size) is
			made.
	"""
	
	def __init__(self, size):
		if isinstance(size, numbers.Number):
			self.size = (int(size), int(size))
		else:
			self.size = size
	
	def __call__(self, img):
		"""
		Args:
			img (PIL.Image): Image to be cropped.
		Returns:
			PIL.Image: Cropped image.
		"""
		w, h = img.size
		th, tw = self.size
		x1 = int(round((w - tw) / 2.))
		y1 = int(round((h - th) / 2.))
		return img.crop((x1, y1, x1 + tw, y1 + th))
	
	def randomize_parameters(self):
		pass


class CornerCrop(object):
	def __init__(self, size, crop_position=None):
		self.size = size
		if crop_position is None:
			self.randomize = True
		else:
			self.randomize = False
		self.crop_position = crop_position
		self.crop_positions = ['c', 'tl', 'tr', 'bl', 'br']
	
	def __call__(self, img):
		image_width = img.size[0]
		image_height = img.size[1]
		
		if self.crop_position == 'c':
			th, tw = (self.size, self.size)
			x1 = int(round((image_width - tw) / 2.))
			y1 = int(round((image_height - th) / 2.))
			x2 = x1 + tw
			y2 = y1 + th
		elif self.crop_position == 'tl':
			x1 = 0
			y1 = 0
			x2 = self.size
			y2 = self.size
		elif self.crop_position == 'tr':
			x1 = image_width - self.size
			y1 = 0
			x2 = image_width
			y2 = self.size
		elif self.crop_position == 'bl':
			x1 = 0
			y1 = image_height - self.size
			x2 = self.size
			y2 = image_height
		elif self.crop_position == 'br':
			x1 = image_width - self.size
			y1 = image_height - self.size
			x2 = image_width
			y2 = image_height
		
		img = img.crop((x1, y1, x2, y2))
		
		return img
	
	def randomize_parameters(self):
		if self.randomize:
			self.crop_position = self.crop_positions[
				random.randint(0, len(self.crop_positions) - 1)]


class RandomHorizontalFlip(object):
	"""Horizontally flip the given PIL.Image randomly with a probability of 0.5."""
	
	def __call__(self, img):
		"""
		Args:
			img (PIL.Image): Image to be flipped.
		Returns:
			PIL.Image: Randomly flipped image.
		"""
		if self.p < 0.5:
			return img.transpose(Image.FLIP_LEFT_RIGHT)
		return img
	
	def randomize_parameters(self):
		self.p = random.random()


class MultiScaleCornerCrop(object):
	"""Crop the given PIL.Image to randomly selected size.
	A crop of size is selected from scales of the original size.
	A position of cropping is randomly selected from 4 corners and 1 center.
	This crop is finally resized to given size.
	Args:
		scales: cropping scales of the original size
		size: size of the smaller edge
		interpolation: Default: PIL.Image.BILINEAR
	"""
	
	def __init__(self, scales, interpolation=Image.BILINEAR):
		self.scales = scales
		# self.size = size
		self.interpolation = interpolation
		self.crop_positions = ['c', 'tl', 'tr', 'bl', 'br']
	
	def __call__(self, img):
		
		if self.scale == 1:
			return img
		
		crop_size = [int(img.size[0] * self.scale), int(img.size[1] * self.scale)]
		
		image_width = img.size[0]
		image_height = img.size[1]
		
		if self.crop_position == 'c':
			center_x = image_width // 2
			center_y = image_height // 2
			box_half = [crop_size[0] // 2, crop_size[1] // 2]
			x1 = center_x - box_half[0]
			y1 = center_y - box_half[1]
			x2 = center_x + box_half[0]
			y2 = center_y + box_half[1]
		elif self.crop_position == 'tl':
			x1 = 0
			y1 = 0
			x2 = crop_size[0]
			y2 = crop_size[1]
		elif self.crop_position == 'tr':
			x1 = image_width - crop_size[0]
			y1 = 0
			x2 = image_width
			y2 = crop_size[1]
		elif self.crop_position == 'bl':
			x1 = 0
			y1 = image_height - crop_size[1]
			x2 = crop_size[0]
			y2 = image_height
		elif self.crop_position == 'br':
			x1 = image_width - crop_size[0]
			y1 = image_height - crop_size[1]
			x2 = image_width
			y2 = image_height
		
		img = img.crop(
			(max(0, round(x1)), max(0, round(y1)), min(image_width, round(x2)), min(image_height, round(y2))))
		
		return img.resize([image_width, image_height], self.interpolation)
	
	def randomize_parameters(self):
		self.scale = self.scales[random.randint(0, len(self.scales) - 1)]
		self.crop_position = self.crop_positions[random.randint(0, len(self.crop_positions) - 1)]


class TwelveCrop(object):
	def __init__(self, size, vertical_flip=False):
		self.size = size
		if isinstance(size, numbers.Number):
			self.size = (int(size), int(size))
		else:
			assert len(size) == 2, "Please provide only two dimensions (h, w) for size."
			self.size = size
		self.vertical_flip = vertical_flip
	
	def __call__(self, img):
		return F.twelve_crop(img, self.size, self.vertical_flip)
	
	def randomize_parameters(self):
		pass


class RandomAffine(object):
	"""Random affine transformation of the image keeping center invariant
	Args:
		degrees (sequence or float or int): Range of degrees to select from.
			If degrees is a number instead of sequence like (min, max), the range of degrees
			will be (-degrees, +degrees). Set to 0 to desactivate rotations.
		translate (tuple, optional): tuple of maximum absolute fraction for horizontal
			and vertical translations. For example translate=(a, b), then horizontal shift
			is randomly sampled in the range -img_width * a < dx < img_width * a and vertical shift is
			randomly sampled in the range -img_height * b < dy < img_height * b. Will not translate by default.
		scale (tuple, optional): scaling factor interval, e.g (a, b), then scale is
			randomly sampled from the range a <= scale <= b. Will keep original scale by default.
		shear (sequence or float or int, optional): Range of degrees to select from.
			If degrees is a number instead of sequence like (min, max), the range of degrees
			will be (-degrees, +degrees). Will not apply shear by default
		resample ({PIL.Image.NEAREST, PIL.Image.BILINEAR, PIL.Image.BICUBIC}, optional):
			An optional resampling filter.
			See http://pillow.readthedocs.io/en/3.4.x/handbook/concepts.html#filters
			If omitted, or if the image has mode "1" or "P", it is set to PIL.Image.NEAREST.
		fillcolor (int): Optional fill color for the area outside the transform in the output image. (Pillow>=5.0.0)
	"""
	
	def __init__(self, degrees, translate=None, scale=None, shear=None, resample=False, fillcolor=0):
		if isinstance(degrees, numbers.Number):
			if degrees < 0:
				raise ValueError("If degrees is a single number, it must be positive.")
			self.degrees = (-degrees, degrees)
		else:
			assert isinstance(degrees, (tuple, list)) and len(degrees) == 2, \
				"degrees should be a list or tuple and it must be of length 2."
			self.degrees = degrees
		
		if translate is not None:
			assert isinstance(translate, (tuple, list)) and len(translate) == 2, \
				"translate should be a list or tuple and it must be of length 2."
			for t in translate:
				if not (0.0 <= t <= 1.0):
					raise ValueError("translation values should be between 0 and 1")
		self.translate = translate
		
		if scale is not None:
			assert isinstance(scale, (tuple, list)) and len(scale) == 2, \
				"scale should be a list or tuple and it must be of length 2."
			for s in scale:
				if s <= 0:
					raise ValueError("scale values should be positive")
		self.scale = scale
		
		if shear is not None:
			if isinstance(shear, numbers.Number):
				if shear < 0:
					raise ValueError("If shear is a single number, it must be positive.")
				self.shear = (-shear, shear)
			else:
				assert isinstance(shear, (tuple, list)) and len(shear) == 2, \
					"shear should be a list or tuple and it must be of length 2."
				self.shear = shear
		else:
			self.shear = shear
		
		self.resample = resample
		self.fillcolor = fillcolor
	
	@staticmethod
	def get_params(degrees, translate, scale_ranges, shears, img_size):
		"""Get parameters for affine transformation
		Returns:
			sequence: params to be passed to the affine transformation
		"""
		angle = random.uniform(degrees[0], degrees[1])
		if translate is not None:
			max_dx = translate[0] * img_size[0]
			max_dy = translate[1] * img_size[1]
			translations = (np.round(random.uniform(-max_dx, max_dx)),
			                np.round(random.uniform(-max_dy, max_dy)))
		else:
			translations = (0, 0)
		
		if scale_ranges is not None:
			scale = random.uniform(scale_ranges[0], scale_ranges[1])
		else:
			scale = 1.0
		
		if shears is not None:
			shear = random.uniform(shears[0], shears[1])
		else:
			shear = 0.0
		
		return angle, translations, scale, shear
	
	def __call__(self, img):
		"""
			img (PIL Image): Image to be transformed.
		Returns:
			PIL Image: Affine transformed image.
		"""
		ret = self.get_params(self.degrees, self.translate, self.scale, self.shear, img.size)
		return F.affine(img, *ret, resample=self.resample, fillcolor=self.fillcolor)
