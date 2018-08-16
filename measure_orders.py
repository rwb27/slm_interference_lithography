import pyqtgraph as pg
import numpy as np

class POIManager(object):
	def __init__(self, camera=None):
		super(POIManager, self).__init__()
		if camera is not None:
			self.camera = camera
		self.box_pen = pg.mkPen("g", width=2) # Boxes are green by default
		
	box_pen = None
	
	_poi_list = None
	@property
	def positions(self):
		"""A list of (x,y) positions the user has clicked."""
		if self._poi_list is None:
			self._poi_list = []
		return self._poi_list
	@positions.setter
	def positions(self, newlist):
		self._poi_list = newlist
		self._poi_list_updated()
	def append_point(self, x, y):
		"""Add a point of interest``"""
		self._poi_list.append((x,y))
		self._poi_list_updated()
	def reset(self):
		"""Empty the POI list"""
		self.positions = []
		
	_poi_labels = None
	@property
	def labels(self):
		"""Labels for each point.  
		
		NB this list must have the same length as the number of POIs"""
		# This should not be None, that's inconvenient (len(None) is a problem)
		if self._poi_labels is None:
			self._poi_labels = []
		# If there are too many labels, trim the list
		if len(self._poi_labels) > len(self.positions):
			self._poi_labels = self._poi_labels[:len(self.positions)]
		# If there are too few labels, expand the list with numbers as strings
		for i in range(len(self.positions)):
			if i >= len(self._poi_labels):
				self._poi_labels.append("POI {}".format(i))
		# Now we should have a list that's the right length - return it
		return self._poi_labels
	@labels.setter
	def labels(self, new_labels):
		if len(new_labels) != len(self.positions):
			raise ValueError("There must be one lable per POI in the list.")
		self._poi_labels = new_labels
		self._poi_list_updated()
			
	@property
	def slices(self):
		"""An iterable of slices that will extract the regions of interest from an image"""
		w, h = self.box_size
		for x, y in self.positions:
			yield (slice(x-w//2, x-w//2+w), slice(y-h//2, y-h//2+h))
		
	_camera = None
	@property
	def camera(self):
		"""The camera that we're taking clicks from"""
		#TODO: this should probably be a weakref
		return self._camera
	@camera.setter
	def camera(self, camera):
		#TODO: remove the callback from the old camera if present
		self._camera = camera
		self.camera.set_legacy_click_callback(self.append_point)
		self._poi_list_updated()
		
	_box_size = (100,100)
	@property
	def box_size(self):
		"""The size of boxes drawn around the points (centred on the point)"""
		return self._box_size
	@box_size.setter
	def box_size(self, new_size):
		assert len(new_size)==2, "Box size is a 2-element tuple"
		self._box_size = new_size
		self._poi_list_updated()
		
	def _poi_list_updated(self):
		"""This is called whenever the list is updated"""
		if self.camera is not None:
			for pw in self.camera._preview_widgets:
				try:
					# Remove any boxes we've previously added
					for item in pw.poi_list_graphics_items:
						pw.view_box.removeItem(item)
				except AttributeError:
					# If there's no poi_list_boxes attribute, don't worry.
					pass
				# For each POI, draw a box around it
				h, w = self.box_size # NB these are swapped deliberately to match x and y
				pw.poi_list_graphics_items = []
				for (y, x), label in zip(self.positions, self.labels):
					# NB x and y are swapped deliberately - they correspond to the indices
					# of the image arrays - which are NOT x and y in the canvas :(
					b = pg.QtGui.QGraphicsRectItem(x-w/2, y-h/2, w, h)
					b.setPen(self.box_pen)
					pw.view_box.addItem(b)
					pw.poi_list_graphics_items.append(b)
					t = pg.TextItem(label, color=self.box_pen.color(), anchor=(0.5,0))
					t.setPos(x,y+h/2)
					pw.view_box.addItem(t)
					pw.poi_list_graphics_items.append(t)
					
def sum_rois(image, poi_manager, **kwargs):
	"""For each point of interest defined, return the total brightness.
	
	Keyword arguments are passed to np.sum.  To sum only X and Y (i.e. preserve
	colour channels) use ``axis=(0,1)`, and you may wish to specify `dtype=float`
	"""
	brightnesses = []
	for s in poi_manager.slices:
		brightnesses.append(np.sum(image[s], **kwargs))
	return brightnesses