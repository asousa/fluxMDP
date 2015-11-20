import ephem
import math
import datetime
from coordinate_structure import coordinate_structure

# 11.2015 -- Modifying to use coordinate structure

class Satellite(object):
  def __init__(self,tle1, tle2, name): 
    self.tle_rec = ephem.readtle(name, tle1, tle2)
    self.curr_time = None
    self.name = name
    #self.coords = None    # Long, Lat! XY on a map, but isn't pleasant to say out loud.
    self.coords = coordinate_structure()
  def compute(self,plotdate):
    self.tle_rec.compute(plotdate)
    self.curr_time = plotdate
    lat = (180.0/math.pi)*self.tle_rec.sublat
    lon = (180.0/math.pi)*self.tle_rec.sublong
    alt = self.tle_rec.elevation*1e-3  #kilometers
    self.coords.set_coords(lat, lon, alt, "geographic")
    #self.coords = [(180.0/math.pi)*self.tle_rec.sublong, (180.0/math.pi)*self.tle_rec.sublat]

  def coords_at(self,plotdate):
    self.tle_rec.compute(plotdate)
    return self.coords
  
  #def coords(self):
  #  return [(180.0/math.pi)*self.tle_rec.sublong, (180.0/math.pi)*self.tle_rec.sublat]

  #def curr_time(self):
  #  return self.datetime
