import numpy as np
import scipy
from scipy import ndimage
from scipy import interpolate
import xarray as xr 
import functools

def rolling_window2d(data, roll_shape):
    #make a rolling window so that a ufunc like np.sum etc can be used on it
    #rolling_window(np.ones((4,4)),(2,2)).sum(axis=(2,3)) Should just 4 in each element
    #to to symmetric conditions along an axis use np.roll(data,axis=?)
    if len(data.shape)==2:
        shape = (data.shape[0] - roll_shape[0] + 1,) + (data.shape[1] - roll_shape[1] + 1,) + roll_shape
    elif len(data.shape)>2:
        shape = (data.shape[0] - roll_shape[0] + 1,) + (data.shape[1] - roll_shape[1] + 1,) + data.shape[2:] +roll_shape
    else:
        print('function doesnt support current invocation') 
        return
    strides = data.strides + data.strides 
    print(strides)
    return np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)

from scipy import signal
def smooth_uniform(data,input_resolution=None,output_resolution=None,npoints=None,mean=True,mode='same',boundary='symm'):
    #smothing using 2d convolution
    #by default returns mean
    #npts tells length of square 
    #if mean=False then returns sum
    if input_resolution and output_resolution:
       npts=int(output_resolution/input_resolution)
    else:
       npts=npoints
    wgt=np.ones((npts,npts))
    if mean:
       wgt=wgt/np.sum(wgt)
      
    return signal.convolve2d(data,wgt,mode,boundary)

class SAL:
    def __init__(self,fcst,obs,threshold,spatial_coords={'lat':'lat','lon':'lon'}):
        self.fcst=fcst
        self.obs=obs
        self.threshold=threshold
        latname=spatial_coords['lat']
        lonname=spatial_coords['lon']
        
        _lats=fcst[latname].values
        _lons=fcst[lonname].values

        
        _dy=abs(np.min(_lats)-np.max(_lats)) #maximum distance in y degrees
        _dx=abs(np.min(_lons)-np.max(_lons))
        
        self.amplitude_error=self.Amplitude(self.fcst,self.obs,spatial_coords)

        Xm=self.calc_X(fcst,spatial_coords)
        Xo=self.calc_X(obs,spatial_coords) 

        location_error1_Y=np.abs(Xm['CM_'+spatial_coords['lat']]-Xo['CM_'+spatial_coords['lat']])/_dy
        location_error1_X=np.abs(Xm['CM_'+spatial_coords['lon']]-Xo['CM_'+spatial_coords['lon']])/_dx
        self.__setattr__('location_error1_'+spatial_coords['lat'],location_error1_Y)
        self.__setattr__('location_error1_'+spatial_coords['lon'],location_error1_X)
        
        rm=self.calc_r(fcst,threshold,spatial_coords)
        ro=self.calc_r(obs,threshold,spatial_coords)
        
        location_error2_Y=2.0*np.abs(rm['r_'+spatial_coords['lat']]-ro['r_'+spatial_coords['lat']])/_dy
        location_error2_X=2.0*np.abs(rm['r_'+spatial_coords['lon']]-ro['r_'+spatial_coords['lon']])/_dx
        self.__setattr__('location_error2_'+spatial_coords['lat'],location_error2_Y)
        self.__setattr__('location_error2_'+spatial_coords['lon'],location_error2_X)
        
        Vmod=self.calc_V(fcst,threshold,spatial_coords)
        Vobs=self.calc_V(obs,threshold,spatial_coords)
        S=2.0*(Vmod-Vobs)/(Vmod+Vobs)
        self.structure_error=S 
        
    def __interp_func(self,vals):
        f=interpolate.interp1d(np.arange(0,len(vals)),vals)
        return f
    @staticmethod
    def Amplitude(fcst,obs,spatial_coords={'lat':'lat','lon':'lon'}):
        "Note: area averaging not done"
        "This function also handles time dimension when present"
        assert (fcst.shape==obs.shape or len(fcst.dims)>3),'''Dimenstions of fcst and obs 
                                                      should be same and not more then 3'''
        Dm=fcst.mean(dim=list(spatial_coords.keys())) #Domain average 
        Do=obs.mean(dim=list(spatial_coords.keys()))
        return (Dm-Do)/0.5*(Dm+Do)

    @staticmethod
    def calc_X(data,spatial_coords={'lat':'lat','lon':'lon'}):
       "Calculates center of mass of a field"
       "Return values center of mass are based on matrix indices and not lat and lon"
       
       if len(data.dims)>3:
           return ("Dimenstions should not more then 3 (time,lat,lon)")
       
       def _interp_func(vals):
           f=interpolate.interp1d(np.arange(0,len(vals)),vals)
           return f

       #to find order of spatial dimensions
       sdim0,sdim1=[d for d in data.dims if d in spatial_coords.values()]
       sdim0_interp_func=_interp_func(data[spatial_coords[sdim0]].values)  
       sdim1_interp_func=_interp_func(data[spatial_coords[sdim1]].values)  

       def _center_of_mass(da):
           X=ndimage.measurements.center_of_mass(da.values)
           X0=xr.DataArray(sdim0_interp_func(X[0]),dims=[]
                                   ,name='CM_'+str(sdim0))
           X1=xr.DataArray(sdim1_interp_func(X[1]),dims=[],name='CM_'+str(sdim1)) 
           return xr.merge([X0,X1])
       
       if len(data.dims)>2:
           #to find name of time dimension
           tdim=[d for d in data.dims if d not in spatial_coords.values()][0] 
           xda=data.groupby(tdim)
           return xda.apply(_center_of_mass)
       else:
           return _center_of_mass(data)
    @staticmethod
    def calc_r(data,threshold,spatial_coords={'lat':'lat','lon':'lon'}):
       """Calculate average distance between center of mass of data objects(by threshold)
          and center of mass of data field.
       """ 
       if len(data.dims)>3:
           return ("Dimenstions should not more then 3 (time,lat,lon)")
       
       def _interp_func(vals):
           f=interpolate.interp1d(np.arange(0,len(vals)),vals)
           return f

       #to find order of spatial dimensions
       sdim0,sdim1=[d for d in data.dims if d in spatial_coords.values()]
       sdim0_interp_func=_interp_func(data[spatial_coords[sdim0]].values)  
       sdim1_interp_func=_interp_func(data[spatial_coords[sdim1]].values)  


       def calc_r_2d(da,threshold=threshold,spatial_coords=spatial_coords):
           tda=np.where(da>=threshold,1,0)
           s = [[1,1,1],
                [1,1,1],
                [1,1,1]]   
          
           X=__class__.calc_X(da,spatial_coords)
    
           obj_lbls,obj_num = ndimage.label(tda,s)
           indices=list(np.arange(1,obj_num+1))

           Rn=ndimage.measurements.sum(da, labels=obj_lbls,index=indices)
           Rn_sum=np.sum(Rn)
    
           Xn=ndimage.measurements.center_of_mass(da, labels=obj_lbls,index=indices)
  
           abs_y_yn=np.array([abs(sdim0_interp_func(v[0])-X['CM_'+sdim0].values) for v in Xn])
           abs_x_xn=np.array([abs(sdim1_interp_func(v[1])-X['CM_'+sdim1].values) for v in Xn])
            
           ry=np.sum(Rn*abs_y_yn)/Rn_sum
           rx=np.sum(Rn*abs_x_xn)/Rn_sum
        
           r0=xr.DataArray([ry],coords={'threshold':[threshold]},dims={'threshold':threshold},name='r_'+str(sdim0))
           r1=xr.DataArray([rx],coords={'threshold':[threshold]},dims={'threshold':threshold},name='r_'+str(sdim1)) 
           return xr.merge([r0,r1]) #,join='inner')
       if len(data.dims)>2:
           #to find name of time dimension
           tdim=[d for d in data.dims if d not in spatial_coords.values()][0] 
           xda=data.groupby(tdim) 
           #fn=functools.partial(calc_r_2d,threshold=threshold,spatial_coords=spatial_coords)
           return xda.apply(calc_r_2d)
       else:
           return calc_r_2d(data,threshold,spatial_coords)

    @staticmethod
    def calc_V(data,threshold,spatial_coords={'lat':'lat','lon':'lon'}):
       """Calculate scaled volume of all data objects(by threshold).
       """ 
       if len(data.dims)>3:
           return ("Dimenstions should not more then 3 (time,lat,lon)")
       
       def _interp_func(vals):
           f=interpolate.interp1d(np.arange(0,len(vals)),vals)
           return f

       #to find order of spatial dimensions
       #sdim0,sdim1=[d for d in data.dims if d in spatial_coords.values()]
       #sdim0_interp_func=_interp_func(data[spatial_coords[sdim0]].values)  
       #sdim1_interp_func=_interp_func(data[spatial_coords[sdim1]].values)  

       def calc_V_2d(da,threshold=threshold):
           tda=np.where(da>=threshold,1,0)
           s = [[1,1,1],
                [1,1,1],
                [1,1,1]]   
    
           obj_lbls,obj_num = ndimage.label(tda,s)
    
           indices=list(np.arange(1,obj_num+1))
           Rn_max=ndimage.measurements.maximum(da,labels=obj_lbls,index=indices)
    
           Rn=ndimage.measurements.sum(da, labels=obj_lbls,index=indices)
    
           Rn_sum=Rn.sum()
    
           Vn=Rn/Rn_max
           V=np.sum(Rn*Vn)/Rn_sum
           return xr.DataArray([V],coords={'threshold':[threshold]},dims={'threshold':threshold},name='V')
       if len(data.dims)>2:
           #to find name of time dimension
           tdim=[d for d in data.dims if d not in spatial_coords.values()][0] 
           xda=data.groupby(tdim) 
           return xda.apply(calc_V_2d)
       else:
           return calc_V_2d(data)
               
