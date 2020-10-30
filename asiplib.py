# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18 10:22:58 2015

@author: DMH
"""
import numpy as np
from numpy.lib.stride_tricks import as_strided as ast
from osgeo import gdal, ogr, osr

def get_gcps(fil, aoi=[]):
    if 'sar_grid_line' in fil.variables:
        lines = np.array(fil.variables.get('sar_grid_line'))
        samps = np.array(fil.variables.get('sar_grid_sample'))
        lats = np.array(fil.variables.get('sar_grid_latitude'))
        lons = np.array(fil.variables.get('sar_grid_longitude'))
        hgts = np.array(fil.variables.get('sar_grid_height'))
    elif 'line' in fil.variables:
        lines = np.array(fil.variables.get('line')).astype(float)
        samps = np.array(fil.variables.get('sample')).astype(float)
        lats = np.array(fil.variables.get('lat')).astype(float)
        lons = np.array(fil.variables.get('lon')).astype(float)
        hgts = np.zeros(lons.size).astype(float)
    else:
        ValueError('Unknown format')

    if aoi:
        inds = np.logical_and(lines>=aoi[0],lines<aoi[1])
        lines, samps, lats, lons, hgts = (lines[inds],samps[inds],lats[inds],
                                          lons[inds],hgts[inds])
        inds = np.logical_and(samps>=aoi[2],samps<aoi[3])
        lines, samps, lats, lons, hgts = (lines[inds],samps[inds],lats[inds],
                                          lons[inds],hgts[inds])
        lines = lines - aoi[0]
        samps = samps - aoi[2]

    gcps = ()
    for i in range(len(lines)):
        x, y, z, pix, lin = lons[i],lats[i],hgts[i],samps[i],lines[i]
        gcps = gcps + (gdal.GCP(x, y, z, pix, lin, '', str(i)),)
    
    return(gcps)

def save_to_tiff(out_gtiff_filename, data, gcps, etype=6, multiband=False, no_data_value=None):
    if len(data.shape)<3:
        bands = 1
        data = np.expand_dims(data, -1)
    else:
        bands = data.shape[2]
    
    gtiff = gdal.GetDriverByName('GTiff').Create(out_gtiff_filename, 
                                                data.shape[1], 
                                                data.shape[0],
                                                bands,
                                                eType=etype) 
    # GDT_Unknown = 0,GDT_Byte = 1,GDT_UInt16 = 2,GDT_Int16 = 3,GDT_UInt32 = 4,
    # GDT_Int32 = 5,GDT_Float32 = 6,GDT_Float64 = 7,GDT_CInt16 = 8,GDT_CInt32 = 9,
    # GDT_CFloat32 = 10,GDT_CFloat64 = 11,GDT_TypeCount = 12
    
    for i in range(bands):
        gtiff.GetRasterBand(i+1).WriteArray(data[:,:,i])
        if no_data_value:
            gtiff.GetRasterBand(i+1).SetNoDataValue(no_data_value)

    spr_gcp = osr.SpatialReference()
    spr_gcp.ImportFromEPSG(4326)

    gtiff.SetGCPs(gcps, spr_gcp.ExportToWkt())


def maskeddata_to_patches_v2(fil, ws, ss=None, rm_swath=0, inc=False, nersc=''):
    wh, ww = ws
    data = []
                     
    """ Add the data components for conv prediction """
    data.append(np.ma.getdata(fil[nersc+'sar_primary'][:]))
    data.append(np.ma.getdata(fil[nersc+'sar_secondary'][:]))
    data.append(np.ma.getdata(fil['distance_map'][:]))
    if inc:
        # Include incidence data.
        inca = np.ma.getdata(
            fil['sar_incidenceangles'][:])
        data.append(np.repeat(inca.reshape(
            (1, inca.size)), data[0].shape[0], axis=0))
        
    mask = np.logical_or(np.ma.getmask(fil[nersc+'sar_primary'][:]), np.ma.getmask(fil[nersc+'sar_secondary'][:]))
    mask = np.logical_or(mask, np.ma.getdata(fil['distance_map'][:])==0)
    mask[:,:rm_swath] = True
    if 'polygon_icechart' in fil.variables:
        data.append(np.ma.getdata(fil['polygon_icechart'][:]))
        icemask = np.ma.getmask(fil['polygon_icechart'][:]) 
        mask = np.logical_or(mask, icemask)
        

    """ Calculate which pixels to mask out. Must be last entry in data """
    rmdata = np.zeros(data[0].shape)
    rmdata[mask] = 1
    data.append(rmdata)
    
    data = np.dstack(data).astype(np.float32)
    
    ws = (wh,ww,data.shape[-1])
    if ss!=None:
        ss = (ss[0],ss[1],data.shape[-1])
    
    strides = sliding_window(data,ws,ss)
    patches = strides.reshape([np.prod(strides.shape[:3])] + list(strides.shape[3:]))
    patch_isnan = patches[:,:,:,-1:].sum(axis=(1,2,3))
    del(data)
    return(patches, patch_isnan)

def norm_shape(shape):
    '''
    Normalize numpy array shapes so they're always expressed as a tuple, 
    even for one-dimensional shapes.
     
    Parameters
        shape - an int, or a tuple of ints
     
    Returns
        a shape tuple
    '''
    try:
        i = int(shape)
        return (i,)
    except TypeError:
        # shape was not a number
        pass
 
    try:
        t = tuple(shape)
        return t
    except TypeError:
        # shape was not iterable
        pass
     
    raise TypeError('shape must be an int, or a tuple of ints')
 
def sliding_window(a,ws,ss = None,flatten = False):
    '''
    Return a sliding window over a in any number of dimensions
     
    Parameters:
        a  - an n-dimensional numpy array
        ws - an int (a is 1D) or tuple (a is 2D or greater) representing the size 
             of each dimension of the window
        ss - an int (a is 1D) or tuple (a is 2D or greater) representing the 
             amount to slide the window in each dimension. If not specified, it
             defaults to ws.
        flatten - if True, all slices are flattened, otherwise, there is an 
                  extra dimension for each dimension of the input.
     
    Returns
        an array containing each n-dimensional window from a
    '''
     
    if None is ss:
        # ss was not provided. the windows will not overlap in any direction.
        ss = ws
    ws = norm_shape(ws)
    ss = norm_shape(ss)
     
    # convert ws, ss, and a.shape to numpy arrays so that we can do math in every 
    # dimension at once.
    ws = np.array(ws)
    ss = np.array(ss)
    shape = np.array(a.shape)
     
     
    # ensure that ws, ss, and a.shape all have the same number of dimensions
    ls = [len(shape),len(ws),len(ss)]
    if 1 != len(set(ls)):
        raise ValueError(\
        'a.shape, ws and ss must all have the same length. They were %s' % str(ls))
     
    # ensure that ws is smaller than a in every dimension
    if np.any(ws > shape):
        raise ValueError(\
        'ws cannot be larger than a in any dimension.\
 a.shape was %s and ws was %s' % (str(a.shape),str(ws)))
     
    # how many slices will there be in each dimension?
    newshape = norm_shape(((shape - ws) // ss) + 1)
    # the shape of the strided array will be the number of slices in each dimension
    # plus the shape of the window (tuple addition)
    newshape += norm_shape(ws)
    # the strides tuple will be the array's strides multiplied by step size, plus
    # the array's strides (tuple addition)
    newstrides = norm_shape(np.array(a.strides) * ss) + a.strides
    strided = ast(a,shape = newshape,strides = newstrides)
    if not flatten:
        return strided
     
    # Collapse strided so that it has one more dimension than the window.  I.e.,
    # the new array is a flat list of slices.
    meat = len(ws) if ws.shape else 0
    firstdim = (np.product(newshape[:-meat]),) if ws.shape else ()
    dim = firstdim + (newshape[-meat:])
    # remove any dimensions with size 1
    dim = filter(lambda i : i != 1,dim)
    return strided.reshape(dim)
