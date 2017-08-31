#!/usr/bin/env python
import healpy
import os
import numpy as np
from collections import OrderedDict
from astrometry.util.fits import fits_table, merge_tables
from astrometry.libkd.spherematch import match_radec

class NoCalibrationStars(Exception):
	def __init__(self, msg=None):
		if msg is None: pass
		else: print msg

def read_chunks(ra, dec):

	chunkdir = os.environ['PS1CHUNKS']
	phi = np.radians(ra)
	theta = np.radians(90-dec)
	pixels = healpy.pixelfunc.ang2pix(32,theta,phi)
	pixels = np.unique(pixels)
	cat = []
	for pix in pixels:
		fn = chunkdir+'ps1-%05d.fits'%pix
		try: 
			cat.append( fits_table(fn, ext=1) )
		except ValueError: 
			continue
		except IOError:
			continue
	if not cat: 
		raise NoCalibrationStars
	return merge_tables(cat)

def read_stars(ra, dec, filt='r', mlo=None, mhi=None):
	
	j = 'gr'.find(filt) # index into PS1 arrays

	cat = read_chunks(ra, dec)
	
	gmi = cat.median[:,0] - cat.median[:,2]
	margin = 20./3600
	
	I = (gmi>0.4) & (gmi<2.7) & \
	    (cat.nmag_ok[:,0] >= 1)  & \
	    (cat.nmag_ok[:,2] >= 1)  & \
	    (cat.ra<ra.max()+margin) & \
	    (cat.ra>ra.min()-margin) & \
	    (cat.dec<dec.max()+margin) & \
	    (cat.dec>dec.min()-margin)

	if mlo is not None:
		J = cat.median[:,j]>mlo
	else:
		J = np.ones(len(cat),dtype=bool)
	if mhi is not None:
		K = cat.median[:,j]<mhi
	else:
		K = np.ones(len(cat),dtype=bool)

	return cat[I & J & K]

def cross_match(cat, stars=True, rad=2.0, **kwargs):
    try:
        if stars:
            objs = read_stars(cat.alpha_j2000, cat.delta_j2000, 
                              filt=kwargs.get('filt','r'), 
                              mlo=kwargs.get('mlo'),
                              mhi=kwargs.get('mhi'))
        else:
            objs = read_chunks(cat.alpha_j2000, cat.delta_j2000)
    except NoCalibrationStars: pass
    #if len(objs)<10: continue
    m1, m2, d12 = match_radec(objs.ra, objs.dec, cat.alpha_j2000, cat.delta_j2000, rad/3600.0, nearest=True)
    if not(m1.size): pass
    cat = cat[m2]; objs=objs[m1]
    cat.add_columns_from(objs)
    cat.angdist = d12*3600.
    return cat

clrcoeff={
        'bassg':[-6.720961301800471022e-03,
                 +9.582588862218254649e-03,
                 +6.630438116195029596e-02],
        'bassr':[-5.626193038947120521e-03,
                 +1.099970217352156179e-02,
                 -4.835552821849404409e-02],
        # Eddie's coeffs
        #'dclsg':[+0.00613,
        #        -0.01028,
        #        -0.03604,
        #        -0.00062],
        #'dclsr':[+0.01140,
        #         -0.03222,
        #         +0.08435,
        #         -0.00495],
        #'dclsz':[+0.00898,
        #         -0.02824,
        #         +0.07690,
        #         -0.02583]
        # Original coeffs
        'dclsg':[+0.00000,
                 -0.04709,
                 -0.00084,
                 +0.00340], 
        'dclsr':[+0.00000,
                 +0.09939,
                 -0.04509,
                 +0.01488],
        'dclsz':[+0.00000,
                 +0.13404, 
                 -0.06591, 
                 +0.01695]
}
	
def ps_to_bass(mtchCat,filt):
    
    filt = filt.strip()[-1] # bokr -> r
    j = 'gr'.find(filt) # index into PS1 arrays
    
    # psone to bok color coeffs
    coeffs=clrcoeff['bass'+filt]
    # convert psone to bok
    psmag = mtchCat.median[:,j]
    psgi = mtchCat.median[:,0]-mtchCat.median[:,2]
    #psgi = np.diff(mtchCat.median[:,[2,0]],axis=1).squeeze()
    psbokmag = psmag + np.polyval(coeffs,psgi)
	
    return psbokmag

def ps_to_dcls(mtchCat,filt):

    filt = filt.strip()[-1] # bokr -> r
    j = 'gr'.find(filt) # index into PS1 arrays
    
    # psone to dcls color coeffs
    coeffs=clrcoeff['dcls'+filt]
    # convert psone to bok
    psmag = mtchCat.median[:,j]
    psgi = mtchCat.median[:,0]-mtchCat.median[:,2]
    colorterm = -(coeffs[0] + coeffs[1]*psgi + coeffs[2]*psgi**2 + coeffs[3]*psgi**3)
    psdclsmag = psmag + colorterm 
	
    return psdclsmag

