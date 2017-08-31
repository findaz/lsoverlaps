#!/usr/bin/env python
import os
import sys
import numpy as np
from fitsio import read_header
from astrometry.util.fits import fits_table, merge_tables
from astrometry.util.util import Tan, wcs_pv2sip_hdr
from astrometry.libkd.spherematch import cluster_radec, match_radec
from legacypipe import survey
from astropy.stats import sigma_clip
#from astropy.wcs import WCS
#from astropy.io.fits import getheader


BASSCCDS='/global/homes/f/findlay/local/overlaps/ccds-annotated-bass-overlaps.fits'
DCLSCCDS='/global/homes/f/findlay/local/overlaps/ccds-annotated-dcls-overlaps.fits'

# nominal full galdepth
nominal3={'g':24.0,'r':23.4}

# nominal single pass galdepth
# Correction for PSF to galaxy depth in 2 arcs EC seeing is small. 
nominal1={'g':23.4,'r':22.8} 

def get_wcs(hdr, tpv_to_sip=True):
    # Thanks Dstn :) 
    # https://github.com/legacysurvey/legacypipe/blob/master/py/legacypipe/cpimage.py#102
    width=hdr['NAXIS1']; height=hdr['NAXIS2']
    if not(tpv_to_sip):
        # extract wcs from fits header directly
        wcs= Tan(hdr['CRVAL1'], hdr['CRVAL2'],hdr['CRPIX1'],hdr['CRPIX2'],\
                 hdr['CD1_1'],hdr['CD1_2'],hdr['CD2_1'],hdr['CD2_2'],\
                 float(width),float(height))
    else:
        # Make sure the PV-to-SIP converter samples enough points for small
        # images
        stepsize = 0
        if min(width, height) < 600:
            stepsize = min(width, height) / 10.;
        wcs = wcs_pv2sip_hdr(hdr, stepsize=stepsize)
    # Dstn adds an offset correction i.e.
    # Correction: ccd,ccdraoff, decoff from zeropoints file  
    # Should I do this?
    return wcs
def srcs_in_wcs(wcs,ra,dec):
    ok,x,y=wcs.radec2pixelxy(ra,dec)
    return (x>=1) & (x<=wcs.get_width()) & (y>=1) & (y<wcs.get_height())

class BassDecalsOverlap(object):
    '''
    Keeps filepath records and provides various utilities
    for analyzing sources in overlapping BASS/DECaLS CCDs
    '''
    def __init__(self,bassCCD,dclsCCD,catDir=os.environ['CPCATS']):
        self.bassCCD=bassCCD
        self.dclsCCD=dclsCCD
        self.catDir=catDir
        self.bassFileName=bassCCD.image_filename
        self.dclsFileName=dclsCCD.image_filename
        self.bassCatPath=self.getCatPath(self.bassFileName)
        self.dclsCatPath=self.getCatPath(self.dclsFileName)
        self.bassCatName=self.getCatName(self.bassFileName)
        self.dclsCatName=self.getCatName(self.dclsFileName)
        self.bassImgPath=self.getImgPath(self.bassFileName)
        self.dclsImgPath=self.getImgPath(self.dclsFileName)
        self.bassSrcs=fits_table(self.bassCatPath,ext=bassCCD.image_hdu)
        self.dclsSrcs=fits_table(self.dclsCatPath,ext=dclsCCD.image_hdu)
        assert(bassCCD.filter.strip()==dclsCCD.filter.strip())
        self.filter=bassCCD.filter.strip()
        self.bassPass=int(self.bassCCD.get('object').strip()[-1])
        self.dclsPass=self.dclsCCD.tilepass
    def getCatName(self,fileName):
        return os.path.basename(fileName).replace('.fits.fz','.cat.fits').strip()
    def getCatPath(self, fileName):
        catNme=self.getCatName(fileName)
        if 'decam' in fileName:
            cpPath=self.dclsFileName.split('/')[1].strip()
            prPath=os.path.join(self.catDir, 'decals')
            return os.path.join(prPath,cpPath,catNme)
        elif '90prime' in fileName:
            cpPath=self.bassFileName.split('/')[1].strip()
            prPath=os.path.join(self.catDir, 'bass')
            return os.path.join(prPath,cpPath,catNme)
    def getImgPath(self,fileName):
        return self.getCatPath(fileName).replace('cat.fits','fits')
    def get_bass_hdr(self):
        return read_header(self.bassImgPath,ext=self.bassCCD.image_hdu)
        #return getheader(self.bassImgPath,self.bassCCD.image_hdu)
    def get_dcls_hdr(self):
        return read_header(self.dclsImgPath,ext=self.dclsCCD.image_hdu)
        #return getheader(self.dclsImgPath,self.dclsCCD.image_hdu)
    def get_bass_primhdr(self):
        return read_header(self.bassImgPath,ext=0)
        #return getheader(self.bassImgPath,self.bassCCD.image_hdu)
    def get_dcls_primhdr(self):
        return read_header(self.dclsImgPath,ext=0)
        #return getheader(self.dclsImgPath,self.dclsCCD.image_hdu)
    def srcs_in_bass_wcs(self,ra,dec):
        '''
        Given a list of celestial coordinates (ra,dec in decimal degrees)
        return a bool array set True for sources falling within the BASS 
        CCD wcs footprint, False otherwise.
        '''
        hdr=self.get_bass_hdr()
        wcs=get_wcs(hdr)
        return srcs_in_wcs(wcs,ra,dec)
    def srcs_in_dcls_wcs(self,ra,dec):
        '''
        Given a list of celestial coordinates (ra,dec in decimal degrees)
        return a bool array set True for sources falling within the DECaLS 
        CCD wcs footprint, False otherwise.
        '''
        hdr=self.get_dcls_hdr()
        wcs=get_wcs(hdr)
        return srcs_in_wcs(wcs,ra,dec)
    def bricks_in_overlap(self,surveyObj,margin=10.):
        '''
        Return a brick catalog for all bricks falling within the overlap
        region between the BASS and DECaLS CCDs
        '''
        raBass,decBass=self.bassSrcs.alpha_j2000,self.bassSrcs.delta_j2000
        raDcls,decDcls=self.dclsSrcs.alpha_j2000,self.dclsSrcs.delta_j2000
        I=self.srcs_in_bass_wcs(raDcls,decDcls)
        J=self.srcs_in_dcls_wcs(raBass,decBass)
        alphas=np.append(raDcls[I],raBass[J])
        deltas=np.append(decDcls[I],decBass[J])
        ralo,rahi=alphas.min(),alphas.max()
        declo,dechi=deltas.min(),deltas.max()
        # wrap around
        if rahi-ralo>180:
            rahi,ralo=ralo,rahi
        ralo-=margin/3600.
        declo-=margin/3600.
        rahi+=margin/3600.
        dechi+=margin/3600.
        brks=surveyObj.get_bricks()
        indx=surveyObj.bricks_touching_radec_box(brks,ralo,rahi,declo,dechi)
        brks=brks[indx]
        return brks
    def brick_cat_from_overlap(self,surveyObj,margin=10.,purge_duplicates=True):
        '''
        Return a catalog of sources within bricks touching the overlap region 
        between the BASS and DECaLS CCDs. Does not cut sources to the overlap 
        region. 
        If purge_duplicates=True then duplicates (i.e. common sources from
        overlapping bricks) are removed based on depth.
        '''
        brks=self.bricks_in_overlap(surveyObj,margin=margin)
        if not len(brks):
            return None
        tractorDir=surveyObj.find_file('tractor').split('/')
        tractorDir='/'.join(tractorDir[0:-2]) 
        tabs=[]
        # merge brick catalogs
        for brk in brks:
            path=os.path.join(tractorDir,brk.brickname[0:3],'tractor-'+brk.brickname+'.fits')
            tabs.append(fits_table(path))
        if len(tabs)==1: return tabs[0]
        tab=merge_tables(tabs)
        if not(purge_duplicates):
            return tab
        # find groups of connected sources within a small radius
        # this is obviously not an exact method!
        grps=cluster_radec(tab.ra,tab.dec,0.3/3600.)
        keep=np.ones(len(tab),dtype=bool)
        for grp in grps:
            brknms=tab.brickname[grp]
            # TODO a small number of srcs in the same brick will be
            # selected as duplicates! 
            if len(grp)>2:
                # if there are more than 2 sources in a group then call
                # the closest 2 duplicates
                ra=tab.ra[grp]; dec=tab.dec[grp]
                m1,m2,d12=match_radec(ra,dec,ra,dec,0.3/3600.,notself=True,nearest=True)
                i=np.argmin(d12)
                inxa=m1[i];inxb=m2[i]
                # chuck the one with larger ivar
                try:
                    dpth=np.array([tab.get('psfdepth_'+self.filter)[inxa],
                                   tab.get('psfdepth_'+self.filter)[inxb]])
                except KeyError:
                    b='ugrizY'.find(self.filter)
                    dpth=np.array([tab.get('decam_depth')[inxa,b],
                                   tab.get('decam_depth')[inxb,b]])
                j=np.argmin(dpth)
                keep[grp[np.array([inxa, inxb])[j]]]=False
            else:
                try:
                    dpth=tab.get('psfdepth_'+self.filter)[grp]
                except KeyError:
                    b='ugrizY'.find(self.filter)
                    dpth=np.array([tab.get('decam_depth')[grp[0],b],
                                   tab.get('decam_depth')[grp[1],b]])
                j=np.argmin(dpth)
                keep[grp[j]]=False
        if np.any(~keep):
            return tab[keep]
        return tab
'''
    def get_tpv_wcs(self,hdr):
        #w=WCS(hdr)
        w = WCS(naxis=2)
        w.wcs.crpix = [hdr['CRPIX1'], hdr['CRPIX2']]
        w.wcs.crval = [hdr['CRVAL1'], hdr['CRVAL2']]
        w.wcs.ctype = [hdr['CTYPE1'], hdr['CTYPE2']]
        w.wcs.cd = [[hdr['CD1_1'], hdr['CD1_2']],[hdr['CD2_1'], hdr['CD2_2']]]
        pv=[]
        for key in hdr:
            if 'PV1_' in key:
                if int(key[key.find('_')+1:])>4: continue
                pv.append(( int(key[2]),int(key[key.find('_')+1:]), hdr[key]))
            if 'PV2_' in key:
                if int(key[key.find('_')+1:])>4: continue
                pv.append(( int(key[2]),int(key[key.find('_')+1:]), hdr[key]))
            w.wcs.set_pv(pv)
        return w
    def get_bass_wcs(self):
        hdr=self.get_bass_hdr()
        return self.get_tpv_wcs(hdr)
    def get_dcls_wcs(self):
        hdr=self.get_dcls_hdr()
        return self.get_tpv_wcs(hdr)
    def sources_in_bass_wcs(self, ra, dec):
        hdr=self.get_bass_hdr() # NAXISi keys not available from wcs object
        wcs=self.get_bass_wcs()
        x,y=wcs.all_world2pix(ra,dec,1)
        return (x>=1) & (x<=hdr['NAXIS1']) & (y>=1) & (y<=hdr['NAXIS2'])
    def sources_in_dcls_wcs(self, ra, dec):
        hdr=self.get_dcls_hdr() # NAXISi keys not available from wcs object
        wcs=self.get_dcls_wcs()
        x,y=wcs.all_world2pix(ra,dec,1)
return (x>=1) & (x<=hdr['NAXIS1']) & (y>=1) & (y<=hdr['NAXIS2']) 
'''

