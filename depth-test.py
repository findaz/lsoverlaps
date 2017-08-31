#!/usr/bin/env python
import os
import sys
import numpy as np
from fitsio import read_header
from astrometry.util.fits import fits_table, merge_tables
from astrometry.libkd.spherematch import cluster_radec, match_radec
from astropy.stats import sigma_clip
from overlaps import *
import pscal

def zeropoints(brkTab,ccdTab,sn=10,psf=True):
    try:
        assert(len(brkTab)==len(ccdTab))
    except AssersionError("tables must be matched element wise"):
        return None
    if psf:
        I=brkTab.type.view(np.chararray).strip()=="PSF"
    else:
        I=np.ones(len(brkTab), dtype=bool)
    try:
        brkFlux=brkTab.get('FLUX_'+overlap.filter.upper())
        brkIvar=brkTab.get('FLUX_IVAR_'+overlap.filter.upper())
    except KeyError:
        i='ugrizY'.find(overlap.filter)
        brkFlux=brkTab.decam_flux[:,i]
        brkIvar=brkTab.decam_flux_ivar[:,i]
    J=brkFlux*np.sqrt(brkIvar)>=10.
    K=ccdTab.flux_psf/ccdTab.fluxerr_psf>=10.
    brkFlux=brkFlux[I & J & K]
    ccdTab=ccdTab[I & J & K]
    dmag=sigma_clip(2.5*np.log10(ccdTab.flux_psf)+(22.5-2.5*np.log10(brkFlux)), iters=None)
    return dmag.mean(),dmag.std()

if __name__ == '__main__':

    # compare SExtractor photometry to tractor photometry
    bassCCDs=fits_table(BASSCCDS)
    dclsCCDs=fits_table(DCLSCCDS)
    #hDclsBrkTot=0.0; hDclsOvrTot=0.0;
    #hBassBrkTot=0.0; hBassOvrTot=0.0;

    for iccd, bassCCD in enumerate(bassCCDs):
        fout='overlap-'+str(iccd).zfill(4)+'.fits'
        if os.path.isfile(fout): continue
        dclsCCD=dclsCCDs[iccd]
        if '_oki_' in dclsCCD.image_filename: 
            # print dclsCCD.image_filename, 'not found'
            continue
        try:
            overlap=BassDecalsOverlap(bassCCD,dclsCCD)
        except:
            continue
        surveyDR3=survey.LegacySurveyData(survey_dir='/project/projectdirs/cosmo/data/legacysurvey/dr3.1/')
        # return DR3 tractor brick catalogs for the overlap region
        try:
            brkCat=overlap.brick_cat_from_overlap(surveyDR3)
        except IOError:
            continue
        
        # cut to PSF and single pass depth
        i='ugrizY'.find(overlap.filter)
        I=(brkCat.type.view(np.chararray).strip()=="PSF") & \
            (brkCat.decam_flux[:,i]>0.0)#=10**(-0.4*(nominal1[overlap.filter]+1.0-22.5)))

        # find brick objects in CCD footprints
        J=overlap.srcs_in_bass_wcs(brkCat.ra,brkCat.dec)
        K=overlap.srcs_in_dcls_wcs(brkCat.ra,brkCat.dec)
        bassBrk=brkCat[I & J]
        dclsBrk=brkCat[I & K]

        # match brk to CCDs
        b1,b2,b12=match_radec(bassBrk.ra,bassBrk.dec,overlap.bassSrcs.alpha_j2000,overlap.bassSrcs.delta_j2000,1.0/3600.,nearest=True)
        d1,d2,d12=match_radec(dclsBrk.ra,dclsBrk.dec,overlap.dclsSrcs.alpha_j2000,overlap.dclsSrcs.delta_j2000,1.0/3600.,nearest=True)

        # let's do completeness again but at 5-sigma
        bvalid=np.in1d(b2,np.flatnonzero( (overlap.bassSrcs.flags==0) & (overlap.bassSrcs.flux_psf/overlap.bassSrcs.fluxerr_psf>=5.0) ))
        dvalid=np.in1d(d2,np.flatnonzero( (overlap.dclsSrcs.flags==0) & (overlap.dclsSrcs.flux_psf/overlap.dclsSrcs.fluxerr_psf>=5.0) ))
        B=np.ones(len(bassBrk),dtype=bool)
        B[b1[~bvalid]]=False
        D=np.ones(len(dclsBrk),dtype=bool)
        D[d1[~dvalid]]=False
        magBins=np.linspace(18.,nominal1[overlap.filter]+1.0,41)
        hBassBrk5,eBassBrk5=np.histogram(22.5-2.5*np.log10(bassBrk.decam_flux[np.flatnonzero(B), i]), bins=magBins)
        hBassOvr5,eBassOvr5=np.histogram(22.5-2.5*np.log10(bassBrk.decam_flux[b1[bvalid],i]), bins=magBins)
        bassCmp5=hBassOvr5.astype(float)/hBassBrk5.astype(float)
        hDclsBrk5,eDclsBrk5=np.histogram(22.5-2.5*np.log10(dclsBrk.decam_flux[np.flatnonzero(D), i]), bins=magBins)
        hDclsOvr5,eDclsOvr5=np.histogram(22.5-2.5*np.log10(dclsBrk.decam_flux[d1[dvalid],i]), bins=magBins)
        dclsCmp5=hDclsOvr5.astype(float)/hDclsBrk5.astype(float)

        # identify unflagged sources
        bvalid=np.in1d(b2,np.flatnonzero(overlap.bassSrcs.flags==0))
        dvalid=np.in1d(d2,np.flatnonzero(overlap.dclsSrcs.flags==0))
        
        # all brick sources except those flagged in CCDs
        B=np.ones(len(bassBrk),dtype=bool)
        B[b1[~bvalid]]=False
        D=np.ones(len(dclsBrk),dtype=bool)
        D[d1[~dvalid]]=False

        # completeness
        #magBins=np.arange(18,nominal1[overlap.filter]+1.0,0.5)
        hBassBrk,eBassBrk=np.histogram(22.5-2.5*np.log10(bassBrk.decam_flux[np.flatnonzero(B), i]), bins=magBins)
        hBassOvr,eBassOvr=np.histogram(22.5-2.5*np.log10(bassBrk.decam_flux[b1[bvalid],i]), bins=magBins)
        bassCmp=hBassOvr.astype(float)/hBassBrk.astype(float)
        #hBassBrkTot+=hBassBrk
        #hBassOvrTot+=hBassOvr

        hDclsBrk,eDclsBrk=np.histogram(22.5-2.5*np.log10(dclsBrk.decam_flux[np.flatnonzero(D), i]), bins=magBins)
        hDclsOvr,eDclsOvr=np.histogram(22.5-2.5*np.log10(dclsBrk.decam_flux[d1[dvalid],i]), bins=magBins)
        dclsCmp=hDclsOvr.astype(float)/hDclsBrk.astype(float)
        #hDclsBrkTot+=hDclsBrk
        #hDclsOvrTot+=hDclsOvr

        bassOvr=overlap.bassSrcs[b2[bvalid]]
        dclsOvr=overlap.dclsSrcs[d2[dvalid]]
        bassBrk=bassBrk[b1[bvalid]]
        dclsBrk=dclsBrk[d1[dvalid]]

        # bass zero points
        bassPS=pscal.cross_match(bassOvr[bassOvr.flux_aper[:,2]>0.0],filt=overlap.filter,mlo=16.,mhi=20.)
        psbokmag=pscal.ps_to_bass(bassPS,overlap.filter)
        bokmag = -2.5*np.log10(bassPS.flux_aper[:,2])
        psgi=bassPS.median[:,0]-bassPS.median[:,2]
        I=(psgi>=0.4) & (psgi<=2.7)
        dmag = sigma_clip(psbokmag[I]-bokmag[I], sigma=2.5, iters=None)
        bassZpt,bassZptRms = dmag.mean(), dmag.std()

        # dcls zero points
        dclsPS=pscal.cross_match(dclsOvr[dclsOvr.flux_aper[:,2]>0.0],filt=overlap.filter,mlo=16.,mhi=20.)
        psdclsmag=pscal.ps_to_dcls(dclsPS,overlap.filter)
        # dcls pixel units are adu so correct for exptime    
        dclsTexp=overlap.get_dcls_primhdr()['EXPTIME']
        dclsmag = -2.5*np.log10(dclsPS.flux_aper[:,2]/dclsTexp)
        psgi=dclsPS.median[:,0]-dclsPS.median[:,2]
        I=(psgi>=0.4) & (psgi<=2.7)
        dmag = sigma_clip(psdclsmag[I]-dclsmag[I], sigma=2.5, iters=None)
        dmagmed=np.median(dmag.data[~dmag.mask])
        dclsZpt,dclsZptRms = dmag.mean(), dmag.std()
        
        # populate tables with mags/depths
        bassOvr.type=bassBrk.type
        dclsOvr.type=bassBrk.type
        bassOvr.set(overlap.filter+'_mag',np.zeros(len(bassOvr)))
        bassOvr.set(overlap.filter+'_depth',np.zeros(len(bassOvr)))
        I=(bassOvr.flux_psf>0.) & (bassOvr.fluxerr_psf>0.)
        bassOvr.get(overlap.filter+'_mag')[I]=bassZpt-2.5*np.log10(bassOvr.flux_psf[I])
        bassOvr.get(overlap.filter+'_depth')[I]=bassZpt-2.5*np.log10(5*bassOvr.fluxerr_psf[I])
        dclsOvr.set(overlap.filter+'_mag',np.zeros(len(dclsOvr)))
        dclsOvr.set(overlap.filter+'_depth',np.zeros(len(dclsOvr)))
        J=(dclsOvr.flux_psf>0.) & (dclsOvr.fluxerr_psf>0.)
        dclsOvr.get(overlap.filter+'_mag')[J]=dclsZpt-2.5*np.log10(dclsOvr.flux_psf[J]/dclsTexp)
        dclsOvr.get(overlap.filter+'_depth')[J]=dclsZpt-2.5*np.log10(5*dclsOvr.fluxerr_psf[J]/dclsTexp)
        
        K=bassBrk.decam_flux[:,i]>=10**(-0.4*(20.0-22.5))
        L=dclsBrk.decam_flux[:,i]>=10**(-0.4*(20.0-22.5))
        # mag offsets
        dmagBass=sigma_clip((22.5-2.5*np.log10(bassBrk.decam_flux[np.flatnonzero(I & K),i])) - bassOvr.get(overlap.filter+'_mag')[I & K], iters=None)
        dmagDcls=sigma_clip((22.5-2.5*np.log10(dclsBrk.decam_flux[np.flatnonzero(J & L),i])) - dclsOvr.get(overlap.filter+'_mag')[J & L], iters=None)
        offBass,offDcls=dmagBass.mean(), dmagDcls.mean()

        bassDepth=sigma_clip(bassOvr.get(overlap.filter+'_depth')[I], iters=None).mean()
        dclsDepth=sigma_clip(dclsOvr.get(overlap.filter+'_depth')[J], iters=None).mean()
        res=fits_table()
        res.det_brk_bass=np.array([hBassBrk])
        res.det_ccd_bass=np.array([hBassOvr])
        res.det_brk_bass_5=np.array([hBassBrk5])
        res.det_ccd_bass_5=np.array([hBassOvr5])
        res.det_bins=np.array([magBins])
        res.magzpt_bass=np.array([bassZpt])
        res.rmszpt_bass=np.array([bassZptRms])
        res.magoff_bass=np.array([offBass])
        res.magdepth_bass=np.array([bassDepth])
        res.psfdepth_bass=np.array([bassCCD.psfdepth])
        res.imgname_bass=np.array([bassCCD.image_filename])
        res.ccdhdu_bass=np.array([bassCCD.image_hdu])
        res.pass_bass=np.array([overlap.bassPass])

        res.det_brk_dcls=np.array([hDclsBrk])
        res.det_ccd_dcls=np.array([hDclsOvr])
        res.det_brk_dcls_5=np.array([hDclsBrk5])
        res.det_ccd_dcls_5=np.array([hDclsOvr5])
        res.magzpt_dcls=np.array([dclsZpt])
        res.rmszpt_dcls=np.array([dclsZptRms])
        res.magoff_dcls=np.array([offDcls])
        res.magdepth_dcls=np.array([dclsDepth])
        res.psfdepth_dcls=np.array([dclsCCD.psfdepth])
        res.imgname_dcls=np.array([dclsCCD.image_filename])
        res.ccdhdu_dcls=np.array([dclsCCD.image_hdu])
        res.pass_dcls=np.array([overlap.dclsPass])

        res.writeto(fout)
