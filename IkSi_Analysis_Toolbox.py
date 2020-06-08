'''
'''
import glob
import warnings
import numpy as np
import pandas as pd
import xarray as xr
import dask.array as da
from dask.diagnostics import ProgressBar
from scipy import interpolate
import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io.shapereader import Reader
from cartopy.feature import ShapelyFeature
from osgeo import gdal, osr
from matplotlib import pyplot as plt
from matplotlib import cm
import matplotlib.patches as patches
import matplotlib.ticker as ticker
import matplotlib.dates as mdates
import colorcet as cc
import windrose
import gsw

def plot_overview_timeseries(start, end, aqd2dir, mbs_mean, tempsXrInterp, rbr, maximet, fws_rsmpl, cnr_rsmpl):
    '''
    Pure plotting function for the massive overview timeseries of the sea ice station data
    '''
    #---Plot------------------
    plt.rcParams.update({'font.size': 16})
    fig,axx = plt.subplots(nrows=10,ncols=1,figsize=(20,28),sharex=True, facecolor='w')
    fig.suptitle('Sea Ice Station Data',fontsize=30,y=0.91)

    labelfontsz=20
    ylims=[-0.5, 12]
    h1 = axx[0].pcolormesh(aqd2dir.time,aqd2dir.bindepth,aqd2dir.speed,vmin=-1.3,vmax=1.3,cmap='bwr')
    axx[0].fill_between(mbs_mean.index,mbs_mean.IceMean,color='k')
    axx[0].fill_between(mbs_mean.index,mbs_mean.SnowMean,color='gray')
    axx[0].set_xlim([start, end])
    axx[0].set_ylim([-0.5,6])
    axx[0].invert_yaxis()
    axx[0].set_ylabel('Depth (m)',fontdict={'fontsize':labelfontsz})
    axx[0].set_title('Current Speed & Direction [Red=Northwards, Blue=Southwards] (Aquadopp)')
    fig.subplots_adjust(right=0.9)
    cbar_ax1 = fig.add_axes([0.91, 0.819, 0.02, 0.06])
    cbar1 = fig.colorbar(h1, cax=cbar_ax1)
    cbar1.set_label('Speed (m/s)',fontdict={'fontsize':labelfontsz})

    h2 = axx[1].pcolormesh(tempsXrInterp.time,tempsXrInterp.depth,tempsXrInterp.values,cmap=cc.cm.rainbow,vmin=-1.7,vmax=0)
    axx[1].fill_between(mbs_mean.index,mbs_mean.IceMean,color='k')
    axx[1].fill_between(mbs_mean.index,mbs_mean.SnowMean,color='gray')
    axx[1].set_ylim(ylims)
    axx[1].set_yticks([0,6,12])
    axx[1].invert_yaxis()
    axx[1].set_ylabel('Depth (m)',fontdict={'fontsize':labelfontsz})
    axx[1].set_title('Water Temperature Profile (SBE39s & RBR Concerto)')
    cbar_ax2 = fig.add_axes([0.91, 0.7415, 0.02, 0.06])
    cbar2 = fig.colorbar(h2, cax=cbar_ax2)
    cbar2.set_label('Temp ($^\circ$C)',fontdict={'fontsize':labelfontsz})
    cbar2.set_ticks([-1.5,-1,-0.5,0])

    axx[4].plot(rbr.index,rbr.Temperature,'orangered')
    axx[4].set_ylabel('Temp ($^\circ$C)',color='orangered',fontdict={'fontsize':labelfontsz})
    axx[4].set_ylim([-1.7,0])
    axx[4].set_title('Water Temperature & Salinity at 3m Depth (RBR Concerto)')
    par0 = axx[4].twinx()
    par0.plot(rbr.index,rbr.Salinity,'xkcd:blue')
    par0.set_ylabel('Salinity (psu)',color='xkcd:blue',fontdict={'fontsize':labelfontsz})

    h3 = axx[2].scatter(maximet.index,maximet.Speed,c=maximet.Direction,s=3,cmap='twilight_shifted')
    axx[2].set_ylabel('Speed (m/s)',fontdict={'fontsize':labelfontsz})
    axx[2].set_title('Wind Speed & Direction (MaxiMet)')
    cbar_ax3 = fig.add_axes([0.91, 0.665, 0.02, 0.06])
    cbar3 = fig.colorbar(h3, cax=cbar_ax3)
    cbar3.set_label('Direction',fontdict={'fontsize':labelfontsz})
    cbar3.set_ticks([0,90,180,270,360])
    cbar3.set_ticklabels(['N','E','S','W','N'])

    axx[6].plot(maximet.index,maximet.Temperature,'xkcd:red')
    axx[6].set_ylabel('Air Temp ($^\circ$C)',color='xkcd:red',fontdict={'fontsize':labelfontsz})
    axx[6].set_title('Air Temperature (MaxiMet)')
    axx[6].set_yticks([-30,-20,-10,0])

    axx[5].plot(maximet.index,maximet.AH,'grey')
    axx[5].set_ylabel("AH (g/kg)",color='grey',fontdict={'fontsize':labelfontsz})
    axx[5].set_title('Humidity (MaxiMet)')

    par1 = axx[5].twinx()
    par1.plot(maximet.index,maximet.RH,'darkorchid',alpha=0.8)
    par1.set_ylabel('RH (%)',color='darkorchid',fontdict={'fontsize':labelfontsz})
    #par1.set_ylim([50, 100])

    axx[3].fill_between(mbs_mean.index,mbs_mean.IceMean,color='k')
    axx[3].fill_between(mbs_mean.index,mbs_mean.SnowMean,color='gray')
    axx[3].legend(['Ice','Snow'],loc=2)
    axx[3].set_ylim([-0.4, 0.6])
    axx[3].invert_yaxis()
    axx[3].set_ylabel('(m)',fontdict={'fontsize':labelfontsz})
    axx[3].set_title('Ice & Snow Depth (Local Observer)')

    axx[7].plot(fws_rsmpl.index,fws_rsmpl.Rs,'brown',alpha=0.5)
    axx[7].plot(fws_rsmpl.index,fws_rsmpl.Rl_corr,'teal')
    axx[7].plot(maximet.index,maximet.SolarRad,'goldenrod')
    axx[7].legend(['SW K&Z','LW K&Z','SW Maximet'])
    axx[7].set_ylabel('($W/m^2$)',fontdict={'fontsize':labelfontsz})
    axx[7].set_title('Down-welling Radiation (MaxiMet & FWS Kipp&Zonen)')

    axx[8].plot(cnr_rsmpl.index,cnr_rsmpl.NetSW,'orange')
    axx[8].plot(cnr_rsmpl.index,cnr_rsmpl.NetLW,'forestgreen')
    axx[8].legend()
    axx[8].set_ylabel('($W/m^2$)',fontdict={'fontsize':labelfontsz});
    axx[8].set_title('Net Radiation (CNR2)')
    axx[8].set_yticks([-100,0,100,200])

    start_alb = pd.datetime(2019,1,8,0,0,0)
    mm_sw = maximet.loc[start_alb:end].SolarRad
    cnr_swn = cnr_rsmpl.loc[start_alb:end].NetSW

    mm_sw = mm_sw.where(mm_sw > 2)
    cnr_swn = cnr_swn.clip(lower=0)

    absorptivity = cnr_swn.resample('D').mean() / mm_sw.resample('D').mean()
    albedo = 1 - absorptivity
    albedo = albedo.rolling(3).mean()

    axx[9].plot(albedo.index,albedo.values,'navy')
    axx[9].set_ylabel('Albedo',color='navy',fontdict={'fontsize':labelfontsz});
    axx[9].set_title('Albedo (MaxiMet & CNR2)')
    #axx[9].set_title('Temperature at Snow/Ice Interface')
    #axx[9].plot(tt.index,tt.Temp)

    axx[0].xaxis.set_major_locator(ticker.MultipleLocator(5))
    myFmt = mdates.DateFormatter('%b %d')
    axx[0].xaxis.set_major_formatter(myFmt);

#########################################################################################################

def bidir_current_correlations(aqdXr, aqd2dir, tempsXrInterp, rbr):
    '''
    Function for calculating cross correlation between bi-directional current vs temp and salinity, and plotting windrose style directional current histogram
    '''
    warnings.filterwarnings("ignore", message="The poly_between function")

    #-----Define Correlation Functions----
    def covariance(x, y, dims=None):
        return xr.dot(x - x.mean(dims), y - y.mean(dims), dims=dims) / x.count(dims)

    def corrrelation(x, y, dims=None):
        return covariance(x, y, dims) / (x.std(dims) * y.std(dims))

    #-----Co-locate vectors in time-------
    tempz = tempsXrInterp.resample({'time':'15min'}).mean(dim=xr.ALL_DIMS)
    curr = aqd2dir.mean(dim='bindepth').speed
    sali = rbr.Salinity.resample('15min').mean()

    #-----De-mean the vectors--------------
    tempzDT = tempz - tempz.mean()
    currDT = curr - curr.mean()
    saliDT_temp = sali - sali.mean()
    saliDT = saliDT_temp.to_xarray()
    saliDT = saliDT.rename({'Date_Time':'time'})

    #----Perform Lag Correlations-----------
    lag_corr_t = {}
    for lag in np.arange(0,400):
        corr_t = corrrelation(currDT.shift(time=lag).fillna(0),tempzDT)
        lag_corr_t[lag] = corr_t.values.tolist()
    lags_t = list(lag_corr_t.keys())
    corrs_t = list(lag_corr_t.values())

    lag_corr_sal = {}
    for lag in np.arange(0,400):
        corr = corrrelation(currDT.shift(time=lag).fillna(0),saliDT)
        lag_corr_sal[lag] = corr.values.tolist()
    lags_sal = list(lag_corr_sal.keys())
    corrs_sal = list(lag_corr_sal.values())

    #----Identify Lag of Maximum Correlation--
    indMax_t = corrs_t.index(min(corrs_t))
    nhours_t = indMax_t*15/60
    maxnegcorr_t = round((min(corrs_t)),2)

    indMax_sal = corrs_sal.index(max(corrs_sal))
    nhours_sal = indMax_sal*15/60
    maxnegcorr_sal = round((max(corrs_sal)),2)

    #initialize figure
    fig = plt.figure(figsize=(24,8),facecolor='w')
    ax1 = plt.subplot2grid(shape=(1,6), loc=(0,3), rowspan=1, colspan=3, fig=fig)

    #add axes for wind rose
    rect=[0.05,0.05,0.5,0.81]  # [left, bottom, width, height]
    wa=windrose.WindroseAxes(fig, rect)
    fig.add_axes(wa)

    #fill wind rose
    h = wa.contourf(aqdXr.mean(dim='bindepth').direction.values, aqdXr.mean(dim='bindepth').speed.values, bins=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],nsector=64,cmap=cm.bone_r)
    wa.set_yticklabels([])
    wa.set_xticklabels(['E','NE','N','NW','W','SW','S','SE'],fontdict={'fontsize':30})
    wa.legend(loc=5,labelspacing=0.001,title='Speed (m/s)',labels=[],fontsize=17,borderaxespad=1)
    wa.set_title('River Channel Current Velocity Distribution',fontdict={'fontsize':24})

    #plot cross correlations
    plt.sca(ax1)
    plt.xcorr(saliDT,currDT,maxlags=400,usevlines=False,zorder=10);
    plt.xcorr(tempzDT,currDT,maxlags=400,usevlines=False,zorder=11);
    plt.yticks([-0.6,-0.4,-0.2,0,0.2,0.4,0.6],fontsize=30)
    plt.ylabel('Cross-Correlation',fontsize=27)
    plt.xticks([-400,-300,-200,-100,0,100,200,300,400],labels=[-100,-75,-50,-25,0,25,50,75,100],fontsize=30)
    plt.xlabel('Lag (hours)',fontsize=30)
    plt.title('Cross-Correlation of bi-directional current with T & S',fontsize=24,pad=15)
    plt.xlim([-200,400])
    plt.ylim([-0.7,0.7])
    plt.legend(['Currents x Salinity','Currents x Temperature'],fontsize=18,loc=5)
    plt.plot([-200,400],[0,0],'gray',zorder=0,alpha=0.5)
    plt.plot([100,100],[-0.7,0.7],'k--')

#########################################################################################################

def calculate_deltaT(rbr, tempsXrInterp):
    '''
    This function takes the rbr concerto and SBE39 data, corrects the under-ice salinity based on the temperature difference between the salinity measurement and the under-ice measurement, assuming the river is an end-member at 0C and 0psu, and then calculates a delta-T (degrees above freezing under the ice)
    '''
    lat, lon = 66.8968, -162.6139

    #-------deltaT------------------
    rbr['freezing_point'] = gsw.t_freezing(gsw.SA_from_SP(rbr.Salinity,rbr.SeaPressure,lon,lat),rbr.SeaPressure,0)
    rbr['deg_above_freezing'] = rbr.Temperature - rbr.freezing_point

    topDepth = 0.5
    lowerDepth = 3.24
    dTdz = (tempsXrInterp.sel(depth=topDepth) - tempsXrInterp.sel(depth=lowerDepth))/(-topDepth+lowerDepth) # + if warmer above, - if warmer below

    adjusted_salinities = np.zeros(len(rbr))
    subice_temp = tempsXrInterp.sel(depth=topDepth).values

    #interpolate a line between observed T-S and assumed 0,0 river end-member, then calculate salinity for temp at ice bottom from that line
    for ind in np.arange(0,len(rbr)):
        f1 = interpolate.interp1d([0,rbr.Temperature[ind]],[0,rbr.Salinity[ind]],fill_value='extrapolate')
        adjusted_salinities[ind] = f1(subice_temp[ind])

    #only keep adjusted salinities where the temperature gradient is positive ie warm water overlaying cold water
    adjusted_salinities = np.where(dTdz>0,adjusted_salinities,rbr.Salinity)

    #calculate delta-T
    adjusted_freezing_point = gsw.t_freezing(gsw.SA_from_SP(adjusted_salinities,0,lon,lat),0,0)
    adjusted_deg_above_freezing = subice_temp - adjusted_freezing_point

    rbr['Salinity_Adjusted'] = adjusted_salinities
    rbr['deltaT_Adjusted'] = adjusted_deg_above_freezing
    rbr['subice_temp'] = subice_temp

    return rbr

#########################################################################################################

def calculate_ustar(aqdXr, roll):
    '''
    '''
    k = 0.4 #dimensionless von karman constant

    deeper_bin = 7
    shallower_bin= 8
    icedepth=0.5
    deepdepth = aqdXr.bindepth[deeper_bin].values.item()
    shallowdepth = aqdXr.bindepth[shallower_bin].values.item()

    ustar = k*(aqdXr.isel(bindepth=deeper_bin).speed.rolling(time=roll,center=True,min_periods=1).mean() - aqdXr.isel(bindepth=shallower_bin).speed.rolling(time=roll,center=True,min_periods=1).mean())/np.log((deepdepth-icedepth)/(shallowdepth-icedepth))
    ustar = ustar.where(ustar>0)

    return ustar

#########################################################################################################

def calculate_Fw(deltaT, ustar, rbr_rsmpl, lon, lat):
    '''
    Calculate ocean-ice heat flux following McPhee '92
    '''
    St = 0.0057 #dimensionless Stanton Number
    SA = gsw.SA_from_SP(rbr_rsmpl.Salinity_Adjusted,0,lon,lat)
    cp = gsw.cp_t_exact(SA,rbr_rsmpl.subice_temp,0)
    rho = gsw.rho_t_exact(SA,rbr_rsmpl.subice_temp,0) #in-situ density, NOT density anomaly

    Fw_92 = rho*cp*St*ustar*deltaT

    return Fw_92

#########################################################################################################

def plot_thermodynamic_forcing(deltaT, ustar, start, end):
    '''
    '''
    linewd = 3

    fig,axx = plt.subplots(figsize=(20,6),facecolor='w')
    plt.rcParams['font.size'] = 24

    #-------Delta-T & u*0------
    deltaT_daily = deltaT.rolling(4*24,center=True,min_periods=24*2).mean()
    axx.fill_between(deltaT_daily.index,deltaT_daily,color='maroon',alpha=1)
    axx.set_ylabel('$\Delta T$  $(^\circ C)$',color='maroon',fontsize=30)
    axx.set_title('Thermodynamic Forcing',fontsize=30)
    axx.set_ylim([0,0.25])

    ustar_daily = ustar.rolling(time=4*24,center=True,min_periods=2).mean()
    #ustar_err = ustar.rolling(time=4*24,center=True,min_periods=2).std()
    par0 = axx.twinx()
    #par0.fill_between(ustar_daily.time.values,y1=ustar_daily-ustar_err,y2=ustar_daily+ustar_err,color='lightgrey',alpha=0.6)
    par0.plot(ustar_daily.time.values,ustar_daily,'goldenrod',linewidth=linewd*2,alpha=0.5)
    par0.set_ylabel('$u_{*0}$  $(m/s)$',color='goldenrod',fontsize=30)
    par0.set_ylim([0,0.1])

    axx.set_xlim([start, end])
    axx.xaxis.set_major_locator(ticker.MultipleLocator(8))
    myFmt = mdates.DateFormatter('%b %d')
    axx.xaxis.set_major_formatter(myFmt);

#########################################################################################################

def plot_St_vs_Re(StvRe):
    '''
    '''
    plt.rcParams['font.size'] = 16
    fig,axx = plt.subplots(figsize=(10,6),facecolor='w')
    mkr_dict = {'Umea': 'x', 'MZX 84': '+', 'CRX 88': 'o','CRX 89':'<','ANZ 94':'2','SHEBA':'*','This Work':'H'}
    for experiment in mkr_dict:
        d = StvRe[StvRe.index==experiment]
        plt.scatter(d.Re, d.St,s = 100, marker = mkr_dict[experiment])

    plt.ylim([0,0.008])
    plt.grid()
    plt.xlabel(r'$Re_* = u_{*0}z_0/\nu$')
    plt.ylabel(r'$St_*$')
    plt.title('Bulk Heat Transfer Coefficient vs. Surface Friction Reynolds Number',fontdict={'fontsize':'16'})
    plt.legend(StvRe.index)

#########################################################################################################

def plot_flux_balances(mbs_mean, maximet, cnr_rsmpl, rbr_rsmpl, Fw_92):
    '''
    Calculate flux balances at both interfaces and plot em up.
    '''

    #temporally align vectors
    start, end = pd.datetime(2019,1,7,22,45,0), pd.datetime(2019,4,1,20,0,0)
    start2,end2 = pd.datetime(2019,1,8,0,0,0),pd.datetime(2019,3,31)
    mm_crop = maximet.loc[start2:end2]
    cnr_crop = cnr_rsmpl.loc[start2:end2].drop(columns=['RECORD','NetSW_Avg','NetLW_Avg','NetRad_Avg','NetRad','PTemp_C','BattV'])
    mbs_crop = mbs_mean.dropna().resample('10min').mean().interpolate().loc[start2:end2]
    Fw_crop = Fw_92.to_pandas().resample('10min').mean().interpolate().loc[start2:end2]

    #----------------UPPER BOUNDARY--------------------
    #constants
    ki = 2.1 #W/m/K
    ks = 0.3 #W/m/K
    rhoa = 1.2 #kg/m^3
    cp = 1000  #J/kg/K
    Chz = 2*10**(-3) #dimensionless

    #inputs
    Ta = mm_crop.Temperature + 273.15
    Uz = mm_crop.Speed
    F_lw = -cnr_crop.NetLW #given with + into surface, so have to reverse
    F_sw = -cnr_crop.NetSW #given with + into surface, so have to reverse
    Hs = -mbs_crop.SnowMean
    Hi = mbs_crop.IceMean
    Tf = np.zeros(len(mm_crop)) + 273

    To = ( -F_lw - F_sw + (Ta * rhoa * cp * Chz * Uz) + (Tf / ((Hi/ki) + (Hs/ks))) )/((rhoa * cp * Chz * Uz) + (1 / ((Hi/ki) + (Hs/ks))))
    Fc = (To - Tf) / ((Hi/ki) + (Hs/ks)) #conductive
    Fs = (rhoa * cp * Chz * Uz)*(To-Ta)  #sensible


    roll_hrs = 24*2
    linewd = 3

    #------------------LOWER BOUNDARY----------------------
    #minimize the residual between Fw and Fc, use that to scale Fw
    Fc_plot = Fc.rolling(6*roll_hrs,center=True,min_periods=None).mean().where(Fc<0)
    Fw_plot = Fw_crop.rolling(6*roll_hrs,center=True,min_periods=2).mean()
    scale = np.mean(-Fc_plot/Fw_plot)
    resid = Fc_plot + scale*Fw_plot


    #--------------------PLOT--------------------------------
    lgndfont = 30
    plt.rcParams['font.size'] = 24

    fig,axx = plt.subplots(nrows=2, figsize=(22,14), sharex=True, facecolor='w')
    axx[0].plot(cnr_crop.index,F_lw.rolling(roll_hrs,center=True).mean(),'g',linewidth=linewd)
    axx[0].plot(cnr_crop.index,F_sw.rolling(roll_hrs,center=True).mean()-2,color='orange',linewidth=linewd)
    axx[0].plot(cnr_crop.index,Fs.rolling(6*roll_hrs,center=True).mean(),color='firebrick',linewidth=linewd)
    axx[0].plot((F_lw+F_sw+Fs).rolling(6*roll_hrs,center=True,min_periods=None).mean(),'k',linestyle='-',linewidth=linewd*2)
    #axx[0].plot(Fc.rolling(6*roll_hrs,center=True,min_periods=None).mean().where(Fc<0),'purple',linewidth=linewd*2)
    axx[0].plot(rbr_rsmpl.index,np.zeros(len(rbr_rsmpl.index)),'k',linewidth=linewd*2/3,zorder=0)

    axx[0].set_ylim([-100,100])
    axx[0].set_xlim([start,end])
    axx[0].set_ylabel('$W/m^2$')


    myFmt = mdates.DateFormatter('%b %d')
    axx[0].xaxis.set_major_locator(ticker.MultipleLocator(8))
    axx[0].xaxis.set_major_formatter(myFmt);
    axx[0].set_title('Fluxes at the Snow-Air Interface')

    l1 = axx[0].legend(['$F_{LW_{net}}$','$F_{SW_{net}}$','$F_{sensible}$','$F_{total}$'],bbox_to_anchor=(1.001,0.83),fontsize=lgndfont)


    axx[1].plot(scale*Fw_plot, linewidth=linewd*2)
    axx[1].plot(Fc_plot,'purple',linewidth=linewd*2)
    axx[1].plot(resid.rolling(1440,center=True,min_periods=2).mean(),linestyle='--',color='gray',zorder=1)
    axx[1].plot(rbr_rsmpl.index,np.zeros(len(rbr_rsmpl.index)),'k',linewidth=linewd*2/3,zorder=0)
    axx[1].set_ylim([-100,100])
    l2 = axx[1].legend(['$F_{water}$','$F_{conductive}$','Residual'],bbox_to_anchor=(1.001,0.75),fontsize=lgndfont)
    axx[1].set_title('Fluxes at the Ice-Water Interface')
    axx[1].set_ylabel('$W/m^2$')

    axx[1].xaxis.set_major_locator(ticker.MultipleLocator(8))
    axx[1].xaxis.set_major_formatter(myFmt);

    plt.savefig('Figures/Flux Balances v1.png',dpi=300,bbox_extra_artists=(l1,l2), bbox_inches='tight')

#########################################################################################################

def plot_sal_v_deltaT(rbr, start_sub1, end_sub1, start_sub2, end_sub2):
    '''
    This function plots a timeseries and modified T-S diagram of Salinity vs. Departure from Freezing Point, for 2 different subsections of the timeseries as defined by the input parameters.
    '''

    fig = plt.figure(figsize=(12,7), facecolor='w')
    plt.rcParams['font.size'] = 16

    salticks = [0,10,20,30]

    #------Section 1 Timeseries-----
    ax1 = plt.subplot2grid((64,6),(0,0),rowspan=27,colspan=4)
    ax1.plot(rbr.Salinity_Adjusted.loc[start_sub1:end_sub1],'k')
    ax1.set_ylabel('Salinity (psu)',color='k')
    ax1.tick_params(axis='y', colors='k')
    ax1.set_ylim([0,30])
    ax1.set_yticks(salticks)
    ax1.set_xlim(start_sub1,end_sub1)
    ax1.set_title('')
    ax1.patch.set_alpha(0.1)
    ax1.set_zorder(2)
    #ax1.set_xlabel('Time')

    par1 = ax1.twinx()
    par1.fill_between(rbr.deltaT_Adjusted.loc[start_sub1:end_sub1].index,rbr.deltaT_Adjusted.loc[start_sub1:end_sub1],color='maroon',alpha=0.4)
    par1.set_yticks([0,0.1,0.2,0.3,0.4,0.5])
    par1.set_yticklabels([])
    par1.tick_params(axis='y', colors='darkred')
    par1.set_ylim([0,0.5])
    par1.set_ylabel('$\Delta T$ ($^{\circ} C$)',color='darkred')
    par1.set_zorder(1)

    myFmt = mdates.DateFormatter('%b %d')
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(3))
    ax1.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax1.xaxis.set_major_formatter(myFmt);


    #------Section 1 "TS" Diagram-----
    ax11 = plt.subplot2grid((64,24),(0,17),rowspan=27,colspan=6)
    h1 = ax11.scatter(rbr.Salinity_Adjusted.loc[start_sub1:end_sub1],rbr.deltaT_Adjusted.loc[start_sub1:end_sub1],s=1,c=rbr.loc[start_sub1:end_sub1].index,cmap=cc.cm.rainbow)
    ax11.set_ylim([0,0.5])
    ax11.set_xlim([0,21])
    ax11.set_yticks([0,0.1,0.2,0.3,0.4,0.5])
    ax11.set_yticklabels(['0.0','0.1','','','0.4','0.5']);
    ax11.tick_params(axis='y', colors='darkred')
    ax11.tick_params(axis='x', colors='k')
    ax11.set_xticks(salticks)
    ax11.grid(alpha=0.2)


    cax11 = plt.subplot2grid((50,6),(21,0),rowspan=3,colspan=4)
    plt.colorbar(h1, orientation='horizontal',cax=cax11,ticks=[])

    #---------------------------------------------------------------------------------
    #------Section 2 Timeseries-----
    ax2 = plt.subplot2grid((64,6),(32,0),rowspan=27,colspan=4)
    ax2.plot(rbr.Salinity_Adjusted.loc[start_sub2:end_sub2],'k')
    ax2.set_ylabel('Salinity (psu)',color='k')
    ax2.tick_params(axis='y', colors='k')
    ax2.set_ylim([0,30])
    ax2.set_yticks(salticks)
    ax2.set_xlim(start_sub2,end_sub2)
    ax2.set_title('')
    ax2.patch.set_alpha(0.1)
    ax2.set_zorder(2)
    #ax2.set_xlabel('Time')

    par2 = ax2.twinx()
    par2.fill_between(rbr.deltaT_Adjusted.loc[start_sub2:end_sub2].index,rbr.deltaT_Adjusted.loc[start_sub2:end_sub2],color='maroon',alpha=0.4)
    par2.set_yticks([0,0.1,0.2,0.3,0.4,0.5])
    par2.set_yticklabels([])
    par2.tick_params(axis='y', colors='darkred')
    par2.set_ylim([0,0.5])
    par2.set_ylabel('$\Delta T$ ($^{\circ} C$)',color='darkred')
    par2.set_zorder(1)

    ax2.xaxis.set_major_locator(ticker.MultipleLocator(3))
    ax2.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax2.xaxis.set_major_formatter(myFmt);


    #------Section 2 "TS" Diagram-----
    ax21 = plt.subplot2grid((64,24),(32,17),rowspan=27,colspan=6)
    h2 = ax21.scatter(rbr.Salinity_Adjusted.loc[start_sub2:end_sub2],rbr.deltaT_Adjusted.loc[start_sub2:end_sub2],s=1,c=rbr.loc[start_sub2:end_sub2].index,cmap=cc.cm.rainbow)
    ax21.set_ylim([0,0.5])
    ax21.set_xlim([0,30])
    ax21.set_yticks([0,0.1,0.2,0.3,0.4,0.5])
    ax21.set_yticklabels(['0.0','0.1','','','0.4','0.5']);
    ax21.tick_params(axis='y', colors='darkred')
    ax21.tick_params(axis='x', colors='k')
    ax21.set_xticks(salticks)
    ax21.set_xlabel('Salinity (psu)',color='k')
    ax21.grid(alpha=0.2)

    cax21 = plt.subplot2grid((50,6),(46,0),rowspan=3,colspan=4)
    plt.colorbar(h2, orientation='horizontal',cax=cax21,ticks=[])

    fig.suptitle('Relationship Between Salinity & Departure From Freezing Point',y=0.94)

#########################################################################################################

def plot_sal_v_deltaT_single(rbr, start_sub1, end_sub1):
    '''
    This function plots a timeseries and modified T-S diagram of Salinity vs. Departure from Freezing Point, for 2 different subsections of the timeseries as defined by the input parameters.
    '''

    fig = plt.figure(figsize=(12,7), facecolor='w')
    plt.rcParams['font.size'] = 16

    salticks = [0,10,20,30]

    #------Section 1 Timeseries-----
    ax1 = plt.subplot2grid((64,6),(0,0),rowspan=27,colspan=4)
    ax1.plot(rbr.Salinity_Adjusted.loc[start_sub1:end_sub1],'k')
    ax1.set_ylabel('Salinity (psu)',color='k')
    ax1.tick_params(axis='y', colors='k')
    ax1.set_ylim([0,30])
    ax1.set_yticks(salticks)
    ax1.set_xlim(start_sub1,end_sub1)
    ax1.set_title('')
    ax1.patch.set_alpha(0.1)
    ax1.set_zorder(2)
    #ax1.set_xlabel('Time')

    par1 = ax1.twinx()
    par1.fill_between(rbr.deltaT_Adjusted.loc[start_sub1:end_sub1].index,rbr.deltaT_Adjusted.loc[start_sub1:end_sub1],color='maroon',alpha=0.4)
    par1.set_yticks([0,0.1,0.2,0.3,0.4,0.5])
    par1.set_yticklabels([])
    par1.tick_params(axis='y', colors='darkred')
    par1.set_ylim([0,0.5])
    par1.set_ylabel('$\Delta T$ ($^{\circ} C$)',color='darkred')
    par1.set_zorder(1)

    myFmt = mdates.DateFormatter('%b %d')
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(3))
    ax1.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax1.xaxis.set_major_formatter(myFmt);


    #------Section 1 "TS" Diagram-----
    ax11 = plt.subplot2grid((64,24),(0,17),rowspan=27,colspan=6)
    h1 = ax11.scatter(rbr.Salinity_Adjusted.loc[start_sub1:end_sub1],rbr.deltaT_Adjusted.loc[start_sub1:end_sub1],s=1,c=rbr.loc[start_sub1:end_sub1].index,cmap=cc.cm.rainbow)
    ax11.set_ylim([0,0.5])
    ax11.set_xlim([0,21])
    ax11.set_yticks([0,0.1,0.2,0.3,0.4,0.5])
    ax11.set_yticklabels(['0.0','0.1','','','0.4','0.5']);
    ax11.tick_params(axis='y', colors='darkred')
    ax11.tick_params(axis='x', colors='k')
    ax11.set_xticks(salticks)
    ax11.grid(alpha=0.2)


    cax11 = plt.subplot2grid((50,6),(21,0),rowspan=3,colspan=4)
    plt.colorbar(h1, orientation='horizontal',cax=cax11,ticks=[])

    #---------------------------------------------------------------------------------

    fig.suptitle('Relationship Between Salinity & Departure From Freezing Point',y=0.94)

#########################################################################################################

def plot_ts_diagram(temp,salt,temp_lims,sal_lims,axx,pres=0,lon=0,lat=0):
    '''
    Pass pandas series with temperature & salinity, arrays of min & max sal+temp for plot, and the axis you want to plot into
    '''

    # Figure out boudaries (mins and maxs)
    smin = sal_lims[0]
    smax = sal_lims[1]
    tmin = temp_lims[0]
    tmax = temp_lims[1]

    # Calculate how many gridcells we need in the x and y dimensions
    xdim = np.int(round((smax-smin)/0.1+1,0))
    ydim = np.int(round((tmax-tmin)/0.01+1,0))

    # Create empty grid of zeros
    dens = np.zeros((ydim,xdim))

    # Create temp and salt vectors of appropiate dimensions
    ti = np.linspace(1,ydim-1,ydim)*0.01+tmin
    si = np.linspace(1,xdim-1,xdim)*0.1+smin

    # Loop to fill in grid with densities
    for j in range(0,int(ydim)):
        for i in range(0, int(xdim)):
            dens[j,i]=gsw.rho(si[i],ti[j],0)

    # Substract 1000 to convert to sigma-t
    #dens = dens - 1000

    #----------- Plot contours----------------
    CS = axx.contour(si,ti,dens, linestyles='dashed',linewidths=0.5, colors='gray')#cmap='viridis_r')
    plt.clabel(CS, fontsize=8, inline_spacing=1, fmt='%1.0f') # Label every second level
    #cbar1 = plt.colorbar(CS)
    #cbar1.set_label('Density')

    #----------Plot data---------------------
    #axx.plot(salt,temp,'or',markersize=1)
    h1 = axx.scatter(salt,temp,c=salt.index,s=1,cmap=cc.cm.rainbow)
    axx.set_ylim(temp_lims)
    axx.set_xlim(sal_lims)
    #axx.set_xlabel('Salinity')
    #axx.set_ylabel('Temperature (C)');

    #----------Colorbar-----------------------
    #N_TICKS = 8
    #indexes = [salt.index[i] for i in np.linspace(0,salt.shape[0]-1,N_TICKS).astype(int)]
    #cbar2 = plt.gcf().colorbar(h2,ax=axx,ticks= salt.loc[indexes].index.astype(int))
    #cbar2.ax.set_yticklabels([index.strftime('%b %d') for index in indexes])
    #cbar2.set_label('Date')

    #----------Freezing Point Line-------------
    t_freeze = gsw.t_freezing(gsw.SA_from_SP(si,pres,lon,lat),pres,0)
    h2 = axx.plot(si,t_freeze,'k',linewidth=0.5)
    #axx.legend(h2,['Freezing Point'],loc='lower left');

    return h1

#########################################################################################################

def plot_OBT_sections(sbeOBT18, sbeOBT, start1, end1, start2, end2, sal_lims, temp_lims, temp_ticks, temp_tick_labels, nticks, title1, title2, supTitle):
    '''
    '''
    #---subset to date range-----
    sbe1 = sbeOBT18.loc[start1:end1]
    sbe2 = sbeOBT.loc[start2:end2]

    #---intiialize x-ticks------
    xticks17 = pd.date_range(start1, end1, periods=nticks+2)[1:-1]
    xticks18 = pd.date_range(start2, end2, periods=nticks+2)[1:-1]
    myFmt = mdates.DateFormatter('%b %d')

    #------initialize figure--------
    fig = plt.figure(figsize=(14,8),facecolor='w')
    plt.rcParams['font.size'] = 16

    #------Fall 2017 Timeseries-----
    ax1 = plt.subplot2grid((64,6),(0,0),rowspan=27,colspan=4)
    ax1.plot(sbe1.sal,'mediumblue')
    ax1.set_xlim([start1,end1])
    ax1.set_ylim(sal_lims)
    ax1.set_ylabel('Salinity (psu)',color='mediumblue')
    ax1.set_title(title1)
    par1 = ax1.twinx()
    par1.plot(sbe1.temp,'darkred')
    par1.set_ylim(temp_lims)
    par1.set_yticklabels([])
    par1.set_yticks(temp_ticks)
    par1.set_ylabel('($^{\circ} C$)',color='darkred')

    ax1.set_xticks(xticks17)
    ax1.set_xticklabels([''])

    #------Fall 2017 TS Diagram-----
    ax11 = plt.subplot2grid((64,24),(0,17),rowspan=27,colspan=6)
    h2 = plot_ts_diagram(sbe1.temp,sbe1.sal,temp_lims,sal_lims,ax11)
    ax11.set_yticks(temp_ticks)
    ax11.set_yticklabels(temp_tick_labels,color='darkred')

    #-----Fall 2018 Timeseries------
    ax2 = plt.subplot2grid((64,6),(32,0),rowspan=27,colspan=4,facecolor='none')
    ax2.set_title(title2)
    ax2.set_yticks([])
    ax2.set_xlim([start2,end2])
    ax2.set_zorder(1)



    par21 = ax2.twinx()
    par21.plot(sbe2.sal,'mediumblue')
    par21.set_ylim(sal_lims)
    par21.yaxis.tick_left()
    par21.yaxis.set_label_position('left')
    par21.set_ylabel('Salinity (psu)',color='mediumblue')


    par2 = ax2.twinx()
    par2.plot(sbe2.temp,'darkred')
    #par2.set_xlim([start2,end2])
    par2.set_ylim(temp_lims)
    par2.set_yticks(temp_ticks)
    par2.set_yticklabels([])
    par2.set_ylabel('($^{\circ} C$)',color='darkred')

    par2.set_zorder(1)

    par2.xaxis.set_major_formatter(myFmt);
    par2.set_xticks(xticks18)

    #-----Fall 2018 TS Diagram-------
    ax21 = plt.subplot2grid((64,24),(32,17),rowspan=27,colspan=6)
    chandle = plot_ts_diagram(sbe2.temp,sbe2.sal,temp_lims,sal_lims,ax21)
    ax21.set_xlabel('Salinity (psu)',color='mediumblue')
    ax21.set_yticks(temp_ticks)
    ax21.set_yticklabels(temp_tick_labels,color='darkred')

    cax21 = plt.subplot2grid((50,6),(46,0),rowspan=3,colspan=4)
    cbar2 = fig.colorbar(chandle, orientation='horizontal',cax=cax21,ticks=[])
    cax21.set_zorder(0)

    fig.suptitle(supTitle,y=0.96,fontsize=22);

#########################################################################################################

def plot_OBT_fallVspring_TS(sbeOBT, fall_start_18, fall_end_18, spring_start_19, spring_end_19, lat=66.8968, lon=-162.6139):
    '''
    '''

    plt.rcParams['font.size'] = 16

    sbe_fall18 = sbeOBT.loc[fall_start_18:fall_end_18]
    sbe_spring19 = sbeOBT.loc[spring_start_19:spring_end_19]

    #-----Set Up Contours-------------#
    salt_fall = sbe_fall18.sal.values
    temp_fall = sbe_fall18.temp.values

    salt_spring = sbe_spring19.sal.values
    temp_spring = sbe_spring19.temp.values

    # Figure out boudaries (mins and maxs)
    smin = 27.4
    smax = 32.5
    tmin = -2
    tmax = 8

    # Calculate how many gridcells we need in the x and y dimensions
    xdim = np.int(round((smax-smin)/0.1+1,0))
    ydim = np.int(round((tmax-tmin)/0.01+1,0))

    # Create empty grid of zeros
    dens = np.zeros((ydim,xdim))

    # Create temp and salt vectors of appropiate dimensions
    ti = np.linspace(1,ydim-1,ydim)*0.01+tmin
    si = np.linspace(1,xdim-1,xdim)*0.1+smin

    # Loop to fill in grid with densities
    for j in range(0,int(ydim)):
        for i in range(0, int(xdim)):
            dens[j,i]=gsw.rho(si[i],ti[j],0)

    # Substract 1000 to convert to sigma-t
    #dens = dens - 1000

    #----------- Plot contours----------------
    fig1,ax1 = plt.subplots(figsize=(6,6),facecolor='w')
    CS = ax1.contour(si,ti,dens, linestyles='dashed',linewidths=0.5, colors='gray')#cmap='viridis_r')
    plt.clabel(CS, fontsize=8, inline_spacing=1, fmt='%1.0f') # Label every second level
    #cbar1 = plt.colorbar(CS)
    #cbar1.set_label('Density')

    #----------Plot data---------------------
    h0 = ax1.plot(salt_fall,temp_fall,'C1')
    h1 = ax1.plot(salt_spring,temp_spring,'C2')
    #h2 = ax1.scatter(salt,temp,c=sbe.index,s=1,cmap=cc.cm.rainbow)
    ax1.set_xlabel('Salinity (psu)')
    ax1.set_ylabel('Temperature ($^\circ C$)');

    #----------Colorbar-----------------------

    #----------Freezing Point Line-------------
    pres=np.mean(sbeOBT.pres)
    t_freeze = gsw.t_freezing(gsw.SA_from_SP(si,pres,lon,lat),pres,0)
    h2 = ax1.plot(si,t_freeze,'k',linewidth=1)
    ax1.legend(['Fall (Oct-Dec)','Spring (Apr-Jun)','Freezing Point'],loc='upper left',framealpha=1);
    ax1.set_title('Fall vs. Spring in T-S Space')
