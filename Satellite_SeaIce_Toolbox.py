'''
Toolbox of Functions for processing satellite sea ice products, developed for processing IkSi-related data.
'''
import warnings
import numpy as np
import pandas as pd
import xarray as xr
import dask.array as da
from dask.diagnostics import ProgressBar
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy
from matplotlib import pyplot as plt
import matplotlib.patches as patches

def subset_ssmi(ssmi, lat_min, lat_max, lon_min, lon_max, xmin, xmax, ymin, ymax):
    '''
    This function takes an xarray dataset of SSM/I Sea Ice Concentration imported from the OpenDAP server at http://icdc.cen.uni-hamburg.de, and subsets the dataset based on the geographical parameters provided.

    Inputs:
        ssmi - xarray dataset following the conventions of the Hamburg ICDC
        lat_min, lat_max - latitude bounds inside of which to keep data
        lon_min, lon_max - longitude bounds inside of which to keep data
        xmin, xmax, ymin, ymax - bounds of the coordinate system that you want to slice. These should be determined by inspection of the dataset to slice around the sides of the lat and lon bounds.
    '''
    si = ssmi.where(
            ssmi.latitude>lat_min).where(
            ssmi.latitude<lat_max).where(
            ssmi.longitude>(lon_min+360)%360).where(
            ssmi.longitude<(lon_max+360)%360).where(
#            ssmi.sea_ice_area_fraction <= 100).where(
            ssmi.sea_ice_area_fraction > 0).sel(x=slice(xmin,xmax),y=slice(ymin,ymax))
    return si

def plot_layered_ice_map(si, si19, startDay, endDay, thresh, transparency, plotflag):
    '''
    '''
    #this code will throw a Runtime Warning because there are Nans in the dataset where there is land. I don't want this popping up 20 times as we work through each year, so I'm suppressing that specific warning.
    warnings.filterwarnings("ignore", message="Mean of empty slice")

    #initialize plotting variables
    lon_min, lon_max = 175, -150
    lat_min, lat_max = 58, 75
    xmin, xmax = lon_min+2.5, lon_max-6 #map limits, tuned for nice plot
    ymin, ymax = lat_min-0.1, lat_max-4.25 #map limits, tuned for nice plot
    cmax = 100
    cmin = thresh
    colorstring='#ffffff'
    color2019='#bc112b'
    seacolor='#121b86'
    fontsz=24

    #create map figure
    fig,axx = plt.subplots(figsize=(16,16),
                           subplot_kw=dict(projection=ccrs.AlbersEqualArea(central_longitude=(-167),central_latitude=(lat_min+lat_max)/2,standard_parallels=[40,65])))
    #put in background color of #121b86
    axx.background_patch.set_facecolor(seacolor)

    #group by year and iterate over each year
    si_gb = si.groupby('time.year')
    for label,group in si_gb:
        #group by day of year
        dayOfYear = group.groupby('time.dayofyear').mean(dim='time')
        #slice out one month & take time mean
        monthIce = dayOfYear.sel(dayofyear=slice(startDay,endDay)).mean(dim='dayofyear')
        #plot sea ice onto map
        axx.contourf(monthIce.longitude,monthIce.latitude,monthIce.sea_ice_area_fraction,levels=[thresh,100],transform=ccrs.PlateCarree(),colors=colorstring,alpha=transparency,vmin=cmin,vmax=cmax,linestyles='solid')

        #add in coastlines and save figures each iteration if plotflag
        if plotflag == 1:
            axx.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '50m', edgecolor='face', facecolor='#efebd3'),zorder=100)
            axx.coastlines(resolution='50m',zorder=200)
            axx.set_extent([xmin,xmax,ymin,ymax])
            plt.savefig(f'Figures/Layered Sea Ice Maps/AprSeaIceExtent_to_{label}.png',dpi=300,facecolor='k')

    #plot 2019 on top in red
    month19 = si19.groupby('time.dayofyear').mean(dim='time').sel(dayofyear=slice(startDay,endDay)).mean(dim='dayofyear')
    axx.contourf(month19.longitude,month19.latitude,month19.sea_ice_area_fraction,levels=[thresh,100],transform=ccrs.PlateCarree(),colors=color2019,vmin=cmin,vmax=cmax,linestyles='solid')

    #Map Formatting
    axx.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '50m', edgecolor='face', facecolor='#efebd3'),zorder=100)
    axx.coastlines(resolution='50m',zorder=200)
    axx.set_extent([xmin,xmax,ymin,ymax])

    #Add title to map
    axx.text(0.5, 0.99, 'April Sea Ice Extent \n Since 2000', fontsize=45, fontname='Calibri', horizontalalignment='center', verticalalignment='top', transform=axx.transAxes,
             bbox=dict(boxstyle='square,pad=0.15', facecolor='w', alpha=0.8),zorder=300)

    #Add data source to map
    axx.text(0.01, 0.01, 'Data from AMSR-2 Satellite Processed with ASI Algorithm by U.Bremen', fontsize=25, fontname='Calibri', horizontalalignment='left', verticalalignment='bottom', transform=axx.transAxes,
             bbox=dict(boxstyle='square,pad=0.15', facecolor='w', alpha=0.8),zorder=300)

    #Add custom legend to map
    #Underlying rectangle of blue with transparent boxes stacked on top. Need 20 boxes to account for 19 years + the zero value.
    x = 0.87
    y = 0.55
    w = 0.025
    h = 0.25
    nyears = 20

    #plot legend boxes
    axx.add_patch(patches.Rectangle(xy=(x,y), width=w, height=h, edgecolor='k', facecolor=seacolor, alpha=1, transform=axx.transAxes, zorder=300))
    for idx in np.arange(1,nyears):
        counter = 0
        while counter < idx:
            axx.add_patch(patches.Rectangle(xy=(x,y+(idx*h/nyears)), width=w, height=h/nyears, edgecolor='k', facecolor=colorstring, alpha=0.15, transform=axx.transAxes, zorder=300))
            counter = counter + 1
    #label legend
    axx.text(x, y + h, '% of Prior Years  \n Covered By Ice  ', fontsize=20, fontname='Calibri', horizontalalignment='right', verticalalignment='top', transform=axx.transAxes,zorder=300)
    axx.text(x + w, y + h, ' 100%', fontsize=20, fontname='Calibri', horizontalalignment='left', verticalalignment='top', transform=axx.transAxes,zorder=300)
    axx.text(x + w, y + h/2, ' 50%', fontsize=20, fontname='Calibri', horizontalalignment='left', verticalalignment='center', transform=axx.transAxes,zorder=300)
    axx.text(x + w, y - (h/nyears/2), ' 0%', fontsize=20, fontname='Calibri', horizontalalignment='left', verticalalignment='bottom', transform=axx.transAxes,zorder=300)

    #add 2019 to legend
    axx.add_patch(patches.Rectangle(xy=(x,y+h+(2*h/nyears)), width=w, height=h/nyears, edgecolor='k', facecolor=color2019, alpha=1, transform=axx.transAxes, zorder=300))
    axx.text(x, y + h +(2.4*h/nyears), '2019  ', fontsize=20, fontname='Calibri', horizontalalignment='right', verticalalignment='center', transform=axx.transAxes,zorder=300)

    #label Kotzebue Sound on plot
    axx.text(0.65, 0.54, 'Kotzebue \n Sound', fontsize=18, transform=axx.transAxes, zorder=300)
    axx.add_patch(patches.Arrow(x=0.69, y=0.58, dx=0.01, dy=0.05, width=0.01, facecolor='k',transform=axx.transAxes,zorder=300))
