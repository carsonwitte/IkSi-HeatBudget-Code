'''
Toolbox of Functions for processing satellite sea ice products, developed for processing IkSi-related data.
'''
import warnings
import numpy as np
import pandas as pd
import xarray as xr
import dask.array as da
from dask.diagnostics import ProgressBar
import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io.shapereader import Reader
from cartopy.feature import ShapelyFeature
from osgeo import gdal, osr
import glob
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import ConnectionPatch
import colorcet as cc


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

#########################################################################################################

def plot_layered_ice_map(si, si19, startDay, endDay, thresh, transparency, plotflag):
    '''
    This function takes xarray datasets of historical sea ice concentration and 2019 sea ice concentration, and plots each year of history as a transparent layer. It then overlays the 2019 map in red.

    Inputs:
        si           - xarray dataset with historical sea ice concentration back to 2000
        si19         - xarray dataset of same form as si, with 2019 sea ice data
        startDay     - first day of the year to include in our average for plotting (currently April 1)
        endDay       - last day of the year to include in our average for plotting (currently April 31)
        thresh       - ice concentration threshold above which to plot ice (currently 70%)
        transparency - how transparent to make each layer (currently 0.15, which makes the transparency of 19 layers stacked on top of each other be about 0.96)
        plotflag     - if set to 1, plot a map for each year. if set to 0, just plot the final map with everything overlaid

    Outputs: This function does not return anything, but outputs a figure.
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

    color2019='#bc112b'
    seacolor='#121b86'
    colorstring = 'w'
    fontsz=24

    #Underlying rectangle  for custom legend. Blue with transparent boxes stacked on top. Need 20 boxes to account for 19 years + the zero value.
    x = 0.87
    y = 0.55
    w = 0.025
    h = 0.25
    nyears = 20

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

        #add in formatting and save figures each iteration if plotflag
        if plotflag == 1:
            axx.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '50m', edgecolor='face', facecolor='#efebd3'),zorder=100)
            axx.coastlines(resolution='50m',zorder=200)
            axx.set_extent([xmin,xmax,ymin,ymax])
            #Add title to map
            axx.text(0.5, 0.99, 'April Sea Ice Extent \n Since 2000', fontsize=45, fontname='Calibri', horizontalalignment='center', verticalalignment='top', transform=axx.transAxes,
                     bbox=dict(boxstyle='square,pad=0.15', facecolor='w', alpha=0.8),zorder=300)

            #Add data source to map
            axx.text(0.01, 0.01, 'AMSR-2 Satellite Data Processed with ASI Algorithm, obtained from Hamburg ICDC', fontsize=25, fontname='Calibri', horizontalalignment='left', verticalalignment='bottom', transform=axx.transAxes,
                     bbox=dict(boxstyle='square,pad=0.15', facecolor='w', alpha=0.8),zorder=300)

            #Add custom legend to map
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

            #label Kotzebue Sound
            axx.text(0.65, 0.54, 'Kotzebue \n Sound', fontsize=18, transform=axx.transAxes, zorder=300)
            axx.add_patch(patches.Arrow(x=0.69, y=0.58, dx=0.01, dy=0.05, width=0.01, facecolor='k',transform=axx.transAxes,zorder=300))

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
    axx.text(0.01, 0.01, 'AMSR-2 Satellite Data Processed with ASI Algorithm, obtained from Hamburg ICDC', fontsize=25, fontname='Calibri', horizontalalignment='left', verticalalignment='bottom', transform=axx.transAxes,
             bbox=dict(boxstyle='square,pad=0.15', facecolor='w', alpha=0.8),zorder=300)

    #Add custom legend to map
    #plot legend boxes
    axx.add_patch(patches.Rectangle(xy=(x,y), width=w, height=h, edgecolor='k', facecolor=seacolor, alpha=1, transform=axx.transAxes, zorder=300))
    for idx in np.arange(1,nyears):
        counter = 0
        while counter < idx:
            axx.add_patch(patches.Rectangle(xy=(x,y+(idx*h/nyears)), width=w, height=h/nyears, edgecolor='k', facecolor=colorstring, alpha=0.15, transform=axx.transAxes, zorder=300))
            counter = counter + 1
    #label legend
    axx.text(x, y + h, '% of Prior Years  \n (2000-2018)  \n Covered By Ice  ', fontsize=20, fontname='Calibri', horizontalalignment='right', verticalalignment='top', transform=axx.transAxes,zorder=300)
    axx.text(x + w, y + h, ' 100%', fontsize=20, fontname='Calibri', horizontalalignment='left', verticalalignment='top', transform=axx.transAxes,zorder=300)
    axx.text(x + w, y + h/2, ' 50%', fontsize=20, fontname='Calibri', horizontalalignment='left', verticalalignment='center', transform=axx.transAxes,zorder=300)
    axx.text(x + w, y - (h/nyears/2), ' 0%', fontsize=20, fontname='Calibri', horizontalalignment='left', verticalalignment='bottom', transform=axx.transAxes,zorder=300)

    #add 2019 to legend
    axx.add_patch(patches.Rectangle(xy=(x,y+h+(2*h/nyears)), width=w, height=h/nyears, edgecolor='k', facecolor=color2019, alpha=1, transform=axx.transAxes, zorder=300))
    axx.text(x, y + h +(2.4*h/nyears), '2019  ', fontsize=20, fontname='Calibri', horizontalalignment='right', verticalalignment='center', transform=axx.transAxes,zorder=300)

    #label Places on plot
    axx.text(0.65, 0.54, 'Kotzebue \n Sound', fontsize=18, transform=axx.transAxes, zorder=300)
    axx.add_patch(patches.Arrow(x=0.69, y=0.58, dx=0.01, dy=0.05, width=0.01, facecolor='k',transform=axx.transAxes,zorder=300))

    axx.text(0.05, 0.75, 'Russia', fontsize=33, transform=axx.transAxes, zorder=300)
    axx.text(0.8, 0.26, 'Alaska', fontsize=33, transform=axx.transAxes, zorder=300)
    axx.text(0.35, 0.8, 'Chukchi Sea', fontsize=26, transform=axx.transAxes, zorder=300)
    axx.text(0.35, 0.27, 'Bering Sea', fontsize=26, transform=axx.transAxes, zorder=300)

    axx.text(0.4, 0.6, 'Bering \nStrait', fontsize=18, transform=axx.transAxes, zorder=300)
    axx.add_patch(patches.Arrow(x=0.459, y=0.615, dx=0.06, dy=-0.013, width=0.01, facecolor='k',transform=axx.transAxes,zorder=300))

    axx.text(-0.065,0.01, '$60^\circ$N', fontsize=18, transform=axx.transAxes, zorder=300)
    axx.text(-0.065,0.53, '$65^\circ$N', fontsize=18, transform=axx.transAxes, zorder=300)
    axx.text(-0.065,0.975, '$70^\circ$N', fontsize=18, transform=axx.transAxes, zorder=300)

    axx.text(0.11,-0.025, '$180^\circ$W', fontsize=18, transform=axx.transAxes, zorder=300)
    axx.text(0.465,-0.025, '$170^\circ$W', fontsize=18, transform=axx.transAxes, zorder=300)
    axx.text(0.795,-0.025, '$160^\circ$W', fontsize=18, transform=axx.transAxes, zorder=300)

#########################################################################################################

def plot_monthly_anomalies(si, sst):
    '''
    This function plots monthly anomalies of sea ice cover and sst in a bar graph spaced by month and colored by year.

    Inputs:
        si  - xarray dataset of sea ice concentration
        sst - xarray dataset of sea surface temperature

    Outputs: no outputs, but a figure.
    '''
    #set font size for plot (could make this an input)
    fontsz = 20

    #suppress warnings that we expect
    warnings.filterwarnings("ignore", message="Slicing")
    warnings.filterwarnings("ignore", message="Mean of empty slice")

    #calculate monthly anomalies
    si_anom = si.groupby('time.month') - si.groupby('time.month').mean(dim='time')
    sst_anom = sst.groupby('time.month') - sst.groupby('time.month').mean(dim='time')

    #take monthly means and then split datasets up into groupby objects by year
    sst_gb = sst_anom.resample(time='1M').mean().analysed_sst.mean(dim={'lat','lon'}).groupby('time.year')
    si_gb = si_anom.resample(time='1M').mean().sea_ice_area_fraction.mean(dim={'y','x'}).groupby('time.year')

    #extract start date and number of years for plotting reference
    start_date = pd.to_datetime(si.time.isel(time=0).values)
    num_yrs = len(si_gb)

    #initialize figure and set color cycle of axes based on how many years we're plotting
    fig, axx = plt.subplots(nrows=2,ncols=1,figsize=(16,10),facecolor='w')
    #cm = plt.get_cmap('gist_rainbow')
    cm = cc.cm['rainbow']
    axx[0].set_prop_cycle('color',[cm(1.*i/num_yrs) for i in range(num_yrs)])
    axx[1].set_prop_cycle('color',[cm(1.*i/num_yrs) for i in range(num_yrs)])

    #width of bars
    w = 1/(num_yrs+2)
    handles={}
    labels={}
    #plot bars for each month in each grouped year
    #sea ice
    for key, group in si_gb:
        ind = (key-start_date.year)
        months = group.time.dt.month.values
        vals = group.values
        offset = ((ind+1)-(num_yrs/2))*w
        h = axx[0].bar(months+offset,vals,width=w,align='center')
        handles[key] = h
    #sst
    for key, group in sst_gb:
        ind = (key-start_date.year)
        months = group.time.dt.month.values
        vals = group.values
        offset = ((ind+1)-(num_yrs/2))*w
        h = axx[1].bar(months+offset,vals,width=w,align='center')

    #plot legend outside of axes
    axx[0].legend(handles,bbox_to_anchor=(1.01,1.01),prop={'size': fontsz-2})

    #format x-ticks
    axx[0].set_xticks(np.arange(1,13));
    axx[1].set_xticks(np.arange(1,13));
    axx[0].set_xticklabels(['Jan','Feb','Mar','Apr','May','June','July','Aug','Sep','Oct','Nov','Dec'],fontsize=fontsz);
    axx[1].set_xticklabels(['Jan','Feb','Mar','Apr','May','June','July','Aug','Sep','Oct','Nov','Dec'],fontsize=fontsz);

    #plot vertical lines between each month to visually separate them
    borders = np.arange(((num_yrs/2)+1.5)/(1/w),13)
    for i in borders:
        axx[0].axvline(x=i,c='k',alpha=0.7)
        axx[1].axvline(x=i,c='k',alpha=0.7)
    axx[0].set_xlim(borders[0],borders[-1]);
    axx[1].set_xlim(borders[0],borders[-1]);

    #label plots
    axx[0].set_title(f'(a) Sea Ice Cover 20yr Monthly Anomaly',fontsize=fontsz);
    axx[0].set_ylabel('% Sea Ice Coverage Anomaly',fontsize=fontsz);

    axx[1].set_title(f'(b) Sea Surface Temperature 20yr Monthly Anomaly',fontsize=fontsz);
    axx[1].set_ylabel('SST Anomaly ($^{\circ}$C)',fontsize=fontsz);


#########################################################################################################

def plot_layered_landfast_ice_map(data_folder, coastline_path, river_path, kotz_path, transparency, xmin, xmax, ymin, ymax, plotflag):
    '''
    This function imports shapefiles of landfast ice extent and plots them in a transparent stack with 2019 in red, along with coastline and river shapefiles and some labels and legends.

    Inputs:

    '''
    #define colors (could be made an input)
    seacolor='#121b86'
    colorstring='#ffffff'
    color2019='#bc112b'

    #Underlying rectangle for legend of blue with transparent boxes stacked on top. Need 20 boxes to account for 19 years + the zero value.
    x = 0.89
    y = 0.54
    w = 0.025
    h = 0.35
    nyears = 20

    #read in shapefiles
    coastline = ShapelyFeature(Reader(coastline_path).geometries(),ccrs.PlateCarree())
    river = ShapelyFeature(Reader(river_path).geometries(),ccrs.PlateCarree())
    kotz = ShapelyFeature(Reader(kotz_path).geometries(),ccrs.PlateCarree())

    shape_list = []
    files = glob.glob(data_folder + '\*.shp')
    for fname in files:
        shape_feature = ShapelyFeature(Reader(fname).geometries(),ccrs.PlateCarree(), edgecolor='black', facecolor=colorstring, alpha=transparency)
        shape_list.append(shape_feature)

    lfi2019_feature = ShapelyFeature(Reader(fname).geometries(),ccrs.PlateCarree(), edgecolor='black', facecolor=color2019, alpha=1)

    #initialize map figure
    fig,axx = plt.subplots(figsize=(16,16),subplot_kw=dict(projection=ccrs.AlbersEqualArea(central_longitude=(xmin+xmax)/2,central_latitude=(ymin+ymax)/2,standard_parallels=[40,65])))
    #put in background color of #121b86
    axx.background_patch.set_facecolor(seacolor)
    #set map extent
    axx.set_extent([xmin,xmax,ymin,ymax])
    #stack layers of transparent ice shapes
    yr = 2000
    for shape in shape_list:
        axx.add_feature(shape)
        #if we want to save a plot of every year...
        if plotflag == 1:
            #add coastline shapefile
            axx.add_feature(coastline,zorder=300, edgecolor='k', facecolor='#efebd3', alpha=1)
            #add rivers shapefile
            axx.add_feature(river,zorder=350, edgecolor='#46d3f6', facecolor='#efebd3', alpha=1,linewidth=4)
            #Add title to map
            #axx.text(0.5, 0.99, 'Landfast Ice Extent \n Since 2000', fontsize=45, fontname='Calibri', horizontalalignment='center', verticalalignment='top', transform=axx.transAxes,
            #         bbox=dict(boxstyle='square,pad=0.15', facecolor='w', alpha=0.8),zorder=400)
            #Add data source to map
            #axx.text(0.006, 0.01, 'Ice Edges Traced from MODIS Visible Images', fontsize=18, fontname='Calibri', horizontalalignment='left', verticalalignment='bottom', transform=axx.transAxes,
            #         bbox=dict(boxstyle='square,pad=0.15', facecolor='w', alpha=0.8),zorder=400)
            #Add custom legend to map
            #plot legend boxes
            axx.add_patch(patches.Rectangle(xy=(x,y), width=w, height=h, edgecolor='k', facecolor=seacolor, alpha=1, transform=axx.transAxes, zorder=400))
            for idx in np.arange(1,nyears):
                counter = 0
                while counter < idx:
                    axx.add_patch(patches.Rectangle(xy=(x,y+(idx*h/nyears)), width=w, height=h/nyears, edgecolor='k', facecolor=colorstring, alpha=0.15, transform=axx.transAxes, zorder=400))
                    counter = counter + 1
            #label legend
            axx.text(x, y + h, '% of Prior Years  \n (2000-2018)  \n Covered By  \n Landfast Ice  ', fontsize=20, fontname='Calibri', horizontalalignment='right', verticalalignment='top', transform=axx.transAxes,zorder=400)
            axx.text(x + w, y + h, ' 100%', fontsize=20, fontname='Calibri', horizontalalignment='left', verticalalignment='top', transform=axx.transAxes,zorder=400)
            axx.text(x + w, y + h/2, ' 50%', fontsize=20, fontname='Calibri', horizontalalignment='left', verticalalignment='center', transform=axx.transAxes,zorder=400)
            axx.text(x + w, y - (h/nyears/2), ' 0%', fontsize=20, fontname='Calibri', horizontalalignment='left', verticalalignment='bottom', transform=axx.transAxes,zorder=400)
            #add 2019 to legend
            #axx.add_patch(patches.Rectangle(xy=(x,y+h+(2*h/nyears)), width=w, height=h/nyears, edgecolor='k', facecolor=color2019, alpha=1, transform=axx.transAxes, zorder=400))
            #axx.text(x, y + h +(2.4*h/nyears), '2019  ', fontsize=20, fontname='Calibri', horizontalalignment='right', verticalalignment='center', transform=axx.transAxes,zorder=400)
            #label features on plot
            axx.text(0.7, 0.475, 'Kobuk \n River', fontsize=18, transform=axx.transAxes, zorder=400)
            axx.text(0.46, 0.75, 'Noatak \n River', fontsize=18, transform=axx.transAxes, zorder=400)
            axx.text(0.538, 0.545, 'Kotzebue', fontsize=13, transform=axx.transAxes, zorder=400)
            axx.add_patch(patches.Arrow(x=0.55, y=0.56, dx=-0.016, dy=0.03, width=0.01, facecolor='k',transform=axx.transAxes,zorder=400));

            plt.savefig(f'Figures/Layered Landfast Ice Maps/LandfastIce_to_{yr}.png',dpi=300,facecolor='k')

        yr = yr + 1

    #add 2019 in red
    axx.add_feature(lfi2019_feature)
    #add coastline shapefile
    axx.add_feature(coastline,zorder=300, edgecolor='k', facecolor='#efebd3', alpha=1)
    #add rivers shapefile
    axx.add_feature(river,zorder=350, edgecolor='#46d3f6', facecolor='#efebd3', alpha=1,linewidth=4)
    #Add title to map
    axx.text(0.5, 0.99, 'Landfast Ice Extent \n Since 2000', fontsize=45, fontname='Calibri', horizontalalignment='center', verticalalignment='top', transform=axx.transAxes,
             bbox=dict(boxstyle='square,pad=0.15', facecolor='w', alpha=0.8),zorder=400)
    #Add data source to map
    axx.text(0.006, 0.01, 'Ice Edges Traced from MODIS Visible Images', fontsize=18, fontname='Calibri', horizontalalignment='left', verticalalignment='bottom', transform=axx.transAxes,
             bbox=dict(boxstyle='square,pad=0.15', facecolor='w', alpha=0.8),zorder=400)
    #Add custom legend to map

    #plot legend boxes
    axx.add_patch(patches.Rectangle(xy=(x,y), width=w, height=h, edgecolor='k', facecolor=seacolor, alpha=1, transform=axx.transAxes, zorder=400))
    for idx in np.arange(1,nyears):
        counter = 0
        while counter < idx:
            axx.add_patch(patches.Rectangle(xy=(x,y+(idx*h/nyears)), width=w, height=h/nyears, edgecolor='k', facecolor=colorstring, alpha=0.15, transform=axx.transAxes, zorder=400))
            counter = counter + 1
    #label legend
    axx.text(x, y + h, '% of Prior Years  \n (2000-2018)  \n Covered By  \n Landfast Ice  ', fontsize=20, fontname='Calibri', horizontalalignment='right', verticalalignment='top', transform=axx.transAxes,zorder=400)
    axx.text(x + w, y + h, ' 100%', fontsize=20, fontname='Calibri', horizontalalignment='left', verticalalignment='top', transform=axx.transAxes,zorder=400)
    axx.text(x + w, y + h/2, ' 50%', fontsize=20, fontname='Calibri', horizontalalignment='left', verticalalignment='center', transform=axx.transAxes,zorder=400)
    axx.text(x + w, y - (h/nyears/2), ' 0%', fontsize=20, fontname='Calibri', horizontalalignment='left', verticalalignment='bottom', transform=axx.transAxes,zorder=400)
    #add 2019 to legend
    axx.add_patch(patches.Rectangle(xy=(x,y+h+(2*h/nyears)), width=w, height=h/nyears, edgecolor='k', facecolor=color2019, alpha=1, transform=axx.transAxes, zorder=400))
    axx.text(x, y + h +(2.4*h/nyears), '2019  ', fontsize=20, fontname='Calibri', horizontalalignment='right', verticalalignment='center', transform=axx.transAxes,zorder=400)
    #label features on plot
    axx.text(0.7, 0.475, 'Kobuk \n River', fontsize=18, transform=axx.transAxes, zorder=400)
    axx.text(0.46, 0.75, 'Noatak \n River', fontsize=18, transform=axx.transAxes, zorder=400)
    axx.text(0.538, 0.545, 'Kotzebue', fontsize=13, transform=axx.transAxes, zorder=400)
    axx.add_patch(patches.Arrow(x=0.55, y=0.56, dx=-0.016, dy=0.03, width=0.01, facecolor='k',transform=axx.transAxes,zorder=400));

    axx.text(0.345, 0.127, '2006', fontsize=14, transform=axx.transAxes, zorder=400)
    axx.text(0.354, 0.153, '2018', fontsize=14, transform=axx.transAxes, zorder=400)
    axx.text(0.363, 0.19, '2011', fontsize=14, transform=axx.transAxes, zorder=400)
    axx.text(0.375, 0.23, '2009', fontsize=14, transform=axx.transAxes, zorder=400)
    axx.text(0.4, 0.285, '2004', fontsize=14, transform=axx.transAxes, zorder=400)
    axx.text(0.44, 0.43, '2014', fontsize=14, transform=axx.transAxes, zorder=400)
    axx.text(0.355, 0.47, '2017', fontsize=14, transform=axx.transAxes, zorder=400)

    axx.text(-0.065, 0.01, '$66.0^\circ$N', fontsize=14, transform=axx.transAxes, zorder=400)
    axx.text(-0.065, 0.34, '$66.5^\circ$N', fontsize=14, transform=axx.transAxes, zorder=400)
    axx.text(-0.065, 0.67, '$67.0^\circ$N', fontsize=14, transform=axx.transAxes, zorder=400)
    axx.text(-0.065, 0.97, '$67.5^\circ$N', fontsize=14, transform=axx.transAxes, zorder=400)

    axx.text(0.06,-0.025, '$165^\circ$W', fontsize=14, transform=axx.transAxes, zorder=400)
    axx.text(0.43,-0.025, '$163^\circ$W', fontsize=14, transform=axx.transAxes, zorder=400)
    axx.text(0.8,-0.025, '$161^\circ$W', fontsize=14, transform=axx.transAxes, zorder=400)

    #plot urban Kotzebue
    axx.add_feature(kotz,zorder=300, edgecolor='k', facecolor='gray', alpha=1)

    #plot circles for locations of ITO and OBT
    from matplotlib.patches import Ellipse
    ITO_xy = (-162.6170,66.8969)
    OBT_xy = (-163.7957,67.0598)
    wd = 0.05

    axx.add_patch(Ellipse(xy=ITO_xy,width=wd,height=wd/2.5,transform=ccrs.PlateCarree(),zorder=500,facecolor='w',edgecolor='k'))
    axx.text(0.498, 0.598, 'ITO', fontsize=13, transform=axx.transAxes, zorder=400)

    axx.add_patch(Ellipse(xy=OBT_xy,width=wd,height=wd/2.5,transform=ccrs.PlateCarree(),zorder=500,color='k'))
    axx.text(0.278, 0.7, 'OBT', fontsize=13, transform=axx.transAxes, zorder=400)


#########################################################################################################

def plot_MODIS_geotiff(ax, fname, coastline, lon_min_MOD, lon_max_MOD, lat_min_MOD, lat_max_MOD):
    '''
    This simple function plots a MODIS geotiff located at fname into the axes specified and crops the extent as specified. The axes must have the same crs as the geotiff.
    '''

    ds = gdal.Open(fname)
    data = ds.ReadAsArray()
    gt = ds.GetGeoTransform()


    extent = (gt[0], gt[0] + ds.RasterXSize * gt[1],
              gt[3] + ds.RasterYSize * gt[5], gt[3])

    img = ax.imshow(data[:3, :, :].transpose((1, 2, 0)), extent=extent,
                    origin='upper')

    ax.set_extent([lon_min_MOD, lon_max_MOD, lat_min_MOD, lat_max_MOD])

    ax.add_feature(coastline,zorder=300, edgecolor='k', facecolor='none', alpha=1)

#########################################################################################################

def plot_breakup_images(paths_2007, paths_2012, paths_2019, coastline_path, lon_min_MOD, lon_max_MOD, lat_min_MOD, lat_max_MOD, fontsz, figsz):
    '''
    This function plots n images from 2007, 2012, and 2019 Kotzebue Sound breakups in a 3xn grid


    '''
    gdal.UseExceptions()

    cols = len(paths_2007)
    coastline = ShapelyFeature(Reader(coastline_path).geometries(),ccrs.PlateCarree())

    #extract coordinate system from first file to be used as projection of all subplots
    fname = paths_2007[0]
    ds = gdal.Open(fname)
    proj = ds.GetProjection()
    inproj = osr.SpatialReference()
    inproj.ImportFromWkt(proj)
    projcs = inproj.GetAuthorityCode('PROJCS')
    projection = ccrs.epsg(projcs)
    subplot_kw = dict(projection=projection)
    #initialize figure
    fig, axx = plt.subplots(nrows=3, ncols=cols, figsize=figsz, subplot_kw=subplot_kw, facecolor='w')
    fig.suptitle('Satellite Imagery of the Sea Ice Breakup Process in Kotzebue Sound',y=0.95,fontsize=40)

    for idx in np.arange(0,cols):
        plot_MODIS_geotiff(axx[0,idx], paths_2007[idx], coastline, lon_min_MOD, lon_max_MOD, lat_min_MOD, lat_max_MOD )
        plot_MODIS_geotiff(axx[1,idx], paths_2012[idx], coastline, lon_min_MOD, lon_max_MOD, lat_min_MOD, lat_max_MOD )
        plot_MODIS_geotiff(axx[2,idx], paths_2019[idx], coastline, lon_min_MOD, lon_max_MOD, lat_min_MOD, lat_max_MOD )

        axx[0,idx].add_patch(patches.Ellipse(xy=(-162.6139,66.8968), width=1.1, height=0.44, edgecolor='r', linewidth=4, transform=ccrs.PlateCarree(), facecolor='none',zorder=400))
        axx[1,idx].add_patch(patches.Ellipse(xy=(-162.6139,66.8968), width=1.1, height=0.44, edgecolor='r', linewidth=4, transform=ccrs.PlateCarree(), facecolor='none',zorder=400))
        axx[2,idx].add_patch(patches.Ellipse(xy=(-162.6139,66.8968), width=1.1, height=0.44, edgecolor='r', linewidth=4, transform=ccrs.PlateCarree(), facecolor='none',zorder=400))


    for idx in np.arange(0,cols-1):
        for row in np.arange(0,3):
            con = ConnectionPatch(xyA=(1,0.5), coordsA=axx[row,idx].transAxes,
                                  xyB=(0,0.5), coordsB=axx[row,idx+1].transAxes,
                                  arrowstyle='->',linewidth=6,mutation_scale=50)
            fig.add_artist(con)


    axx[0,0].set_title('May 24',fontsize=fontsz)
    axx[0,1].set_title('June 1',fontsize=fontsz)
    axx[0,2].set_title('June 5',fontsize=fontsz)
    axx[0,3].set_title('June 11',fontsize=fontsz)

    axx[1,0].set_title('May 24',fontsize=fontsz)
    axx[1,1].set_title('May 31',fontsize=fontsz)
    axx[1,2].set_title('June 7',fontsize=fontsz)
    axx[1,3].set_title('June 16',fontsize=fontsz)

    axx[2,0].set_title('April 23',fontsize=fontsz)
    axx[2,1].set_title('May 10',fontsize=fontsz)
    axx[2,2].set_title('May 16',fontsize=fontsz)
    axx[2,3].set_title('May 25',fontsize=fontsz)

    plt.figtext(x=0.1,y=0.745,s='2007',rotation=90,fontsize=40)
    plt.figtext(x=0.1,y=0.475,s='2012',rotation=90,fontsize=40)
    plt.figtext(x=0.1,y=0.21,s='2019',rotation=90,fontsize=40)

#########################################################################################################

def plot_measurement_stations(image_path, image_2path, coastline_path, river_path, figsz, lon_min_MOD, lon_max_MOD, lat_min_MOD, lat_max_MOD, sis_color, obt_color, fontsz):
    '''
    Function to plot simple map of the locations and deployment durations of the OBT and SIS
    '''
    gdal.UseExceptions()

    coastline = ShapelyFeature(Reader(coastline_path).geometries(),ccrs.PlateCarree())
    river = ShapelyFeature(Reader(river_path).geometries(),ccrs.PlateCarree())

    #open tiff file and extract data, geotransform, and crs
    ds = gdal.Open(image_path)
    data = ds.ReadAsArray()
    gt = ds.GetGeoTransform()
    proj = ds.GetProjection()

    inproj = osr.SpatialReference()
    inproj.ImportFromWkt(proj)
    projcs = inproj.GetAuthorityCode('PROJCS')
    projection = ccrs.epsg(projcs)
    subplot_kw = dict(projection=projection)

    #initialize figure
    fig, axx = plt.subplots(figsize=figsz, subplot_kw=subplot_kw)

    extent = (gt[0], gt[0] + ds.RasterXSize * gt[1],
                  gt[3] + ds.RasterYSize * gt[5], gt[3])

    img = axx.imshow(data[:3, :, :].transpose((1, 2, 0)), extent=extent,
                    origin='upper')

    axx.set_extent([lon_min_MOD, lon_max_MOD, lat_min_MOD, lat_max_MOD])

    plot_MODIS_geotiff(axx,image_2path, coastline, lon_min_MOD, lon_max_MOD, lat_min_MOD, lat_max_MOD)

    #add coastlines and features
    axx.add_feature(coastline,zorder=300, edgecolor='k', facecolor='#efebd3', alpha=1)
    axx.add_feature(river,zorder=350, edgecolor='#46d3f6', facecolor='none', alpha=1,linewidth=4)

    #add title to map
    axx.text(0.5, 0.99, 'Measurement Stations \n in Kotzebue Sound', fontsize=45, fontname='Calibri', horizontalalignment='center', verticalalignment='top', transform=axx.transAxes,
             bbox=dict(boxstyle='square,pad=0.15', facecolor='w', alpha=0.8),zorder=400)
    #add data source to map
    axx.text(0.006, 0.01, ' Satellite Image From \n MODIS Visible, \n May 7 2019', fontsize=18, fontname='Calibri', horizontalalignment='left', verticalalignment='bottom', transform=axx.transAxes,
             bbox=dict(boxstyle='square,pad=0.15', facecolor='w', alpha=0.8),zorder=400)


    #add circles
    axx.add_patch(patches.Ellipse(xy=(-162.6139,66.8968), width=0.1, height=0.04, edgecolor=sis_color, linewidth=6, transform=ccrs.PlateCarree(), facecolor='none',zorder=400));
    axx.add_patch(patches.Ellipse(xy=(-163.7957,67.0598), width=0.1, height=0.04, edgecolor=obt_color, linewidth=6, transform=ccrs.PlateCarree(), facecolor='none',zorder=400));

    axx.text(0.57, 0.53, 'ITO', fontsize=fontsz, fontname='Calibri', horizontalalignment='left', verticalalignment='center', transform=axx.transAxes,
             bbox=dict(boxstyle='square,pad=0.3', facecolor=sis_color, edgecolor='k', linewidth=3,alpha=1),zorder=400)
    axx.text(0.57, 0.47, 'Jan 2019 - Apr 2019', fontsize=fontsz-2, fontname='Calibri', horizontalalignment='left', verticalalignment='center', transform=axx.transAxes,
             bbox=dict(boxstyle='square,pad=0.3', facecolor=sis_color, edgecolor='k', linewidth=3,alpha=1),zorder=400)
    axx.text(0.57, 0.41, '66.8968 N, 162.6139 W', fontsize=fontsz-2, fontname='Calibri', horizontalalignment='left', verticalalignment='center', transform=axx.transAxes,
             bbox=dict(boxstyle='square,pad=0.3', facecolor=sis_color, edgecolor='k', linewidth=3,alpha=1),zorder=400)

    axx.text(0.3, 0.66, 'OBT', fontsize=fontsz, fontname='Calibri', horizontalalignment='right', verticalalignment='center', transform=axx.transAxes,
             bbox=dict(boxstyle='square,pad=0.3', facecolor=obt_color, edgecolor='k', linewidth=3,alpha=1),zorder=400)
    axx.text(0.3, 0.6, 'Sep 2017 - Jun 2019', fontsize=fontsz-2, fontname='Calibri', horizontalalignment='right', verticalalignment='center', transform=axx.transAxes,
             bbox=dict(boxstyle='square,pad=0.3', facecolor=obt_color, edgecolor='k', linewidth=3,alpha=1),zorder=400)
    axx.text(0.3, 0.54, '67.0598 N, 163.7957 W', fontsize=fontsz-2, fontname='Calibri', horizontalalignment='right', verticalalignment='center', transform=axx.transAxes,
             bbox=dict(boxstyle='square,pad=0.3', facecolor=obt_color, edgecolor='k', linewidth=3,alpha=1),zorder=400)

#########################################################################################################

def plot_may2019_image(image_path, image_2path, coastline_path, river_path, figsz, lon_min_MOD, lon_max_MOD, lat_min_MOD, lat_max_MOD, sis_color, obt_color, fontsz):
    '''
    Function to plot simple map of the locations and deployment durations of the OBT and SIS
    '''
    gdal.UseExceptions()

    coastline = ShapelyFeature(Reader(coastline_path).geometries(),ccrs.PlateCarree())
    river = ShapelyFeature(Reader(river_path).geometries(),ccrs.PlateCarree())

    #open tiff file and extract data, geotransform, and crs
    ds = gdal.Open(image_path)
    data = ds.ReadAsArray()
    gt = ds.GetGeoTransform()
    proj = ds.GetProjection()

    inproj = osr.SpatialReference()
    inproj.ImportFromWkt(proj)
    projcs = inproj.GetAuthorityCode('PROJCS')
    projection = ccrs.epsg(projcs)
    subplot_kw = dict(projection=projection)

    #initialize figure
    fig, axx = plt.subplots(figsize=figsz, subplot_kw=subplot_kw)

    extent = (gt[0], gt[0] + ds.RasterXSize * gt[1],
                  gt[3] + ds.RasterYSize * gt[5], gt[3])

    img = axx.imshow(data[:3, :, :].transpose((1, 2, 0)), extent=extent,
                    origin='upper')

    axx.set_extent([lon_min_MOD, lon_max_MOD, lat_min_MOD, lat_max_MOD])

    plot_MODIS_geotiff(axx,image_2path, coastline, lon_min_MOD, lon_max_MOD, lat_min_MOD, lat_max_MOD)

    #add coastlines and features
    axx.add_feature(coastline,zorder=300, edgecolor='k', facecolor='#efebd3', alpha=1)
    axx.add_feature(river,zorder=350, edgecolor='#46d3f6', facecolor='none', alpha=1,linewidth=4)

    #add data source to map
    axx.text(0.006, 0.01, ' Satellite Image From \n MODIS Visible, \n May 7 2019', fontsize=18, fontname='Calibri', horizontalalignment='left', verticalalignment='bottom', transform=axx.transAxes,
             bbox=dict(boxstyle='square,pad=0.15', facecolor='w', alpha=0.8),zorder=400)

    #label features on plot
    axx.text(0.7, 0.475, 'Kobuk \n River', fontsize=18, transform=axx.transAxes, zorder=400)
    axx.text(0.46, 0.75, 'Noatak \n River', fontsize=18, transform=axx.transAxes, zorder=400)
    axx.text(0.538, 0.545, 'Kotzebue', fontsize=13, transform=axx.transAxes, zorder=400)
    axx.add_patch(patches.Arrow(x=0.55, y=0.56, dx=-0.018, dy=0.03, width=0.01, facecolor='k',transform=axx.transAxes,zorder=400));
