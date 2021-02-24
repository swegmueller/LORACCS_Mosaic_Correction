import os
import fiona
import geopandas as gpd
from geopandas import GeoDataFrame, GeoSeries
from loess.loess_1d import loess_1d        
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import matplotlib.patches as mpatches
import numpy as np
from osgeo import gdal, gdal_array
import pandas as pd 
import rasterio
import rasterio.mask
from shapely.geometry import Polygon, LineString  

class LORACCS():
    '''
    Normalizes an image using the LORACCS method, and saves the image as a tif file into the
    given directory. Developed for Planet imagery, the default parameters are set for 4-band 
    images. However, the method is agnostic to the data, and optional parameters can be adjusted 
    for different sensors. The program was written to be able to accept any number of band
    combinations. NOTE: images are written using UInt16 -- including bands with float values
    is not recommended. 

    Required parameters:
    - outdir = directory into which files will be written
    - ref_img_fp = reference image file path
    - tgt_img_fp = target image (image to be normalized) filepath

    Optional paremeters:
    - band_list = list of bands as integers, default is [1, 2, 3, 4]
    - band_names = list of band names as strings, default is ['Blue', 'Green', 'Red', 'NIR']
    - max_spectra = list of maximum reasonable spectral values in the images. This is used to 
                    mitigate errors and avoid excessive processing time caused by bad pixels.
                    Default is [3000, 3000, 3000, 8000], corresponding to BGR and NIR, respectively. 
    - loess_frac = float, percent of pixel values to consider when fitting model. Default is 0.15.
                   Higher values will result in a less conformed curve (may risk underfitting), and
                   lower values more conformed (may risk overfitting). Recommend adjusting up by 0.05
                   increments if results are poor.

    IMPORTANT: The optional parameters above MUST be the same length. For example, if 5 bands are
               used, five names and five max_spectra must also be provided.

    Other parameters:
    - delete_working_files= Boolean, whether or not to delete the files generated
                            during the program. If True, the transformed image, graph of the model, 
                            and dataframe with normalized RMSE values are retained 
                            (original imagery is not affected). If False, all working files
                            will remain in the directory assinged to "outdir". Default is
                            True.
    '''        
        
        
    def __init__(self, outdir, ref_img_fp, tgt_img_fp,
                 band_list=[1, 2, 3, 4], 
                 band_names=['Blue', 'Green', 'Red', 'NIR'],
                 max_spectra = [3000, 3000, 3000, 8000],
                 loess_frac=0.15,
                 delete_working_files=True):   
        
        os.chdir(outdir)
        
        self.get_overlap_areas(ref_img_fp, tgt_img_fp, outdir)          

        # Do LORACCS correction on target image      
        ref_img = gdal.Open('Ref_training_pixels.tif')
        tgt_img = gdal.Open('Tgt_training_pixels.tif')
        
        loraccs_img_fp = 'LORACCS_normalized_img.tif' # File name of transformed image
        tgt_img_2 = rasterio.open(tgt_img_fp) # Will pull metadata from original image for LORACCS image
                
        for num, band in enumerate(band_list):
            # Get the band data as a numpy array
            ref_data = ref_img.GetRasterBand(band)
            tgt_data = tgt_img.GetRasterBand(band)
            ref_img_band = gdal_array.BandReadAsArray(ref_data)
            tgt_img_band = gdal_array.BandReadAsArray(tgt_data)

            # Do LORACCS normalization
            loraccs_array = self.run_loraccs(ref_img_band, tgt_img_band, band, band_names[num],
                                             max_spectra[num], loess_frac, tgt_img_fp, outdir)

            # Write transformed array to new image
            with rasterio.Env():
                profile = tgt_img_2.profile
                profile.update(nodata=0)

                if num == 0:
                    with rasterio.open(loraccs_img_fp, 'w', **profile) as dst:
                        dst.write(loraccs_array, band)
                else:
                    with rasterio.open(loraccs_img_fp, 'r+', **profile) as dst:
                        dst.write(loraccs_array, band)
                        
        # Get the diagonal pixels from the LORACCS image to run quality assessment
        self.get_qa_pixels('overlap.shp', loraccs_img_fp, outdir)
            
        # Generate NRMSE file 
        self.nrmse = self.get_NRMSE(band_list, band_names, outdir)
        print(self.nrmse)
        
        # Print completed message
        print('LORACCS transformation complete. New image file can be found in the given directory.')
        print('File name: LORACCS_normalized_img.tif')
            
        # Delete working files
        if delete_working_files == True:
            print('Removing working files.')
            
            working_files = ['Ref_training_pixels.tif', 'QA_diag_lines.shp',
                             'Reference_clip_overlap.tif', 'overlap.prj', 'QA_diag_lines.cpg',
                             'LORACCS_image_overlap_clip.tif', 'QA_diag_lines.prj', 
                             'overlap.shp', 'QA_diag_lines.dbf', 'Ref_assessment_pixels.tif',
                             'overlap.cpg', 'Tgt_assessment_pixels.tif', 'LORACCS_assessment_pixels.tif',
                             'Target_clip_overlap.tif', 'Tgt_training_pixels.tif', 'QA_diag_lines.shx',
                             'overlap.shx', 'overlap.dbf']

            for name in band_names:
                band_files = ['%s_2d_hist.png' %name,
                             '%s_df.csv' %name,
                             '%s full spectra.csv' %name]
                working_files.extend(band_files[:])

            for file in working_files:
                os.remove(file)
                
    def get_img_orientation(self, ref_img, tgt_img):
        '''
        Ascertaines the physical orientation of the images. Used with the
        get_overlap_areas.
        
        Ref image and tgt image are image files already opened in rasterio
        '''
               
        r_top = ref_img.bounds[3]
        t_top = tgt_img.bounds[3]

        if t_top > r_top:
            top = 'N'
        else: 
            top = 'S'

        r_left = ref_img.bounds[0]
        t_left = tgt_img.bounds[0]

        if t_left > r_left:
            left = 'E'
        else: 
            left = 'W'

        tgt_loc = top+left
        
        return tgt_loc
        
    
    def get_overlap_areas(self, ref_img_fp, tgt_img_fp, outdir):
        '''
        Gets the overlapping area between images, and crops the images to that
        area. Also creates diagonal lines across the overlap, and sets aside
        the pixels along those lines for QA metrics.
        '''
        
        os.chdir(outdir)
        
        tgt_img = rasterio.open(tgt_img_fp)
        ref_img = rasterio.open(ref_img_fp)
        
        tgt_loc = self.get_img_orientation(ref_img, tgt_img)
        
        if tgt_loc == 'NW':
            # IF THE TARGET IMAGE IS MORE NORTH AND WEST OF THE REF IMAGE
            ulc = (ref_img.bounds[0], ref_img.bounds[3])
            urc = (tgt_img.bounds[2], ref_img.bounds[3])
            blc = (ref_img.bounds[0], tgt_img.bounds[1])
            brc = (tgt_img.bounds[2], tgt_img.bounds[1])
            overlap_extent_bounds = [ulc, blc, brc, urc]
        elif tgt_loc == 'NE':
            # IF THE TARGET IMAGE IS MORE NORTH AND EAST OF THE REF IMAGE 
            ulc = (tgt_img.bounds[0], ref_img.bounds[3])
            urc = (ref_img.bounds[2], ref_img.bounds[3])
            blc = (tgt_img.bounds[0], tgt_img.bounds[1])
            brc = (ref_img.bounds[2], tgt_img.bounds[1])
            overlap_extent_bounds = [ulc, blc, brc, urc]
        elif tgt_loc =='SE':
            # IF THE TARGET IMAGE IS MORE SOUTH AND EAST OF THE REF IMAGE
            ulc = (tgt_img.bounds[0], tgt_img.bounds[3])
            urc = (ref_img.bounds[2], tgt_img.bounds[3])
            blc = (tgt_img.bounds[0], ref_img.bounds[1])
            brc = (ref_img.bounds[2], ref_img.bounds[1])
            overlap_extent_bounds = [ulc, blc, brc, urc]
        else:
            # IF THE TARGET IMAGE IS MORE SOUTH AND WEST OF THE REF IMAGE 
            ulc = (ref_img.bounds[0], tgt_img.bounds[3])
            urc = (tgt_img.bounds[2], tgt_img.bounds[3])
            blc = (ref_img.bounds[0], ref_img.bounds[1])
            brc = (tgt_img.bounds[2], ref_img.bounds[1])
            overlap_extent_bounds = [ulc, blc, brc, urc]
            
        # CROP IMAGES 
        
        # Create a polyon to clip to
        overlap_poly = GeoSeries(Polygon(overlap_extent_bounds))
        overlap_shp = GeoDataFrame(geometry=overlap_poly)
        overlap_shp.crs = ref_img.crs

        overlap_shp_fp = 'overlap.shp'
        overlap_shp.to_file(overlap_shp_fp)
        
        # Clip images to overlap area
        ref_clip = 'Reference_clip_overlap.tif'
        tgt_clip = 'Target_clip_overlap.tif'
        
        self.crop_plot(overlap_shp_fp, tgt_img_fp, tgt_clip)
        self.crop_plot(overlap_shp_fp, ref_img_fp, ref_clip) 
        
        # Create diagonal lines
        diag_line1 = LineString([blc, urc])
        diag_line2 = LineString([ulc, brc])

        diag_lines = GeoDataFrame(geometry=[diag_line1, diag_line2])
        diag_lines.crs = ref_img.crs

        diag_lines_fp = 'QA_diag_lines.shp'
        diag_lines.to_file(diag_lines_fp) 
        
        # Select out pixels along diagonal lines; mask them for training

        # Reference Image
        ref_assessment_pixels = os.path.join(outdir, 'Ref_assessment_pixels.tif')
        ref_training_pixels = os.path.join(outdir, 'Ref_training_pixels.tif')

        with fiona.open(diag_lines_fp, 'r') as shapefile:
            features = [feature['geometry'] for feature in shapefile]

        with rasterio.open(ref_clip) as src:
            asmt_image, asmt_transform = rasterio.mask.mask(src, features)
            trg_image, trg_transform = rasterio.mask.mask(src, features, invert=True)
            ref_meta = src.meta.copy()

        ref_meta.update({"driver": "GTiff",
                         "height": asmt_image.shape[1],
                         "width": asmt_image.shape[2],
                         "transform": asmt_transform,
                         "nodata":0})

        with rasterio.open(ref_assessment_pixels, "w", **ref_meta) as dest:
            dest.write(asmt_image)

        with rasterio.open(ref_training_pixels, "w", **ref_meta) as dest:
            dest.write(trg_image)

        # Target Image - Uncorrected   
        tgt_assessment_pixels = os.path.join(outdir, 'Tgt_assessment_pixels.tif')
        tgt_training_pixels = os.path.join(outdir, 'Tgt_training_pixels.tif')

        with fiona.open(diag_lines_fp, 'r') as shapefile:
            features = [feature['geometry'] for feature in shapefile]

        with rasterio.open(tgt_clip) as src:
            asmt_image, asmt_transform = rasterio.mask.mask(src, features)
            trg_image, trg_transform = rasterio.mask.mask(src, features, invert=True)
            tgt_meta = src.meta.copy()

        tgt_meta.update({"driver": "GTiff",
                         "height": asmt_image.shape[1],
                         "width": asmt_image.shape[2],
                         "transform": asmt_transform,
                         "nodata":0})

        with rasterio.open(tgt_assessment_pixels, "w", **tgt_meta) as dest:
            dest.write(asmt_image)

        with rasterio.open(tgt_training_pixels, "w", **tgt_meta) as dest:
            dest.write(trg_image)
    
    
    def crop_plot(self, shape, org_img, crop_file):
        ''' 
        Crops orignal imagery to extent of desired shape.
        Input a shapefile with ONE feature
        Shape = shapefile to be used for cropping
        Org_img = original imagery to be cropped
        crop_file = File for cropped imagery to be written to
        ''' 

        with fiona.open(shape, 'r') as shapefile:
            features = [feature['geometry'] for feature in shapefile]

        with rasterio.open(org_img) as src:
            out_image, out_transform = rasterio.mask.mask(src, features, crop=True)
            out_meta = src.meta.copy()

        out_meta.update({"driver": "GTiff",
                         "height": out_image.shape[1],
                         "width": out_image.shape[2],
                         "transform": out_transform,
                         "nodata":0})

        with rasterio.open(crop_file, "w", **out_meta) as dest:
            dest.write(out_image)

        shape=None

    def get_new_spec_val(self, x1, x2, y1, y2, new_x):
        '''
        Function to extrapolate missing spectral values in LORACCS model
        by drawing a line between the nearest LOESS values
        '''
        m = (float(y2)-float(y1))/(float(x2)-float(x1))
        b = y1 - (m*float(x1))
        new_y = (m*new_x) + b
    
        return int(new_y)

    def run_loraccs(self, ref_img_band, tgt_img_band, band_num, band_name, 
                    band_max_spectra, loess_frac, tgt_img_fp, outdir):
        
        '''
        Runs the LORACCS method.
        '''
        
        os.chdir(outdir)

        # Plot 2d histogram
        index = (ref_img_band>0)&(tgt_img_band>0)
        ref_img_band_sub = ref_img_band[index]
        tgt_img_band_sub = tgt_img_band[index]

        plt.hist2d(tgt_img_band_sub, ref_img_band_sub, bins=200, cmin = 5, cmap=plt.cm.jet, )
        plt.colorbar()
        plt.title('%s Band 2D Histogram' %band_name)
        plt.xlabel('Target')
        plt.ylabel('Reference')
        save_fig = '%s_2d_hist.png' %band_name
        plt.savefig(save_fig)
        plt.show()   

        ### Extract spectral values into a dict

        # Get unique values from target image
        tgt_uniq = np.unique(tgt_img_band)
        counts_dict = dict()
        for uniq in tgt_uniq:
            counts_dict[uniq] = []

        img_rows = range(0, tgt_img_band.shape[0])
        img_row_pixel = range(0, tgt_img_band.shape[1])

        for band_row in img_rows:       # iterate through rows
            for pixel in img_row_pixel: # iterate through pixels
                tgt_val = tgt_img_band[band_row][pixel]
                ref_val = ref_img_band[band_row][pixel]
                if tgt_val != 0:
                    if ref_val != 0:
                        # Add value to the dict
                        values = counts_dict[tgt_val]
                        try:
                            values.append(ref_val)
                        except:
                            values = ref_val
                else:
                    continue 

        # Generate stats
        for uniq in tgt_uniq:
            values = np.array(counts_dict[uniq])
            
            pixels = len(values)

            # Subset out values to get rid of outliers
            sub = np.sort(values)
            sub = sub[sub < band_max_spectra]
            val_sub = sub[int(len(sub) * .025) : int(len(sub) * .975)]

            try:
                mean = np.mean(val_sub)
                std = np.std(val_sub)
            except:
                print('Exception used')
                mean = np.mean(counts_dict[uniq])
                std = np.std(counts_dict[uniq]) 

            new_dict = {'values' : counts_dict[uniq], 'mean' : mean, 'std' : std, 'pixels' : pixels}
            counts_dict[uniq] = new_dict  

        # Create pandas DataFrame of values
        spec_vals = tgt_uniq
        mean = []
        std = []
        pix = []

        for uniq in tgt_uniq:
            mean.append(counts_dict[uniq]['mean'])
            std.append(counts_dict[uniq]['std'])
            pix.append(counts_dict[uniq]['pixels'])

        stats_df = pd.DataFrame()
        stats_df['Spec_vals'] = spec_vals
        stats_df['Mean'] = mean
        stats_df['Std'] = std
        stats_df['Pixels'] = pix
        # Remove all NaN
        stats_df = stats_df.fillna(0)
        stats_df_valid = stats_df[stats_df.Mean != 0]
        # Remove entries with pixel count less than 6
        stats_df_valid = stats_df_valid[stats_df_valid.Pixels > 5]

        ### Create model
        
        # Set up params for LOESS
        x = stats_df_valid.Spec_vals.values
        y = stats_df_valid.Mean.values

        # Run LOESS
        xout, yout, wout = loess_1d(x, y, frac=loess_frac, degree=2, rotate=False)

        # Save values into the dataframe
        stats_df_valid['Mean_LOESS'] = yout
        
        # Remove any bad LOESS values (rare)
        stats_df_valid = stats_df_valid[stats_df_valid['Mean_LOESS'].values < band_max_spectra].copy()
        stats_df_valid = stats_df_valid[stats_df_valid['Mean_LOESS'].values != 0].copy()
        
        # Save the data to CSV
        stats_df_valid.to_csv('%s_df.csv' %band_name, index=False)

        # Fill gaps in spectra
        min_spectra = min(stats_df_valid.Spec_vals.values)
        max_spectra = max(stats_df_valid.Spec_vals.values)

        if max_spectra > band_max_spectra:
            reasonable_spec_vals = stats_df_valid[stats_df_valid['Spec_vals'] < band_max_spectra]
            max_spectra = reasonable_spec_vals['Spec_vals'].values[-1]

        print('Maxiumum spectral value being set to: ', max_spectra)

        spectral_range = range(int(min_spectra), int(max_spectra+1))

        full_spectra = pd.DataFrame()
        full_spectra['Spec_vals'] = spectral_range
        full_spectra = full_spectra.merge(stats_df_valid, how='left', on='Spec_vals')
        full_spectra.drop(['Std'], axis=1, inplace=True)
        full_spectra.rename(columns={'Mean':'Org_Mean'}, inplace=True)
        full_spectra['Missing'] = pd.isna(full_spectra['Mean_LOESS']) # Identify missing spectral values
                
        all_y_values = []

        # Predict missing spectral values
        for item in range(0, len(full_spectra)):
                        
            if full_spectra['Missing'].iloc[item] == True:

                # Find nearest values on either side   
                invalid_before_value = True
                n = item
                while invalid_before_value == True:
                    n = n-1
                    invalid_before_value = full_spectra['Missing'].iloc[n]

                x1 = full_spectra['Spec_vals'].iloc[n]
                y1 = full_spectra['Mean_LOESS'].iloc[n]  

                n = item
                invalid_after_value = True
                while invalid_after_value == True:
                    n = n+1
                    invalid_after_value = full_spectra['Missing'].iloc[n]

                x2 = full_spectra['Spec_vals'].iloc[n]
                y2 = full_spectra['Mean_LOESS'].iloc[n]  

                # Predict new spectra value using the equation of a line between points
                new_x = full_spectra['Spec_vals'].iloc[item]
                new_y = self.get_new_spec_val(x1, x2, y1, y2, new_x)

            else: 
                new_y = full_spectra['Mean_LOESS'].iloc[item]
            
            all_y_values.append(new_y)
        
        full_spectra['Filled_LOESS']=all_y_values
        full_spectra.fillna(0, inplace=True)
        
        ### Write full spectra data frame to csv
        full_spectra.to_csv('%s full spectra.csv' %band_name, index=False)

        ### Plot result of LORACCS along with histogram
        fig, ax = plt.subplots(nrows=1, figsize=(6,4))

        for_plot = full_spectra.copy()
        for_plot = for_plot[for_plot['Missing'] == False]

        x=for_plot['Spec_vals'].values
        y1=for_plot['Filled_LOESS'].values
        y2=for_plot['Pixels'].values
        y3=for_plot['Org_Mean'].values

        # Plot histogram
        ax.bar(x, y2, width=1, color='lightgray')
        gray_patch = mpatches.Patch(color='lightgray', label='Histogram')

        # Set plot to have two y axes
        ax2 = ax.twinx()

        # Original target values as a scatterplot 
        ax2.scatter(x, y3, color='tab:gray', marker='.', label='Mean Reference')

        #LORACCS regression line
        ax2.plot(x, y1, color='tab:orange', label='LORACCS Target', linewidth=2)

        # Fix tick marks
        ylabs = ax2.get_yticks()
        ax2.yaxis.tick_left()
        ax2.set_yticklabels(ylabs, fontsize=13)
        ax2.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))

        y2labs = ax.get_yticks()

        ax.yaxis.tick_right()
        ax.set_yticklabels(y2labs, fontsize=13)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))

        xlabs = ax2.get_xticks()
        ax2.set_xticklabels(xlabs, fontsize=13)
        ax2.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))

        ax.set_title('LORACCS Model: %s Band' %band_name, fontsize=20)
        ax.set_xlabel('Target Spectral Values', fontsize=15)

        ax.yaxis.set_label_position('right')
        ax.set_ylabel('Reference Histogram', fontsize=15)        

        ax2.yaxis.set_label_position('left')
        ax2.set_ylabel('Reference Spectral Values', fontsize=15)

        ax.legend(fontsize=12, loc='upper left', handles=[gray_patch])
        ax2.legend(fontsize=12, loc='lower right')

        save_fig = '%s_LORACCS_full_spectra_plot.png' %band_name
        plt.savefig(save_fig)
        plt.show()

        ### Transform image using filled-in LORACCS function

        # Read in target image
        full_tgt_img = gdal.Open(tgt_img_fp)

        # Get bands
        band_data = full_tgt_img.GetRasterBand(band_num)

        # Read in as numpy arrays
        data = gdal_array.BandReadAsArray(band_data)

        spec_vals_dict = dict(zip(full_spectra.Spec_vals, full_spectra.Filled_LOESS))

        # Change the data type in preparation for changing values
        data = data.astype('float')

        # Loop through spectral values, replace with new value / 100000.  Division
        # necessary so already replaced values are not overwritten
        for spec_val in spec_vals_dict:   
            data[data == spec_val] = spec_vals_dict[spec_val] / 100000

        # Multiply by 100000 to restore proper values, return dtype
        data = data*100000
        data = data.astype('uint16')

        return data # Returns band array transformed by the LORACCS method
            
    def get_qa_pixels(self, overlap_shp_fp, loraccs_img_fp, outdir):
        '''
        Uses the diagonals generated to pull pixel values for comparission 
        with original imagery.
        '''
        
        os.chdir(outdir)
        
        # Target - Transformed (get pixels for quality assessment)  
        loraccs_clip = 'LORACCS_image_overlap_clip.tif'
        self.crop_plot(overlap_shp_fp, loraccs_img_fp, loraccs_clip)

        loraccs_assessment_pixels = 'LORACCS_assessment_pixels.tif'

        with fiona.open('QA_diag_lines.shp', 'r') as shapefile:
            features = [feature['geometry'] for feature in shapefile]

        with rasterio.open(loraccs_clip) as src:
            asmt_image, asmt_transform = rasterio.mask.mask(src, features)
            tgt_meta = src.meta.copy()

        tgt_meta.update({"driver": "GTiff",
                         "height": asmt_image.shape[1],
                         "width": asmt_image.shape[2],
                         "transform": asmt_transform,
                         "nodata":0})

        with rasterio.open(loraccs_assessment_pixels, "w", **tgt_meta) as dest:
            dest.write(asmt_image)
            
    def get_NRMSE (self, band_list, band_names, outdir):
        '''
        Returns a dataframe of mean-normalized RMSE values. Used to assess
        quality of a LORACCS-normalized image as compared to the original imagery.
        '''

        os.chdir(outdir)

        reference_file = 'Ref_assessment_pixels.tif'        
        org_file = 'Tgt_assessment_pixels.tif'
        loraccs_file = 'LORACCS_assessment_pixels.tif'

        # Set up dataframe for NRMSE values
        nrmse_df = pd.DataFrame(index=band_list)
        nrmse_df['Band Mean'] = None
        nrmse_df['Original NRMSE'] = None
        nrmse_df['LORACCS NRMSE'] = None

        # Get data and calculate NRMSE
        ref_img = gdal.Open(reference_file)
        org_img = gdal.Open(org_file)
        loraccs_img = gdal.Open(loraccs_file)

        for num, band in enumerate(band_list):
            band_num = band
            band_name = band_names[num]

            ref_data = ref_img.GetRasterBand(band)
            org_data = org_img.GetRasterBand(band)
            loraccs_data = loraccs_img.GetRasterBand(band)

            # Read in as numpy array
            ref_img_band = gdal_array.BandReadAsArray(ref_data)
            org_img_band = gdal_array.BandReadAsArray(org_data)
            loraccs_img_band = gdal_array.BandReadAsArray(loraccs_data)

            # Select values in array with data
            index = (ref_img_band>0)&(org_img_band>0)
            ref_img_band = np.array(ref_img_band[index])
            org_img_band = np.array(org_img_band[index])
            loraccs_img_band = np.array(loraccs_img_band[index])

            ref_img_band.ravel()
            org_img_band.ravel()
            loraccs_img_band.ravel()

            # Get band mean to use for scaling
            band_mean = np.mean(ref_img_band)
            nrmse_df['Band Mean'][band] = band_mean            

            pix_dif_array_org = abs(np.subtract((ref_img_band.astype(np.int16)),
                                                (org_img_band.astype(np.int16))))
            pix_dif_array_lor = abs(np.subtract((ref_img_band.astype(np.int16)),
                                                (loraccs_img_band.astype(np.int16))))            

            pix_dif_org = pix_dif_array_org.ravel()
            pix_dif_lor = pix_dif_array_lor.ravel()

            pix_dif_org.sort()
            pix_dif_lor.sort()

            pix_dif_org_test = pix_dif_org[int(len(pix_dif_org) * .05) : int(len(pix_dif_org) * .95)]
            pix_dif_lor_test = pix_dif_lor[int(len(pix_dif_lor) * .05) : int(len(pix_dif_lor) * .95)]

            # Scale by band mean
            scaled_pix_dif_org_test = pix_dif_org_test / band_mean
            scaled_pix_dif_lor_test = pix_dif_lor_test / band_mean

            pix_org_res_sq = abs(np.square(scaled_pix_dif_org_test))
            pix_lor_res_sq = abs(np.square(scaled_pix_dif_lor_test))            

            pix_org_res_ave = abs(np.mean(pix_org_res_sq))
            pix_lor_res_ave = abs(np.mean(pix_lor_res_sq))

            NRMSE_org = np.sqrt(pix_org_res_ave)
            NRMSE_lor = np.sqrt(pix_lor_res_ave)

            nrmse_df['Original NRMSE'][band] = NRMSE_org
            nrmse_df['LORACCS NRMSE'][band] = NRMSE_lor            


        nrmse_df['Pixel_Cnt'] = len(pix_dif_org)

        nrmse_df.to_csv('NRMSE_per_band.csv')

        return nrmse_df
