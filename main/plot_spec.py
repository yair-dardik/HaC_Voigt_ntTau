import HaC_main as HaC
import numpy as np
import matplotlib.pyplot as plt
import math

if __name__ == "__main__":
    # @CHANGE_ME: Set the experiment and frame numbers you want to analyze here
    exp_i = 559
    first_frame = 5
    last_frame = 50

    frame_amount = last_frame - first_frame + 1
    
    # Pixel range covering [MIN_WAVELENGTH_NM, MAX_WAVELENGTH_NM]
    _min_pix = max(0, int(np.floor(HaC.nm_to_pixel(HaC.MIN_WAVELENGTH_NM))))
    _max_pix = min(HaC.DETECTOR_X_MAX, int(np.ceil(HaC.nm_to_pixel(HaC.MAX_WAVELENGTH_NM))) + 1)
    
    # Calculate an optimal grid size (e.g., 6 columns, and as many rows as needed)
    ncols = 6
    nrows = math.ceil(frame_amount / ncols)
    
    # Create the massive figure (adjust figsize if it's too big/small for your monitor)
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 2 * nrows), sharex=True)
    axes = axes.flatten() # Flatten the 2D array of axes for easy iteration

    print(f"Generating {frame_amount} plots. Please wait...")

    for i, frame_i in enumerate(range(first_frame, last_frame + 1)):
        ax = axes[i]
        
        try:
            # Load the data
            tiff_path, _, _ = HaC.prompt_c3_tiff_path(exp_i, frame_i)
            x_px, profile = HaC.load_full_profile(tiff_path, HaC.Y_RANGE, x_min=_min_pix, x_max=_max_pix)
            
            # Convert pixels to wavelength
            x_nm = HaC.pixel_to_nm(x_px)
            
            # (Optional) If you want to view the baseline-corrected spectra instead of raw, uncomment this line:
            # profile, _, _ = HaC.apply_baseline_correction(x_px, profile)

            # Plot on the corresponding subplot
            ax.plot(x_nm, profile, color='black', linewidth=1)
            ax.set_title(f"Frame {frame_i}", fontsize=10)
            ax.grid(True, linestyle=':', alpha=0.6)
            
            # Hide Y-axis labels to reduce clutter (optional, remove if you want to see raw intensity numbers)
            ax.set_yticks([]) 

        except Exception as e:
            ax.set_title(f"Frame {frame_i} (Error)", fontsize=10, color='red')
            ax.text(0.5, 0.5, 'File Not Found', ha='center', va='center', transform=ax.transAxes, color='red')

    # Add a global X-axis label to the bottom of the figure
    fig.supxlabel("Wavelength (nm)", fontsize=14)
    fig.suptitle(f"Experiment C{exp_i} Full Spectra (Frames {first_frame}-{last_frame})", fontsize=16)

    # Hide any extra empty subplots if your frames don't perfectly fill the grid
    for j in range(frame_amount, len(axes)):
        axes[j].set_visible(False)
        
    plt.tight_layout()
    plt.subplots_adjust(top=0.95) # Make slightly more room for the main title
    plt.show()