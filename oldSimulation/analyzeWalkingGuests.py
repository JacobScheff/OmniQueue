# analyze_walking_guests.py
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

def load_data(filename='walking_guests_stats.csv'):
    times = []
    run_data =[]
    
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        
        # Find exactly which columns contain the 'Run_' data
        run_indices =[i for i, col in enumerate(header) if col.startswith('Run_')]
        
        for row in reader:
            times.append(row[0])  # Time string (e.g., "08:00 AM")
            runs = [int(row[i]) for i in run_indices]
            run_data.append(runs)
            
    return times, np.array(run_data)

def main():
    try:
        times, data = load_data()
    except FileNotFoundError:
        print("Error: 'walking_guests_stats.csv' not found.")
        return

    num_intervals, num_runs = data.shape

    # --- Plot Setup ---
    fig, ax = plt.subplots(figsize=(12, 7))
    plt.subplots_adjust(bottom=0.35, right=0.75) # Extra bottom room for 2 sliders
    
    # Stats Text Box (Attached to figure, not axes)
    stats_box = fig.text(0.78, 0.5, '', fontsize=12, va='center', 
                         bbox=dict(boxstyle='round,pad=1', facecolor='lightgray', alpha=0.5))

    # --- Interactive UI: Time Slider ---
    ax_time_slider = plt.axes([0.15, 0.20, 0.55, 0.03])
    time_slider = Slider(
        ax=ax_time_slider,
        label='Time Scrubber',
        valmin=0,
        valmax=num_intervals - 1,
        valinit=0,
        valstep=1
    )
    time_slider.valtext.set_visible(False)

    # --- Interactive UI: Bucket Size Slider ---
    ax_bin_slider = plt.axes([0.15, 0.12, 0.55, 0.03])
    bin_slider = Slider(
        ax=ax_bin_slider,
        label='Bucket Size',
        valmin=5,
        valmax=500,
        valinit=50, # Default bucket range
        valstep=5
    )

    # --- Interactive UI: Prev/Next Buttons ---
    ax_prev = plt.axes([0.3, 0.04, 0.1, 0.04])
    ax_next = plt.axes([0.45, 0.04, 0.1, 0.04])
    btn_prev = Button(ax_prev, 'Previous Time')
    btn_next = Button(ax_next, 'Next Time')

    def update_plot(val=None):
        idx = int(time_slider.val)
        bucket_size = int(bin_slider.val)
        current_data = data[idx]
        
        # 1. Clear the old histogram
        ax.clear()
        
        # 2. Calculate DYNAMIC bins based on the current data bounds
        min_val = np.min(current_data)
        max_val = np.max(current_data)
        
        # Floor and Ceil the bounds to align perfectly with the bucket size
        bin_start = (min_val // bucket_size) * bucket_size
        bin_end = ((max_val // bucket_size) + 2) * bucket_size
        bins = np.arange(bin_start, bin_end, bucket_size)
        
        # 3. Draw new Histogram and capture counts for Y-axis scaling
        counts, _, _ = ax.hist(current_data, bins=bins, color='skyblue', edgecolor='black', alpha=0.8)
        
        # 4. Format Axes (DYNAMIC AUTO-SCALING)
        # Add a little padding to the left and right of the data
        ax.set_xlim(max(0, bin_start - bucket_size), bin_end)
        
        # Add +1 padding to the top of the tallest bar
        max_freq = np.max(counts) if len(counts) > 0 else 1
        ax.set_ylim(0, max_freq + 1)
        
        # Force Y-axis to use integers (since frequency is a count of trials)
        ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
        
        ax.set_xlabel(f"Number of Guests Walking (Buckets of {bucket_size})")
        ax.set_ylabel("Frequency (Number of Trials)")
        ax.set_title(f"Distribution of Walking Guests at {times[idx]}", fontsize=16, fontweight='bold')
        
        # 5. Calculate & Update Statistics Box
        mean = np.mean(current_data)
        std = np.std(current_data)
        minimum = np.min(current_data)
        q1 = np.percentile(current_data, 25)
        median = np.median(current_data)
        q3 = np.percentile(current_data, 75)
        maximum = np.max(current_data)
        
        stats_text = (
            f"Time: {times[idx]}\n\n"
            f"Mean: {mean:.1f}\n"
            f"Std Dev: {std:.1f}\n"
            f"------------------\n"
            f"Max: {maximum}\n"
            f"Q3 (75%): {q3:.1f}\n"
            f"Median: {median:.1f}\n"
            f"Q1 (25%): {q1:.1f}\n"
            f"Min: {minimum}"
        )
        stats_box.set_text(stats_text)
        fig.canvas.draw_idle()

    # --- Event Listeners ---
    time_slider.on_changed(update_plot)
    bin_slider.on_changed(update_plot)

    def next_time(event):
        current_val = int(time_slider.val)
        if current_val < num_intervals - 1:
            time_slider.set_val(current_val + 1)

    def prev_time(event):
        current_val = int(time_slider.val)
        if current_val > 0:
            time_slider.set_val(current_val - 1)

    btn_next.on_clicked(next_time)
    btn_prev.on_clicked(prev_time)

    # Initialize first view
    update_plot()

    plt.show()

if __name__ == "__main__":
    main()