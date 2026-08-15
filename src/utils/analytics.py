"""
Analytics utilities for analyzing experimental results.
"""

import csv
import matplotlib.pyplot as plt
import os

def analyze_batch_size_changes():
    """
    Analyze batch size change history and time.
    Reads the CSV file containing timestamp events and 
    analyzes patterns in batch size changes.
    """
    csv_file = "timeslice_events.csv"
    
    if not os.path.exists(csv_file):
        print(f"Error: Could not find event log file {csv_file}")
        return
    
    # Load data
    events = []
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            row['time_seconds'] = float(row['time_seconds'])
            if row['batch_idx'] and row['batch_idx'] != '':
                row['batch_idx'] = int(row['batch_idx'])
            if row['batch_size'] and row['batch_size'] != '':
                row['batch_size'] = int(row['batch_size'])
            events.append(row)
    
    # Extract batch size change events
    batch_size_changes = [e for e in events if 'batch size change' in e['message']]
    
    print("\n===== Batch size change analysis =====")
    print(f"Total batch size change events: {len(batch_size_changes)}")
    
    if batch_size_changes:
        # Visualize changes
        print("\nBatch size change history:")
        for i, change in enumerate(batch_size_changes):
            print(f"  {i+1}. {change['timestamp']} - {change['message']}")
        
        # Calculate intervals
        if len(batch_size_changes) > 1:
            intervals = []
            for i in range(1, len(batch_size_changes)):
                interval = batch_size_changes[i]['time_seconds'] - batch_size_changes[i-1]['time_seconds']
                intervals.append(interval)
            
            avg_interval = sum(intervals) / len(intervals)
            print(f"\nAverage batch size change interval: {avg_interval:.2f} seconds")
            
            # Additional statistics
            min_interval = min(intervals)
            max_interval = max(intervals)
            print(f"Minimum interval: {min_interval:.2f} seconds")
            print(f"Maximum interval: {max_interval:.2f} seconds")
            
            # Try to create a visual representation if matplotlib is available
            try:
                plt.figure(figsize=(10, 6))
                plt.plot(range(1, len(intervals) + 1), intervals, marker='o')
                plt.title('Batch Size Change Intervals')
                plt.xlabel('Change Number')
                plt.ylabel('Interval (seconds)')
                plt.grid(True)
                plt.savefig('batch_size_intervals.png')
                print(f"Saved interval plot to batch_size_intervals.png")
            except Exception as e:
                print(f"Could not create visualization: {e}")
    
    return batch_size_changes