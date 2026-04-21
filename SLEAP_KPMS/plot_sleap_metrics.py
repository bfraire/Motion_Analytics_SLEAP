import os
import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def parse_metrics_file(filepath):
    """Parse a SLEAP metrics txt file and extract all metrics."""
    metrics = {
        'filename': os.path.basename(filepath),
        'folder': os.path.basename(os.path.dirname(filepath))
    }
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Extract visibility metrics
    visibility_patterns = {
        'true_positives': r'true_positives:\s+(\d+)',
        'false_positives': r'false_positives:\s+(\d+)',
        'true_negatives': r'true_negatives:\s+(\d+)',
        'false_negatives': r'false_negatives:\s+(\d+)',
        'precision': r'precision:\s+([\d.]+)',
        'recall': r'recall:\s+([\d.]+)',
        'f1_score': r'f1_score:\s+([\d.]+)'
    }
    
    for key, pattern in visibility_patterns.items():
        match = re.search(pattern, content)
        if match:
            metrics[key] = float(match.group(1))
    
    # Extract localization metrics
    localization_patterns = {
        'oks_voc_mAP': r'oks_voc_mAP:\s+([\d.]+)',
        'pck_voc_mAP': r'pck_voc_mAP:\s+([\d.]+)'
    }
    
    for key, pattern in localization_patterns.items():
        match = re.search(pattern, content)
        if match:
            metrics[key] = float(match.group(1))
    
    # Extract distance error statistics
    distance_patterns = {
        'n_samples': r'n_samples:\s+(\d+)',
        'mean': r'mean:\s+([\d.]+)',
        'std': r'std:\s+([\d.]+)',
        'min': r'min:\s+([\d.]+)',
        'max': r'max:\s+([\d.]+)',
        'p25': r'p25:\s+([\d.]+)',
        'p50_median': r'p50_median:\s+([\d.]+)',
        'p75': r'p75:\s+([\d.]+)',
        'p90': r'p90:\s+([\d.]+)',
        'p95': r'p95:\s+([\d.]+)',
        'p99': r'p99:\s+([\d.]+)'
    }
    
    for key, pattern in distance_patterns.items():
        match = re.search(pattern, content)
        if match:
            metrics[key] = float(match.group(1))
    
    # Extract distance error values
    distance_section = re.search(
        r'DISTANCE ERROR VALUES.*?={80}\n([\d.\n]+)',
        content,
        re.DOTALL
    )
    if distance_section:
        values_text = distance_section.group(1).strip()
        metrics['distance_errors'] = [float(v) for v in values_text.split('\n') if v.strip()]
    
    return metrics

def collect_all_metrics(base_folder):
    """Collect metrics from all txt files in subfolders."""
    all_metrics = []
    
    for root, dirs, files in os.walk(base_folder):
        for file in files:
            if file.endswith('.txt'):
                filepath = os.path.join(root, file)
                try:
                    metrics = parse_metrics_file(filepath)
                    all_metrics.append(metrics)
                    print(f"Parsed: {metrics['folder']}/{metrics['filename']}")
                except Exception as e:
                    print(f"Error parsing {filepath}: {e}")
    
    return all_metrics

def plot_metrics_comparison(all_metrics, output_folder='comparison_plots'):
    """Create comparison plots for all metrics."""
    os.makedirs(os.path.join(base_folder, output_folder), exist_ok=True)
    
    # Extract model names
    model_names = [m['folder'] for m in all_metrics]
    
    # 1. Visibility Metrics Comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Visibility Metrics Comparison', fontsize=16, fontweight='bold')
    
    metrics_to_plot = [
        ('precision', 'Precision'),
        ('recall', 'Recall'),
        ('f1_score', 'F1 Score'),
    ]
    
    for idx, (metric, title) in enumerate(metrics_to_plot):
        ax = axes[idx // 2, idx % 2]
        values = [m.get(metric, 0) for m in all_metrics]
        bars = ax.bar(model_names, values, color='steelblue', alpha=0.7)
        ax.set_ylabel('Score', fontweight='bold')
        ax.set_title(title, fontweight='bold')
        ax.set_ylim(0, 1)
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Confusion matrix metrics
    ax = axes[1, 1]
    cm_metrics = ['true_positives', 'false_positives', 'true_negatives', 'false_negatives']
    x = np.arange(len(model_names))
    width = 0.2
    
    for i, metric in enumerate(cm_metrics):
        values = [m.get(metric, 0) for m in all_metrics]
        ax.bar(x + i*width, values, width, label=metric.replace('_', ' ').title(), alpha=0.7)
    
    ax.set_ylabel('Count', fontweight='bold')
    ax.set_title('Confusion Matrix Counts', fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(base_folder, output_folder, 'visibility_metrics.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Localization Metrics
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Localization Metrics Comparison', fontsize=16, fontweight='bold')
    
    loc_metrics = [
        ('oks_voc_mAP', 'OKS VOC mAP'),
        ('pck_voc_mAP', 'PCK VOC mAP')
    ]
    
    for idx, (metric, title) in enumerate(loc_metrics):
        ax = axes[idx]
        values = [m.get(metric, 0) for m in all_metrics]
        bars = ax.bar(model_names, values, color='coral', alpha=0.7)
        ax.set_ylabel('mAP Score', fontweight='bold')
        ax.set_title(title, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(base_folder, output_folder, 'localization_metrics.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Distance Error Statistics
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Distance Error Statistics Comparison', fontsize=16, fontweight='bold')
    
    stat_metrics = [
        ('mean', 'Mean Error'),
        ('std', 'Std Dev'),
        ('p50_median', 'Median Error'),
        ('p95', '95th Percentile')
    ]
    
    for idx, (metric, title) in enumerate(stat_metrics):
        ax = axes[idx // 2, idx % 2]
        values = [m.get(metric, 0) for m in all_metrics]
        bars = ax.bar(model_names, values, color='seagreen', alpha=0.7)
        ax.set_ylabel('Distance (pixels)', fontweight='bold')
        ax.set_title(title, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}', ha='center', va='bottom', fontsize=9)
        
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(base_folder,output_folder, 'distance_statistics.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Distance Error Distributions (Box Plot)
    fig, ax = plt.subplots(figsize=(12, 6))
    
    error_data = [m.get('distance_errors', []) for m in all_metrics]
    bp = ax.boxplot(error_data, labels=model_names, patch_artist=True)
    
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
        patch.set_alpha(0.7)
    
    ax.set_ylabel('Distance Error (pixels)', fontweight='bold')
    ax.set_title('Distance Error Distribution Comparison', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(base_folder, output_folder, 'distance_distributions.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Percentile Comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    
    percentiles = ['p25', 'p50_median', 'p75', 'p90', 'p95', 'p99']
    x = np.arange(len(percentiles))
    width = 0.8 / len(all_metrics)
    
    for i, metrics in enumerate(all_metrics):
        values = [metrics.get(p, 0) for p in percentiles]
        ax.bar(x + i*width, values, width, label=metrics['folder'], alpha=0.7)
    
    ax.set_ylabel('Distance Error (pixels)', fontweight='bold')
    ax.set_title('Distance Error Percentiles Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * (len(all_metrics) - 1) / 2)
    ax.set_xticklabels(['25th', '50th', '75th', '90th', '95th', '99th'])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(base_folder, output_folder, 'percentiles_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nAll plots saved to '{base_folder}/{output_folder}/' folder")

# Main execution
if __name__ == "__main__":
    # Set your base folder path here
    base_folder = "/Volumes/Lab/SLEAP/models/bottom-up_9ptskeleton/archive"  # Current directory; change to your folder path
    
    print("Collecting metrics from all txt files...")
    all_metrics = collect_all_metrics(base_folder)
    
    if not all_metrics:
        print("No metrics files found!")
    else:
        print(f"\nFound {len(all_metrics)} metrics files")
        print("\nGenerating comparison plots...")
        plot_metrics_comparison(all_metrics)
        print("\nDone!")
        
        # Print summary table
        print("\n" + "="*80)
        print("SUMMARY TABLE")
        print("="*80)
        print(f"{'Model':<20} {'F1':<8} {'Precision':<10} {'Recall':<8} {'Mean Err':<10} {'Median Err':<12}")
        print("-"*80)
        for m in all_metrics:
            print(f"{m['folder']:<20} {m.get('f1_score', 0):<8.3f} {m.get('precision', 0):<10.3f} "
                  f"{m.get('recall', 0):<8.3f} {m.get('mean', 0):<10.2f} {m.get('p50_median', 0):<12.2f}")
        print("="*80)