"""
2D Plotting functions for propeller parameter sweep
Generates heatmaps and contour plots for amplitude × position design space
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns


def make_plots_2d(df, amplitudes, positions, output_dir,
                  diameter, rpm, vinf, amplitude_frac, 
                  r_root_frac, r_tip_frac, n_steps, radius):
    """
    Generate 2D heatmaps and contour plots for amplitude × position sweep.
    
    Parameters
    ----------
    df : pd.DataFrame
        Summary dataframe with columns: amplitude_idx, position_idx, 
        CT_H, CQ_H, Thrust_total, FOM_total, etc.
    amplitudes : np.ndarray
        Amplitude fractions w.r.t radius
    positions : np.ndarray
        Position values (radial fraction)
    output_dir : str
        Output directory path
    diameter : float
        Propeller diameter (m)
    rpm : float
        RPM
    vinf : float
        Free-stream velocity (m/s)
    amplitude_frac : float
        Max amplitude as fraction of R
    r_root_frac : float
        Root cutout fraction
    r_tip_frac : float
        Tip fraction
    n_steps : int
        Number of steps per dimension
    radius : float
        Propeller radius (m)
    """
    
    # ── Reshape data into 2D grids ─────────────────────────────────────────
    ct_grid = np.full((n_steps, n_steps), np.nan)
    cq_grid = np.full((n_steps, n_steps), np.nan)
    thrust_grid = np.full((n_steps, n_steps), np.nan)
    fom_grid = np.full((n_steps, n_steps), np.nan)
    
    for _, row in df.iterrows():
        a_idx = int(row['amplitude_idx'])
        p_idx = int(row['position_idx'])
        
        ct_grid[p_idx, a_idx] = row.get('CT_H')
        cq_grid[p_idx, a_idx] = row.get('CQ_H')
        thrust_grid[p_idx, a_idx] = row.get('Thrust_total')
        fom_grid[p_idx, a_idx] = row.get('FOM_total')
    
    amp_mm = amplitudes * 1000  # Convert to mm for labels
    
    # ── Figure 1: 2×2 Heatmap Grid (CT, CQ, Thrust, FOM) ─────────────────
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    metrics = [
        ('CT_H', ct_grid, 'Thrust Coefficient', 'viridis'),
        ('CQ_H', cq_grid, 'Torque Coefficient', 'viridis'),
        ('Thrust', thrust_grid, 'Thrust (N)', 'plasma'),
        ('FOM', fom_grid, 'Figure of Merit', 'RdYlGn'),
    ]
    
    for idx, (key, grid, title, cmap) in enumerate(metrics):
        ax = fig.add_subplot(gs[idx // 2, idx % 2])
        
        im = ax.imshow(grid, aspect='auto', cmap=cmap, origin='lower',
                       extent=[amp_mm[0], amp_mm[-1], positions[0], positions[-1]])
        
        ax.set_xlabel('Amplitude (t/R)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Position (r/R)', fontsize=11, fontweight='bold')
        ax.set_title(f'{title} vs Amplitude & Position', fontsize=12, fontweight='bold')
        
        cbar = plt.colorbar(im, ax=ax, label=key)
        cbar.formatter.set_powerlimits((0, 0))
        
        # Add grid for clarity
        ax.grid(True, alpha=0.2, linestyle='--')
    
    fig.suptitle(
        f'2D Parameter Sweep: {rpm:.0f} RPM, V∞={vinf} m/s, D={diameter*1000:.0f}mm',
        fontsize=14, fontweight='bold', y=0.995
    )
    
    heatmap_path = f'{output_dir}/heatmaps_2d.png'
    plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
    print(f"✓ Heatmap grid → {heatmap_path}")
    plt.close()
    
    # ── Figure 2: Contour plots ────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.subplots_adjust(hspace=0.3, wspace=0.3)
    
    # Create meshgrid for contours
    A_mesh, P_mesh = np.meshgrid(amp_mm, positions)
    
    contour_data = [
        (ct_grid, 'CT_H', 'Thrust Coefficient', axes[0, 0]),
        (cq_grid, 'CQ_H', 'Torque Coefficient', axes[0, 1]),
        (thrust_grid, 'Thrust', 'Thrust (N)', axes[1, 0]),
        (fom_grid, 'FOM', 'Figure of Merit', axes[1, 1]),
    ]
    
    for grid, key, title, ax in contour_data:
        # Contour lines
        cs = ax.contour(A_mesh, P_mesh, grid, levels=10, colors='black', 
                        linewidths=0.5, alpha=0.4)
        ax.clabel(cs, inline=True, fontsize=8, fmt='%.2f')
        
        # Filled contours
        cf = ax.contourf(A_mesh, P_mesh, grid, levels=15, cmap='RdYlGn_r')
        cbar = plt.colorbar(cf, ax=ax, label=key)
        cbar.formatter.set_powerlimits((0, 0))
        
        ax.set_xlabel('Amplitude (mm)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Position (r/R)', fontsize=11, fontweight='bold')
        ax.set_title(f'{title}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.2, linestyle='--')
    
    fig.suptitle(
        f'Contour Plot: {rpm:.0f} RPM, V∞={vinf} m/s, D={diameter*1000:.0f}mm',
        fontsize=14, fontweight='bold'
    )
    
    contour_path = f'{output_dir}/contours_2d.png'
    plt.savefig(contour_path, dpi=150, bbox_inches='tight')
    print(f"✓ Contour plot → {contour_path}")
    plt.close()
    
    # ── Figure 3: Slices through parameter space ───────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.subplots_adjust(hspace=0.35, wspace=0.3)
    
    # Slice 1: Fixed position, vary amplitude (FOM)
    mid_pos_idx = n_steps // 2
    fom_vs_amp = [fom_grid[mid_pos_idx, i] for i in range(n_steps)]
    axes[0, 0].plot(amp_mm, fom_vs_amp, 'o-', linewidth=2, markersize=6, color='#2E86AB')
    axes[0, 0].set_xlabel('Amplitude [-]', fontsize=11, fontweight='bold')
    axes[0, 0].set_ylabel('FOM', fontsize=11, fontweight='bold')
    axes[0, 0].set_title(f'FOM vs Amplitude (pos={positions[mid_pos_idx]:.3f} r/R)',
                         fontsize=11, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Slice 2: Fixed amplitude, vary position (FOM)
    mid_amp_idx = n_steps // 2
    fom_vs_pos = [fom_grid[i, mid_amp_idx] for i in range(n_steps)]
    axes[0, 1].plot(positions, fom_vs_pos, 'o-', linewidth=2, markersize=6, color='#A23B72')
    axes[0, 1].set_xlabel('Position (r/R)', fontsize=11, fontweight='bold')
    axes[0, 1].set_ylabel('FOM', fontsize=11, fontweight='bold')
    axes[0, 1].set_title(f'FOM vs Position (amp={amp_mm[mid_amp_idx]:.1f}mm)',
                         fontsize=11, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Slice 3: CT vs CQ
    ct_flat = ct_grid.flatten()
    cq_flat = cq_grid.flatten()
    valid_mask = ~(np.isnan(ct_flat) | np.isnan(cq_flat))
    axes[1, 0].scatter(ct_flat[valid_mask], cq_flat[valid_mask], 
                       c=fom_grid.flatten()[valid_mask], cmap='RdYlGn', 
                       s=50, alpha=0.6, edgecolors='black', linewidth=0.5)
    axes[1, 0].set_xlabel('CT_H', fontsize=11, fontweight='bold')
    axes[1, 0].set_ylabel('CQ_H', fontsize=11, fontweight='bold')
    axes[1, 0].set_title('CT vs CQ (colored by FOM)', fontsize=11, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Slice 4: Thrust vs FOM
    thrust_flat = thrust_grid.flatten()
    valid_mask2 = ~(np.isnan(thrust_flat) | np.isnan(fom_flat := fom_grid.flatten()))
    axes[1, 1].scatter(thrust_flat[valid_mask2], fom_flat[valid_mask2],
                       c=ct_flat[valid_mask2], cmap='viridis',
                       s=50, alpha=0.6, edgecolors='black', linewidth=0.5)
    axes[1, 1].set_xlabel('Thrust (N)', fontsize=11, fontweight='bold')
    axes[1, 1].set_ylabel('FOM', fontsize=11, fontweight='bold')
    axes[1, 1].set_title('Thrust vs FOM (colored by CT)', fontsize=11, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    
    fig.suptitle(
        f'1D Slices & Cross-Plots: {rpm:.0f} RPM, V∞={vinf} m/s',
        fontsize=14, fontweight='bold'
    )
    
    slice_path = f'{output_dir}/slices_2d.png'
    plt.savefig(slice_path, dpi=150, bbox_inches='tight')
    print(f"✓ Slice plots → {slice_path}")
    plt.close()
    
    # ── Figure 4: Optimization landscape (surface plot) ────────────────────
    try:
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure(figsize=(16, 6))
        
        # FOM surface
        ax1 = fig.add_subplot(121, projection='3d')
        surf1 = ax1.plot_surface(A_mesh, P_mesh, fom_grid, cmap='RdYlGn',
                                 alpha=0.8, edgecolor='none')
        ax1.set_xlabel('Amplitude [-]', fontsize=10, fontweight='bold')
        ax1.set_ylabel('Position (r/R)', fontsize=10, fontweight='bold')
        ax1.set_zlabel('FOM', fontsize=10, fontweight='bold')
        ax1.set_title('Figure of Merit Surface', fontsize=11, fontweight='bold')
        fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=5)
        
        # Thrust surface
        ax2 = fig.add_subplot(122, projection='3d')
        surf2 = ax2.plot_surface(A_mesh, P_mesh, thrust_grid, cmap='plasma',
                                 alpha=0.8, edgecolor='none')
        ax2.set_xlabel('Amplitude [-]', fontsize=10, fontweight='bold')
        ax2.set_ylabel('Position (r/R)', fontsize=10, fontweight='bold')
        ax2.set_zlabel('Thrust (N)', fontsize=10, fontweight='bold')
        ax2.set_title('Thrust Surface', fontsize=11, fontweight='bold')
        fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=5)
        
        fig.suptitle(
            f'3D Optimization Landscape: {rpm:.0f} RPM, V∞={vinf} m/s',
            fontsize=14, fontweight='bold'
        )
        
        surface_path = f'{output_dir}/surfaces_3d.png'
        plt.savefig(surface_path, dpi=150, bbox_inches='tight')
        print(f"✓ 3D surface plot → {surface_path}")
        plt.close()
    except Exception as e:
        print(f"⚠ 3D plot skipped: {e}")
    
    # ── Summary statistics ─────────────────────────────────────────────────
    print("\n" + "="*70)
    print("2D SWEEP SUMMARY STATISTICS")
    print("="*70)
    
    valid_fom = fom_grid[~np.isnan(fom_grid)]
    valid_ct = ct_grid[~np.isnan(ct_grid)]
    valid_cq = cq_grid[~np.isnan(cq_grid)]
    valid_thrust = thrust_grid[~np.isnan(thrust_grid)]
    
    print(f"FOM:    min={np.nanmin(fom_grid):.4f}  max={np.nanmax(fom_grid):.4f}  "
          f"mean={np.nanmean(fom_grid):.4f}  std={np.nanstd(fom_grid):.4f}")
    print(f"CT:     min={np.nanmin(ct_grid):.5f}  max={np.nanmax(ct_grid):.5f}  "
          f"mean={np.nanmean(ct_grid):.5f}  std={np.nanstd(ct_grid):.5f}")
    print(f"CQ:     min={np.nanmin(cq_grid):.5f}  max={np.nanmax(cq_grid):.5f}  "
          f"mean={np.nanmean(cq_grid):.5f}  std={np.nanstd(cq_grid):.5f}")
    print(f"Thrust: min={np.nanmin(thrust_grid):.3f}  max={np.nanmax(thrust_grid):.3f}  "
          f"mean={np.nanmean(thrust_grid):.3f}  std={np.nanstd(thrust_grid):.3f}")
    
    # Find optima
    fom_max_idx = np.unravel_index(np.nanargmax(fom_grid), fom_grid.shape)
    ct_max_idx = np.unravel_index(np.nanargmax(ct_grid), ct_grid.shape)
    thrust_max_idx = np.unravel_index(np.nanargmax(thrust_grid), thrust_grid.shape)
    
    print(f"\nFOM optimum:    amp={amp_mm[fom_max_idx[1]]:.1f}mm  "
          f"pos={positions[fom_max_idx[0]]:.4f}  FOM={fom_grid[fom_max_idx]:.4f}")
    print(f"CT optimum:     amp={amp_mm[ct_max_idx[1]]:.1f}mm  "
          f"pos={positions[ct_max_idx[0]]:.4f}  CT={ct_grid[ct_max_idx]:.5f}")
    print(f"Thrust optimum: amp={amp_mm[thrust_max_idx[1]]:.1f}mm  "
          f"pos={positions[thrust_max_idx[0]]:.4f}  Thrust={thrust_grid[thrust_max_idx]:.3f}N")
    print("="*70 + "\n")
