"""
rotor_gradient_extraction.py

Complete example: Extract radial distributed force gradients from VSPAERO rotor adjoint solve.

Usage:
    python rotor_gradient_extraction.py path/to/nacapropeller-mod

This script:
1. Loads a rotor/propeller model
2. Runs steady (pseudo-steady) VLM adjoint solve
3. Extracts per-node adjoint sensitivities (dCT/dX, dCP/dX, etc.)
4. Maps nodes to radial blade sections
5. Aggregates to get radial distribution of force gradients
6. Saves results for use in OpenMDAO optimization
"""

import numpy as np
import sys
from pathlib import Path
from typing import List, Tuple

from prep_vspaero import set_prop_blades_mode, vspaero_compute_geometry, prep_sweep_analysis

try:
    import openvsp as vsp
except ImportError:
    print("Warning: openvsp not available, some features disabled")
    vsp = None

from vspaero.optimizer import pyVSPOptimizer
from vspaero import functions


class RotorGradientExtractor:
    """
    Extract and post-process rotor adjoint gradients.
    
    Attributes
    ----------
    opt : pyVSPOptimizer
        The optimizer instance.
    n_sections : int
        Number of blade radial sections to extract.
    n_nodes : int
        Total number of mesh nodes.
    nodes : np.ndarray, shape (n_nodes, 3)
        All mesh node coordinates.
    node_sections : List[List[int]]
        node_sections[i] = list of node indices in section i.
    """
    
    def __init__(self, n_sections: int = 10):
        self.opt = None
        self.n_sections = n_sections
        self.n_nodes = 0
        self.nodes = None
        self.node_sections = None
        self._solve_results = {}
        
    def setup(self, vsp_prefix: str, mode, omega, R, rho, vref, mref, sref, bref, cref, Reref, nwakenodes, ncpu, wakeiter, revs, time_step):
        """
        Load VSP model and prepare optimizer.
        
        Parameters
        ----------
        vsp_prefix : str
            Path to .vsp3 file without extension.
        """
        print(f"Loading VSP model: {vsp_prefix}.vsp3")
        if vsp is not None:
            vsp.ClearVSPModel() # clearing if a model was already loaded
            vsp.ReadVSPFile(f"{vsp_prefix}.vsp3")
            set_prop_blades_mode(mode=mode)
            vspaero_compute_geometry()
            prep_sweep_analysis(omega, R, mode, rho, vref, mref, sref, bref, cref, Reref, nwakenodes,
                                ncpu, wakeiter=wakeiter, revs=revs, time_step=time_step)
        
        # Initialize optimizer
        self.opt = pyVSPOptimizer()
        self.opt.set_unsteady_analysis(0)  # TODO 0 steady 1 unsteady what number pseudo steady??
        
        # Register functions to optimize
        self.opt.set_optimization_functions([
            functions.ROTOR_CT_TOTAL,
            functions.ROTOR_CP_TOTAL,
        ])

        print(f"Setting up optimizer with prefix: {vsp_prefix}")
        self.opt.setup(vsp_prefix)
        
        # Extract node coordinates
        self._extract_node_coordinates()
        
        # Map nodes to blade sections
        self._group_nodes_by_section()
        
    def solve(self, calculate_gradients: bool = True):
        """
        Run forward and adjoint solves.
        
        Parameters
        ----------
        calculate_gradients : bool
            If True, compute adjoint sensitivities.
        """
        print("Running forward + adjoint solve...")
        self.opt.solve(calculate_gradients=calculate_gradients)
        print("Solve complete!")
        
    def _extract_node_coordinates(self):
        """Get coordinates of all mesh nodes."""
        self.n_nodes = self.opt.get_number_of_nodes()
        self.nodes = np.zeros((self.n_nodes, 3), dtype=np.float64)
        
        for i in range(self.n_nodes):
            self.nodes[i, 0] = self.opt.node_x(i)
            self.nodes[i, 1] = self.opt.node_y(i)
            self.nodes[i, 2] = self.opt.node_z(i)
        
        print(f"Extracted {self.n_nodes} mesh nodes")
        
    def _group_nodes_by_section(self):
        """
        Map mesh nodes to blade radial sections.
        
        Strategy: Sort by spanwise (y) coordinate and group into n_sections bins.
        For a rotor with blades aligned along Y-axis, this groups nodes by radial location.
        """
        if self.nodes is None:
            raise RuntimeError("Call _extract_node_coordinates() first")
        
        # Sort nodes by y-coordinate (spanwise)
        y_coords = self.nodes[:, 1]
        y_min, y_max = y_coords.min(), y_coords.max()
        y_span = y_max - y_min
        
        # Create bins for radial sections
        section_edges = np.linspace(y_min, y_max, self.n_sections + 1)
        self.node_sections = [[] for _ in range(self.n_sections)]
        
        # Assign each node to a section
        for node_idx, y in enumerate(y_coords):
            # Find which section bin this node belongs to
            section_idx = min(int((y - y_min) / (y_span + 1e-10) * self.n_sections), 
                             self.n_sections - 1)
            self.node_sections[section_idx].append(node_idx)
        
        # Report section occupancy
        print(f"\nNode grouping into {self.n_sections} spanwise sections:")
        for i, nodes_in_section in enumerate(self.node_sections):
            if nodes_in_section:
                y_vals = y_coords[nodes_in_section]
                print(f"  Section {i:2d}: {len(nodes_in_section):4d} nodes, "
                      f"y ∈ [{y_vals.min():.4f}, {y_vals.max():.4f}]")
            else:
                print(f"  Section {i:2d}: EMPTY")
    
    def get_radial_gradients(self, func_idx: int = 0, component: str = 'magnitude') -> np.ndarray:
        """
        Extract radial distribution of force gradients for a given function.
        
        Parameters
        ----------
        func_idx : int
            Index into the functions list (0=CT, 1=CP if that's your setup).
        component : str
            'x', 'y', 'z', or 'magnitude' (default: sqrt(dx^2 + dy^2 + dz^2))
        
        Returns
        -------
        np.ndarray, shape (n_sections,)
            dF/dSection_i aggregated from all nodes in that section.
        """
        radial_grad = np.zeros(self.n_sections, dtype=np.float64)
        
        for section_idx, node_list in enumerate(self.node_sections):
            if not node_list:
                continue
            
            if component == 'x':
                grad_values = np.array([self.opt.gradient_x(func_idx, i) for i in node_list])
            elif component == 'y':
                grad_values = np.array([self.opt.gradient_y(func_idx, i) for i in node_list])
            elif component == 'z':
                grad_values = np.array([self.opt.gradient_z(func_idx, i) for i in node_list])
            elif component == 'magnitude':
                gx = np.array([self.opt.gradient_x(func_idx, i) for i in node_list])
                gy = np.array([self.opt.gradient_y(func_idx, i) for i in node_list])
                gz = np.array([self.opt.gradient_z(func_idx, i) for i in node_list])
                grad_values = np.sqrt(gx**2 + gy**2 + gz**2)
            else:
                raise ValueError(f"Unknown component: {component}")
            
            # Aggregate: sum of gradients in this section
            radial_grad[section_idx] = np.sum(grad_values)
        
        return radial_grad
    
    def get_radial_forces(self, func_idx: int = 0) -> float:
        """
        Get the total function value (e.g., CT or CP).
        
        Parameters
        ----------
        func_idx : int
            Index into functions list.
        
        Returns
        -------
        float
            The function value.
        """
        out = np.zeros(1, dtype=np.float64)
        self.opt.get_function_value(func_idx, out)
        return out[0]
    
    def print_summary(self):
        """Print a summary of the solve and gradients."""
        print("\n" + "="*70)
        print("ROTOR ADJOINT SOLVE SUMMARY")
        print("="*70)
        
        # Function values
        ct = self.get_radial_forces(0)
        cp = self.get_radial_forces(1)
        
        print(f"\nFunction Values:")
        print(f"  CT (thrust coefficient) = {ct:.6f}")
        print(f"  CP (power coefficient)  = {cp:.6f}")
        if cp > 0:
            print(f"  Figure of Merit = {ct**1.5 / (2*np.sqrt(2)*cp):.4f}")
        
        # Radial gradients
        print(f"\nRadial Gradient Distribution (dCT/dSection):")
        dct = self.get_radial_gradients(0, 'magnitude')
        dcp = self.get_radial_gradients(1, 'magnitude')
        
        for i in range(self.n_sections):
            y_vals = self.nodes[self.node_sections[i], 1] if self.node_sections[i] else []
            y_mid = np.mean(y_vals) if y_vals.size > 0 else 0
            print(f"  Section {i:2d} (y={y_mid:7.4f}): dCT={dct[i]:+.4e}, dCP={dcp[i]:+.4e}")
        
        # Design parameter gradients
        print(f"\nDesign Parameter Gradients (dF/dParam):")
        param_names = self.opt.get_param_names()
        param_grads = self.opt.get_dF_dParam_array()  # shape (n_params, 6)
        
        for i, name in enumerate(param_names[:5]):  # Show first 5
            grad_ct = param_grads[i, 0]
            grad_cp = param_grads[i, 1]
            print(f"  {name:30s}: dCT={grad_ct:+.4e}, dCP={grad_cp:+.4e}")
        if len(param_names) > 5:
            print(f"  ... and {len(param_names) - 5} more parameters")
        
        print("="*70 + "\n")
    
    def export_radial_data(self, filename: str = 'rotor_gradients.txt'):
        """
        Save radial gradient distribution to file.
        
        Parameters
        ----------
        filename : str
            Output filename.
        """
        dct = self.get_radial_gradients(0, 'magnitude')
        dcp = self.get_radial_gradients(1, 'magnitude')
        
        # Compute section midpoints
        y_mids = np.zeros(self.n_sections)
        for i, node_list in enumerate(self.node_sections):
            if node_list:
                y_mids[i] = np.mean(self.nodes[node_list, 1])
        
        # Write to file
        with open(filename, 'w') as f:
            f.write("# Rotor Radial Gradient Distribution\n")
            f.write("# Section   Y_mid    dCT/dSection    dCP/dSection\n")
            for i in range(self.n_sections):
                f.write(f"{i:3d}  {y_mids[i]:10.6f}  {dct[i]:+14.6e}  {dcp[i]:+14.6e}\n")
        
        print(f"Saved radial data to {filename}")


def main():
    """
    Example usage.
    """
    if len(sys.argv) < 2:
        vsp_prefix = "nacapropeller-mod"  # Default
    else:
        vsp_prefix = sys.argv[1]
    
    # Create extractor
    extractor = RotorGradientExtractor(n_sections=10)
    
    # Setup and solve
    extractor.setup(vsp_prefix, mode=)
    extractor.solve(calculate_gradients=True)
    
    # Print summary
    extractor.print_summary()
    
    # Export radial data
    extractor.export_radial_data('rotor_gradients.txt')
    
    print("\nExample: Accessing gradients programmatically")
    print("-" * 70)
    dct = extractor.get_radial_gradients(func_idx=0, component='magnitude')
    print(f"dCT by section shape: {dct.shape}")
    print(f"dCT values: {dct}")


if __name__ == '__main__':
    main()
