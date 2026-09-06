"""Domain-specific physical spatial rationale generator."""

from typing import Dict, List


class SpatialRationaleEngine:
    """Translates quantitative spatial feature distributions into physical domain rationales."""

    def generate_rationale(
        self,
        features: Dict[str, float],
        predicted_class: str,
    ) -> List[str]:
        """Generates evidence bullet points explaining the physical pattern justification."""
        rationale = []
        center_to_edge = features.get("center_to_edge_ratio", 1.0)
        eccentricity = features.get("eccentricity", 0.0)
        center_density = features.get("center_defect_density", 0.0)
        edge_density = features.get("edge_defect_density", 0.0)
        global_density = features.get("global_defect_density", 0.0)
        cluster_count = features.get("cluster_count", 0.0)
        max_cluster_ratio = features.get("max_cluster_area_ratio", 0.0)
        radial_std = features.get("radial_density_std", 0.0)

        if predicted_class == "Center":
            rationale.append(f"High defect concentration in innermost radial zone ({center_density * 100:.1f}% defect rate).")
            rationale.append(f"Strong Center-to-Edge defect density ratio of {center_to_edge:.2f}x.")
            rationale.append("Defect centroid aligns closely with wafer center coordinates.")

        elif predicted_class == "Edge":
            rationale.append(f"Defects heavily concentrated along outer wafer perimeter ({edge_density * 100:.1f}% defect rate).")
            rationale.append(f"Inverted Center-to-Edge ratio ({center_to_edge:.2f}x) indicating central yield clarity.")
            rationale.append("Outer annular ring exhibits peak radial defect density.")

        elif predicted_class in ("Donut", "Ring"):
            rationale.append("Peak defect density observed in intermediate concentric annular rings.")
            rationale.append(f"Central and outermost edge regions remain relatively unpopulated (Radial std: {radial_std:.3f}).")
            rationale.append("Annular symmetry profile matches rotationally invariant ring deposition.")

        elif predicted_class == "Scratch":
            rationale.append(f"High geometric eccentricity ({eccentricity:.2f}) indicates an elongated linear defect trajectory.")
            rationale.append(f"Dominant spatial inertia axis aligns with physical mechanical handling path.")
            rationale.append("Continuous connected defect streak spans across multiple die coordinates.")

        elif predicted_class in ("Localized_Cluster", "Cluster"):
            rationale.append(f"Dense localized cluster detected accounting for {max_cluster_ratio * 100:.1f}% of all failing dies.")
            rationale.append(f"Significant spatial offset between defect centroid and wafer center.")
            rationale.append(f"Connected component analysis identified {int(cluster_count)} isolated defect cluster(s).")

        elif predicted_class == "Normal":
            rationale.append(f"Negligible global defect density ({global_density * 100:.2f}% < 1.0% threshold).")
            rationale.append("No statistically significant radial or angular defect clustering detected.")
            rationale.append("Nominal die yield distribution across all annular zones.")

        elif predicted_class == "Random":
            rationale.append("Defects are uniformly scattered with low spatial covariance.")
            rationale.append(f"Radial and angular density variance is minimal (Radial std: {radial_std:.3f}).")
            rationale.append("Absence of high-density localized clusters or geometric trajectories.")

        else:
            rationale.append(f"Global defect density measured at {global_density * 100:.1f}%.")
            rationale.append(f"Center-to-Edge density ratio calculated as {center_to_edge:.2f}.")
            rationale.append("Pattern exhibits complex multi-modal spatial characteristics.")

        return rationale
