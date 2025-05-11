from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from scipy.optimize import minimize
from scipy.stats.mstats import gmean
import matplotlib.pyplot as plt
import seaborn as sns

class BreadOptimizer:
    """A class to optimize bread baking parameters based on texture, flavor and practicality.

    The optimizer uses desirability functions and weighted geometric means to find optimal
    baking parameters given preferences for different quality aspects.
    """

    def __init__(self) -> None:
        """Initialize the BreadOptimizer with parameter bounds."""
        # Define parameter bounds
        self.bounds: Dict[str, Tuple[float, float]] = {
            'fermentation_time': (1, 24),  # hours
            'ferment_temp': (20, 30),  # Celsius
            'hydration': (60, 85),  # percentage
            'kneading_time': (0, 20),  # minutes
            'baking_temp': (180, 250)  # Celsius
        }

    def desirability_smaller_is_better(self, x: float, x_min: float, x_max: float) -> float:
        """Calculate desirability function value where smaller values are better.

        Args:
            x: Input parameter value
            x_min: Minimum acceptable value
            x_max: Maximum acceptable value

        Returns:
            Desirability score between 0 and 1
        """
        if x <= x_min:
            return 1.0
        elif x >= x_max:
            return 0.0
        else:
            return ((x_max - x) / (x_max - x_min))

    def desirability_larger_is_better(self, x: float, x_min: float, x_max: float) -> float:
        """Calculate desirability function value where larger values are better.

        Args:
            x: Input parameter value
            x_min: Minimum acceptable value
            x_max: Maximum acceptable value

        Returns:
            Desirability score between 0 and 1
        """
        if x <= x_min:
            return 0.0
        elif x >= x_max:
            return 1.0
        else:
            return ((x - x_min) / (x_max - x_min))

    def desirability_two_sided(self, x: float, x_min: float, x_target: float, x_max: float) -> float:
        """Calculate two-sided desirability function value with target value.

        Args:
            x: Input parameter value
            x_min: Minimum acceptable value
            x_target: Target (optimal) value
            x_max: Maximum acceptable value

        Returns:
            Desirability score between 0 and 1
        """
        if x_min <= x <= x_target:
            return (x - x_min) / (x_target - x_min)
        elif x_target < x <= x_max:
            return (x_max - x) / (x_max - x_target)
        else:
            return 0.0

    def compute_texture_quality(self, params: List[float]) -> float:
        """Compute texture quality score based on input parameters.

        Args:
            params: List of parameter values [fermentation_time, ferment_temp, hydration,
                   kneading_time, baking_temp]

        Returns:
            Weighted texture quality score between 0 and 1
        """
        fermentation_d = self.desirability_two_sided(params[0], 2, 12, 24)
        ferment_temp_d = self.desirability_two_sided(params[1], 20, 25, 30)
        hydration_d = self.desirability_two_sided(params[2], 60, 75, 85)
        kneading_d = self.desirability_two_sided(params[3], 5, 12, 20)
        baking_temp_d = self.desirability_two_sided(params[4], 180, 220, 250)

        # Weighted combination
        weights = [0.25, 0.15, 0.2, 0.2, 0.2]  # Adjusted to include baking temp
        return np.average([fermentation_d, ferment_temp_d, hydration_d, kneading_d, baking_temp_d],
                          weights=weights)

    def compute_flavor_profile(self, params: List[float]) -> float:
        """Compute flavor profile score based on input parameters.

        Args:
            params: List of parameter values [fermentation_time, ferment_temp, hydration,
                   kneading_time, baking_temp]

        Returns:
            Weighted flavor profile score between 0 and 1
        """
        # Flavor mainly affected by fermentation parameters
        fermentation_d = self.desirability_larger_is_better(params[0], 4, 18)
        ferment_temp_d = self.desirability_two_sided(params[1], 20, 24, 28)
        hydration_d = self.desirability_two_sided(params[2], 65, 75, 85)

        # Baking temperature has minimal effect on flavor
        weights = [0.5, 0.3, 0.2]
        return np.average([fermentation_d, ferment_temp_d, hydration_d],
                          weights=weights)

    def compute_practicality(self, params: List[float]) -> float:
        """Compute practicality score based on input parameters.

        Args:
            params: List of parameter values [fermentation_time, ferment_temp, hydration,
                   kneading_time, baking_temp]

        Returns:
            Weighted practicality score between 0 and 1
        """
        fermentation_d = self.desirability_smaller_is_better(params[0], 2, 24)
        hydration_d = self.desirability_two_sided(params[2], 65, 70, 80)
        kneading_d = self.desirability_smaller_is_better(params[3], 5, 20)
        baking_temp_d = self.desirability_smaller_is_better(params[4], 180, 250)

        weights = [0.4, 0.2, 0.2, 0.2]
        return np.average([fermentation_d, hydration_d, kneading_d, baking_temp_d],
                          weights=weights)

    def overall_desirability(self, desirabilities: List[float], weights: Optional[List[float]] = None) -> float:
        """Compute overall desirability using geometric mean.

        Args:
            desirabilities: Individual desirability scores
            weights: Weights for each desirability. If None, equal weights are used.

        Returns:
            Overall desirability score between 0 and 1
        """
        if weights is None:
            weights = [1] * len(desirabilities)

        # Convert to numpy arrays
        d = np.array(desirabilities)
        w = np.array(weights)

        # Calculate geometric mean
        return np.prod(d ** w) ** (1 / np.sum(w))

    def objective_function(self, params: List[float], weights: List[float]) -> float:
        """Compute overall desirability score based on individual quality metrics.

        Args:
            params: List of parameter values
            weights: Weights for texture, flavor and practicality scores

        Returns:
            Negative overall desirability score (for minimization)
        """
        # Compute individual desirability scores
        texture = self.compute_texture_quality(params)
        flavor = self.compute_flavor_profile(params)
        practicality = self.compute_practicality(params)

        # Ensure weights sum up to one
        weights = np.array(weights) / np.sum(weights)

        # Calculate overall desirability using geometric mean
        overall_d = self.overall_desirability([texture, flavor, practicality], weights)

        # Return negative value since we want to maximize desirability
        # but optimization functions typically minimize
        return -overall_d

    def optimize(self, preference: Optional[Union[str, Tuple[float, float, float]]] = None) -> Dict:
        """Find optimal parameters based on preference weights.

        Args:
            preference: Preference profile to use:
                - "texture": Emphasize texture quality
                - "flavor": Emphasize flavor profile
                - "practicality": Emphasize practicality
                - "balanced": Equal weights (default)
                - Custom tuple of weights can also be provided

        Returns:
            Dictionary containing:
                - parameters: Optimized parameter values
                - achieved_scores: Achieved quality scores
                - weights: Used preference weights
                - overall_desirability: Final desirability score
                - success: Whether optimization succeeded
        """
        # Define parameter bounds
        bounds = {
            'fermentation_time': (1, 24),
            'fermentation_temp': (20, 30),
            'hydration_level': (60, 85),
            'kneading_time': (0, 20),
            'baking_temp': (180, 250)
        }

        # Set weights based on preference
        if preference == "texture":
            weights = [0.7, 0.2, 0.1]  # High emphasis on texture
        elif preference == "flavor":
            weights = [0.2, 0.7, 0.1]  # High emphasis on flavor
        elif preference == "practicality":
            weights = [0.1, 0.2, 0.7]  # High emphasis on practicality
        elif isinstance(preference, (list, tuple)) and len(preference) == 3:
            weights = preference  # Custom weights provided
        else:
            weights = [0.33, 0.33, 0.34]  # Balanced by default

        # Initial guess (middle of bounds)
        x0 = [(b[0] + b[1]) / 2 for b in bounds.values()]

        # Run optimization
        result = minimize(
            self.objective_function,
            x0,
            args=(weights,),
            bounds=list(bounds.values()),
            method='SLSQP'
        )

        # Return parameters and achieved scores
        params = {name: value for name, value in zip(bounds.keys(), result.x)}
        achieved = {
            'texture': self.compute_texture_quality(result.x),
            'flavor': self.compute_flavor_profile(result.x),
            'practicality': self.compute_practicality(result.x)
        }

        overall_d = self.overall_desirability([achieved['texture'],
                                          achieved['flavor'],
                                          achieved['practicality']],
                                         weights=weights)

        return {
            'parameters': params,
            'achieved_scores': achieved,
            'weights': weights,
            'overall_desirability': overall_d,
            'success': result.success
        }

    def plot_scores_radar(self, results: Dict) -> plt.Figure:
        """Create radar plot of achieved scores.

        Args:
            results: Dictionary containing optimization results

        Returns:
            Matplotlib figure with radar plot
        """
        # Create polar plot
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})

        # Plot achieved scores
        categories = ['Texture', 'Flavor', 'Practicality']
        achieved_scores = [results['achieved_scores'][k.lower()] for k in categories]

        # Use weights from results
        weights = results['weights']
        weighted_scores = [s * w for s, w in zip(achieved_scores, weights)]

        # Compute angle for each axis
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False)

        # Close the plot
        achieved_scores = np.concatenate((achieved_scores, [achieved_scores[0]]))
        weighted_scores = np.concatenate((weighted_scores, [weighted_scores[0]]))
        angles = np.concatenate((angles, [angles[0]]))

        # Plot data
        ax.plot(angles, achieved_scores, 'o-', label='Desirability Scores')
        ax.plot(angles, weighted_scores, 'o--', label='Weighted Scores')
        ax.fill(angles, achieved_scores, alpha=0.25)

        # Set category labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)

        # Add legend and title
        weights_str = ", ".join([f"{w:.2f}" for w in weights])
        plt.legend(loc='upper right')
        plt.title(f'Achieved Scores (Overall: {results["overall_desirability"]:.2f}, Weights: {weights_str})')

        return fig

    def plot_parameters(self, results: Dict) -> plt.Figure:
        """Create bar plot of optimized parameters.

        Args:
            results: Dictionary containing optimization results

        Returns:
            Matplotlib figure with parameter bar plot
        """
        # Create horizontal bar chart
        params = results['parameters']
        y_pos = np.arange(len(params))
        values = list(params.values())
        labels = [' '.join(k.split('_')).title() for k in params.keys()]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.barh(y_pos, values)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels)
        ax.set_xlabel('Value')

        # Add emphasis info to title
        weights = results['weights']
        if weights[0] > weights[1] and weights[0] > weights[2]:
            emphasis = "Texture"
        elif weights[1] > weights[0] and weights[1] > weights[2]:
            emphasis = "Flavor"
        elif weights[2] > weights[0] and weights[2] > weights[1]:
            emphasis = "Practicality"
        else:
            emphasis = "Balanced"

        ax.set_title(f'Optimized Parameters (Emphasis: {emphasis})')

        return fig

# Example usage
optimizer = BreadOptimizer()
result = optimizer.optimize(preference='flavor')

# Create visualization
fig = optimizer.plot_scores_radar(result)
fig = optimizer.plot_parameters(result)
plt.show()
