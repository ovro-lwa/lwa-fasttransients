import hiplot as hip
import pandas as pd
import os

class HiPlotVisualizer:
    """
    Class for generating HiPlot visualizations.

    Args:
        output_dir (str): Directory to save the HiPlot files.
    """
    def __init__(self, output_dir):
        self.output_dir = output_dir

    def generate_hiplot(self, csv_file):
        """
        Generate a HiPlot visualization from a CSV file.

        Args:
            csv_file (str): Path to the CSV file.

        Returns:
            str: Path to the generated HiPlot HTML file.
        """
        # Read the CSV file
        df = pd.read_csv(csv_file)

        # Round the numbers to 3 decimal places
        df = df.round(3)

        # Generate HiPlot visualization
        exp = hip.Experiment.from_dataframe(df)

        # Create a unique filename for the output
        base_name = os.path.splitext(os.path.basename(csv_file))[0]
        hiplot_file = os.path.join(self.output_dir, f"{base_name}_hiplot.html")

        # Save the HiPlot visualization as an HTML file
        exp.to_html(hiplot_file)

        return hiplot_file
