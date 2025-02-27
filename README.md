IMA: A Tool for Automated Sarcomere Measurement
IMA is a Python-based software designed to streamline and standardize the measurement of sarcomere length and width from microscopic images. It offers automated, semi-manual, and manual modes to accommodate diverse image conditions and user preferences.

Key Features:

   Automated Analysis: Automatically identifies Z-discs, generates splines, extracts histograms, and fits Gaussian curves to calculate sarcomere length.
   Manual Override: Allows for user intervention to correct point order or manually place points for challenging images.
   Width Calculation: Provides both automated and manual options for determining sarcomere width.
   Multi-format Support: Handles TIFF, LIF, and CZI image formats.
   Visual Output: Generates informative images and histograms to visualize the analysis process.
   Data Export: Compiles results into an Excel spreadsheet for convenient data management.
   
   Installation:
   IMA requires Python 3.10 or newer. Using an IDE like PyCharm is recommended. Detailed installation instructions are provided within the repository.

Usage:
   Import images.
   Select channels for length and width calculation.
   Choose analysis mode (automatic, semi-manual, or manual).
   Run the evaluation.
   Review the generated images and data output.

Detailed information and instructions can be found in the documentation within the repository.
