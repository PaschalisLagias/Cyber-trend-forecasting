"""
Dashboard Visualisation Utilities

This module provides helper functions for displaying comparative model 
outputs, evaluation tables, and forecast grids within Jupyter Notebooks.
"""

import math
import os
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


def display_single_image(img_path, title=None, figsize=(14, 4)):
    """
    Renders a single image centered within the notebook output.
    Displays a placeholder box if the image file is missing.

    Args:
        img_path (str): File path to the image asset.
        title (str, optional): Title for the figure.
        figsize (tuple): Width and height of the figure in inches.
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)

    if os.path.exists(img_path):
        img = mpimg.imread(img_path)
        ax.imshow(img)
        ax.axis("off")
    else:
        ax.text(
            0.5,
            0.5,
            "Pending Generation\nFile not found",
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=12,
            color="gray",
            bbox=dict(
                facecolor="whitesmoke",
                edgecolor="lightgray",
                boxstyle="round,pad=1",
            ),
        )
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def display_side_by_side(
    img_path_left, img_path_right, title_left, title_right, main_title=None
):
    """
    Renders two images side by side for direct comparative evaluation.
    Displays a placeholder box if an image file is missing from disk.

    Args:
        img_path_left (str): File path for the left image (B-MTGNN).
        img_path_right (str): File path for the right image (VisionTS++).
        title_left (str): Subplot title for the left image.
        title_right (str): Subplot title for the right image.
        main_title (str, optional): Overall figure title.
    """
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    if main_title:
        fig.suptitle(main_title, fontsize=16, fontweight="bold", y=1.05)

    # Render Left Image (B-MTGNN)
    if os.path.exists(img_path_left):
        img_left = mpimg.imread(img_path_left)
        axes[0].imshow(img_left)
        axes[0].axis("off")
    else:
        axes[0].text(
            0.5,
            0.5,
            "Pending Generation\n(B-MTGNN)",
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=14,
            color="gray",
            bbox=dict(
                facecolor="whitesmoke",
                edgecolor="lightgray",
                boxstyle="round,pad=1",
            ),
        )
        axes[0].axis("off")
    axes[0].set_title(title_left, fontsize=14)

    # Render Right Image (VisionTS++)
    if os.path.exists(img_path_right):
        img_right = mpimg.imread(img_path_right)
        axes[1].imshow(img_right)
        axes[1].axis("off")
    else:
        axes[1].text(
            0.5,
            0.5,
            "Pending Generation\n(VisionTS++)",
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=14,
            color="gray",
            bbox=dict(
                facecolor="whitesmoke",
                edgecolor="lightgray",
                boxstyle="round,pad=1",
            ),
        )
        axes[1].axis("off")
    axes[1].set_title(title_right, fontsize=14)

    plt.tight_layout()
    plt.show()


def display_image_grid(image_paths, titles, cols=3):
    """
    Renders multiple images in a grid format.
    Displays a placeholder box if an image file is missing from disk.

    Args:
        image_paths (list): List of file paths to display.
        titles (list): List of titles corresponding to each image.
        cols (int): Number of grid columns (default is 3).
    """
    rows = math.ceil(len(image_paths) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(18, 5 * rows))
    axes = axes.flatten()

    for i, (path, title) in enumerate(zip(image_paths, titles)):
        if os.path.exists(path):
            img = mpimg.imread(path)
            axes[i].imshow(img)
            axes[i].axis("off")
        else:
            axes[i].text(
                0.5,
                0.5,
                "Pending Generation",
                horizontalalignment="center",
                verticalalignment="center",
                fontsize=14,
                color="gray",
                bbox=dict(
                    facecolor="whitesmoke",
                    edgecolor="lightgray",
                    boxstyle="round,pad=1",
                ),
            )
            axes[i].axis("off")
        axes[i].set_title(title, fontsize=14)

    # Hide unused subplot axes
    for j in range(len(image_paths), len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()