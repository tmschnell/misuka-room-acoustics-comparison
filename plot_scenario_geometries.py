"""Plot top-down and side-on outlines of two comparison-suite scenes, with
source/receiver marked, as a quick visual sanity check of the scene
geometry defined in test_pyroomacoustics_comparison.py.

Run directly: python plot_scenario_geometries.py
"""

import os

import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pra_comparison_results")

# Geometry mirrors scenario_shoebox_diffuse() / scenario_l_room() in
# test_pyroomacoustics_comparison.py.
SCENARIOS = [
    dict(
        name="shoebox_diffuse_200m3",
        # axis-aligned box [0,8] x [0,5] x [0,5]
        top_outline=[(0, 0), (8.0, 0), (8.0, 5.0), (0, 5.0)],
        side_outline=[(0, 0), (8.0, 0), (8.0, 5.0), (0, 5.0)],
        src=(2.0, 2.5, 1.2),
        mic=(5.0, 2.5, 1.2),
    ),
    dict(
        name="l_room_1000m3",
        # corners per misuka_lroom_shapes() docstring, a=5,b=10,c=10,d=5,height=10
        # footprint = a*b + c*d = 50 + 50 = 100 m^2, *height = 1000 m^3
        top_outline=[(0, 0), (15.0, 0), (15.0, 5.0), (5.0, 5.0), (5.0, 10.0), (0, 10.0)],
        # side view = silhouette when viewed along Y, i.e. the full a+c x height
        # bounding rectangle (the notch only shows up in the top view)
        side_outline=[(0, 0), (15.0, 0), (15.0, 10.0), (0, 10.0)],
        grayline=[(5.0, 5.0), (10.0, 0.0)],  # vertical line at the notch
        src=(4.0, 8.0, 1.2),
        mic=(14.0, 4.0, 1.2),
    ),
]


def plot_view(ax, outline, src_xy, mic_xy, xlabel, ylabel, grayline=None):
    xs, ys = zip(*(outline + [outline[0]]))
    ax.plot(xs, ys, "k-", linewidth=1.5)
    if grayline is not None:
        ax.plot(*grayline, color="gray", linestyle="--", linewidth=1.5, zorder=-3)
    ax.plot(*src_xy, "ro", markersize=8, label="source")
    ax.plot(*mic_xy, "bo", markersize=8, label="receiver")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_aspect("equal")
    ax.grid(True, linewidth=0.5, alpha=0.5)
    ax.legend(loc="upper right", fontsize=8)


if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)
    for scenario in SCENARIOS:
        src, mic = scenario["src"], scenario["mic"]
        fig, (ax_top, ax_side) = plt.subplots(1, 2, figsize=(10, 5))
        fig.suptitle(scenario["name"])
        ax_top.set_title("Top view (X-Y)")
        ax_side.set_title("Side view (X-Z)")
        plot_view(ax_top, scenario["top_outline"], (src[0], src[1]), (mic[0], mic[1]),
                  "x [m]", "y [m]")
        plot_view(ax_side, scenario["side_outline"], (src[0], src[2]), (mic[0], mic[2]),
                  "x [m]", "z [m]", grayline=scenario.get("grayline", None))
        fig.tight_layout()
        out_svg = os.path.join(RESULTS_DIR, f"{scenario['name']}_geometry.svg")
        fig.savefig(out_svg, format="svg")
        print(f"-> saved {out_svg}")
    plt.show()
