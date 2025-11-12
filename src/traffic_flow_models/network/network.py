from typing import List, Optional
import matplotlib.pyplot as plt
from matplotlib import patches

from .cell import Cell
from .onramp import Onramp
from .offramp import Offramp


class Network:
    """A simple ordered mainline network container.

    The Network class stores an ordered list of mainline `Cell` instances
    arranged from upstream (index 0) to downstream (last index). It
    provides convenience methods to add cells and attach or detach on-/off-
    ramps. The Network does not perform simulation — it only manages the
    topology and basic validation when celling objects together.

    Attributes:
        cells: Ordered list of mainline `Cell` objects (upstream ->
            downstream).
    """

    def __init__(self) -> None:
        """Initialize an empty Network.

        The created network contains an empty `cells` list. Cells can be
        added with `add_cell` which takes physical parameters and optionally
        attaches existing ramp objects.
        """
        # ordered list of mainline cells (upstream -> downstream)
        self.cells: List[Cell] = []

    def add_cell(
        self,
        length: float,
        lanes: int,
        lane_capacity: float,
        free_flow_speed: float,
        jam_density: float,
        onramp: Optional[Onramp] = None,
        offramp: Optional[Offramp] = None,
    ) -> Cell:
        """Create a new mainline cell and append it to the network.

        This method constructs a `Cell` instance using the provided
        physical parameters and appends it to the end of the network. If a
        previous cell exists it will set the upstream/downstream references
        so the two cells are connected. Optional `Onramp`/`Offramp`
        instances can be attached directly; their types are validated.

        Args:
            length: Cell length in kilometers.
            lanes: Number of lanes on the cell.
            lane_capacity: Capacity per lane in vehicles per hour.
            free_flow_speed: Free-flow speed in km/h.
            jam_density: Jam density in vehicles per km per lane.
            onramp: Optional existing `Onramp` instance to attach.
            offramp: Optional existing `Offramp` instance to attach.

        Returns:
            The newly created `Cell` instance.

        Raises:
            TypeError: If provided `onramp`/`offramp` are not of the expected
                types.
        """

        new_cell = Cell(
            length=length,
            lanes=lanes,
            lane_capacity=lane_capacity,
            free_flow_speed=free_flow_speed,
            jam_density=jam_density,
        )

        # chain from previous downstream reference: set downstream and upstream
        # pointers so both sides of the connection are known.
        if len(self.cells) > 0:
            prev = self.cells[-1]
            prev.downstream_cell = new_cell
            new_cell.upstream_cell = prev
        self.cells.append(new_cell)

        # attach provided ramp objects directly (do not attempt to construct
        # ramps from dictionaries). Validate types for helpful errors.
        if onramp is not None:
            if not isinstance(onramp, Onramp):
                raise TypeError("onramp must be an Onramp instance")
            new_cell.onramp = onramp

        if offramp is not None:
            if not isinstance(offramp, Offramp):
                raise TypeError("offramp must be an Offramp instance")
            new_cell.offramp = offramp

        return new_cell

    def add_onramp(
        self,
        cell_index: int,
        lanes: int,
        lane_capacity: float,
        free_flow_speed: float,
        jam_density: float,
    ) -> Onramp:
        """Attach a new `Onramp` to a cell by index.

        Args:
            cell_index: Index of the cell in `self.cells` to attach the ramp
                to.
            lanes: Number of lanes on the onramp.
            lane_capacity: Capacity per lane in vehicles per hour.
            free_flow_speed: Free-flow speed in km/h for the onramp.
            jam_density: Jam density in vehicles per km per lane for the
                onramp.

        Returns:
            The created `Onramp` instance.

        Raises:
            ValueError: If the target cell already has an onramp attached.
        """

        cell = self.cells[cell_index]
        if cell.onramp is not None:
            raise ValueError("Cell already has an onramp attached")

        ramp = Onramp(
            lanes=lanes,
            lane_capacity=lane_capacity,
            free_flow_speed=free_flow_speed,
            jam_density=jam_density,
        )
        cell.onramp = ramp
        return ramp

    def add_offramp(
        self,
        cell_index: int,
        lanes: int,
        lane_capacity: float,
        free_flow_speed: float,
        jam_density: float,
        split_ratio: float,
    ) -> Offramp:
        """Attach a new `Offramp` to a cell by index.

        Args:
            cell_index: Index of the cell in `self.cells` to attach the ramp
                to.
            lanes: Number of lanes on the offramp.
            lane_capacity: Capacity per lane in vehicles per hour.
            free_flow_speed: Free-flow speed in km/h for the offramp.
            jam_density: Jam density in vehicles per km per lane for the
                offramp.
            split_ratio: Portion of mainline flow exiting onto the offramp.

        Returns:
            The created `Offramp` instance.

        Raises:
            ValueError: If the target cell already has an offramp attached.
        """

        cell = self.cells[cell_index]
        if cell.offramp is not None:
            raise ValueError("Cell already has an offramp attached")
        ramp = Offramp(
            lanes=lanes,
            lane_capacity=lane_capacity,
            free_flow_speed=free_flow_speed,
            jam_density=jam_density,
            split_ratio=split_ratio,
        )
        cell.offramp = ramp
        return ramp

    def get_onramp(self, cell_index: int) -> Optional[Onramp]:
        """Return the `Onramp` attached to the cell at `cell_index`.

        Returns None if no onramp is attached.
        """

        return self.cells[cell_index].onramp

    def get_offramp(self, cell_index: int) -> Optional[Offramp]:
        """Return the `Offramp` attached to the cell at `cell_index`.

        Returns None if no offramp is attached.
        """

        return self.cells[cell_index].offramp

    def remove_onramp(self, cell_index: int) -> None:
        """Detach and remove the onramp from the cell at `cell_index`.

        After calling this the cell's `onramp` attribute will be set to
        `None`.
        """

        cell = self.cells[cell_index]
        cell.onramp = None

    def remove_offramp(self, cell_index: int) -> None:
        """Detach and remove the offramp from the cell at `cell_index`.

        After calling this the cell's `offramp` attribute will be set to
        `None`.
        """

        cell = self.cells[cell_index]
        cell.offramp = None

    def plot(self, show: bool = True, save_path: Optional[str] = None):
        """Plot the network using Matplotlib primitives.

        Args:
            show: Whether to call matplotlib.pyplot.show() after drawing.
            save_path: Optional path to save the produced figure.

        Returns:
            The Matplotlib Axes instance used for drawing.
        """

        # figure/axes setup
        _, ax = plt.subplots(figsize=(10, 3))

        # basic layout parameters
        total_length = sum(
            max(0.0, float(getattr(l, "length", 0.0))) for l in self.cells
        )

        # fall back to simple spacing when lengths are zero
        spacing = max(total_length * 0.02, 0.05) if total_length > 0 else 0.2
        lane_h = 0.25  # height per lane in plot units
        y_center = 0.0
        x = 0.0
        drawn_right = x

        # draw each cell as a rectangle whose width equals its length
        for i, cell in enumerate(self.cells):
            width = cell.length
            height = cell.lanes * lane_h
            lower = y_center - height / 2

            # rectangle for the mainline cell
            rect = patches.Rectangle(
                (x, lower),
                width,
                height,
                linewidth=1,
                edgecolor="black",
                facecolor="lightgrey",
            )
            ax.add_patch(rect)

            # draw lane separators (visual cue for multiple lanes)
            for ln in range(1, cell.lanes):
                sep_y = lower + ln * lane_h
                ax.plot(
                    [x, x + width],
                    [sep_y, sep_y],
                    color="white",
                    linewidth=0.8,
                    zorder=3,
                )

            # cell label
            ax.text(
                x + width / 2,
                lower + height / 2,
                f"Cell {i+1} [{cell.lanes} lane(s)]",
                ha="center",
                va="center",
                fontsize=8,
                zorder=4,
            )

            # draw downstream connector only when downstream_cell points to the next cell
            next_idx = i + 1
            if (
                next_idx < len(self.cells)
                and cell.downstream_cell is self.cells[next_idx]
            ):
                # small arrow between this cell and the next (flow left->right)
                edge_off = min(width, spacing) * 0.05
                start_x = x + width - edge_off
                end_x = x + width + spacing - edge_off
                ax.annotate(
                    "",
                    xy=(end_x, y_center),
                    xytext=(start_x, y_center),
                    arrowprops=dict(arrowstyle="->", color="black"),
                )

            # draw onramp if present (attach near upstream side of cell)
            onramp = cell.onramp
            if onramp is not None:
                ramp_w = max(0.2, width * 0.5)
                ramp_h = max(0.4 * lane_h, onramp.lanes * lane_h) + 0.05
                rx = x + width * 0.2 - ramp_w / 2
                ry = lower + height + 0.2
                rrect = patches.Rectangle(
                    (rx, ry),
                    ramp_w,
                    ramp_h,
                    linewidth=1,
                    edgecolor="black",
                    facecolor="green",
                )
                ax.add_patch(rrect)

                # directional connector: arrow from ramp -> mainline (merge)
                small = min(width, ramp_w) * 0.03
                ax.annotate(
                    "",
                    xy=(x + width * 0.3 + 0.03, lower + height - small * 2),
                    xytext=(rx + ramp_w / 2, ry + ramp_h / 2),
                    arrowprops=dict(arrowstyle="->", color="green"),
                )
                ax.text(
                    rx + ramp_w / 2,
                    ry + ramp_h / 2,
                    f"Onramp\n[{onramp.lanes} lane(s)]",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white",
                )

            # draw offramp if present (attach near downstream side of cell)
            offramp = cell.offramp
            if offramp is not None:
                ramp_w = max(0.2, width * 0.5)
                ramp_h = max(0.4 * lane_h, offramp.lanes * lane_h) + 0.05
                rx = x + width * 0.8 - ramp_w / 2
                ry = lower - ramp_h - 0.2
                rrect = patches.Rectangle(
                    (rx, ry),
                    ramp_w,
                    ramp_h,
                    linewidth=1,
                    edgecolor="black",
                    facecolor="red",
                )
                ax.add_patch(rrect)

                # directional connector: arrow from mainline -> offramp (diverge)
                small = min(width, ramp_w) * 0.03
                ax.annotate(
                    "",
                    xy=(rx + ramp_w / 2, ry + ramp_h - small),
                    xytext=(x + width * 0.7 + 0.03, lower + small * 0),
                    arrowprops=dict(arrowstyle="->", color="red"),
                )
                ax.text(
                    rx + ramp_w / 2,
                    ry + ramp_h / 2,
                    f"Offramp\n[{offramp.lanes} lane(s)]",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white",
                )

            drawn_right = x + width
            x += width + spacing

        # finalize axes
        ax.set_aspect("auto")
        ax.set_xlim(-spacing, drawn_right + spacing)
        ax.set_ylim(-1.5, 1.5)
        ax.set_axis_off()
        ax.set_title("Traffic Network")
        plt.tight_layout()

        if save_path is not None:
            plt.savefig(save_path, dpi=200, bbox_inches="tight")

        if show:
            plt.show()

        return ax
