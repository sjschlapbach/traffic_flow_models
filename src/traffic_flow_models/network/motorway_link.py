import uuid
from typing import Optional

from .cell import Cell


class MotorwayLink:
    """A motorway link composed of connected `Cell` instances.

    The MotorwayLink class stores mainline `Cell` instances as a bidirectional
    linked list arranged from upstream to downstream. It provides convenience
    methods to add cells and attach or detach on-/off-ramps. The MotorwayLink does
    not perform simulation — it only manages the topology and basic
    validation when linking objects together. Link-level physical
    parameters (e.g. lane capacity, free-flow speed, jam density) are
    stored on the link; per-cell geometry is stored on the contained
    `Cell` objects.

    Attributes:
        length (float): Total link length in kilometers.
        lanes (int): Number of lanes on the motorway link.
        lane_capacity (float): Capacity per lane in vehicles per time.
        vf (float): Free-flow speed for the link (length per time).
        rho_jam (float): Jam density (vehicles per length per lane).
        id (str): Unique identifier for the link.
        origin_node_id (Optional[str]): Optional ID of the upstream node.
        destination_node_id (Optional[str]): Optional ID of the downstream node.
        _head (Optional[Cell]): Reference to the first (upstream) cell.
        _tail (Optional[Cell]): Reference to the last (downstream) cell.
        _cell_count (int): Number of cells in the motorway.
    """

    def __init__(
        self,
        length: float,
        lanes: int,
        lane_capacity: float,
        free_flow_speed: float,
        jam_density: float,
        id: str | None = None,
        origin_node_id: str | None = None,
        destination_node_id: str | None = None,
    ) -> None:
        """
        Initialize an empty motorway link.

        The created motorway link contains no cells initially. Cells are created
        automatically during the partitioning of the network according to the
        model requirements, the CFL condition and the desired cell size specificed
        by the user.
        """

        if length <= 0:
            raise ValueError("Link length must be positive.")

        if lanes <= 0:
            raise ValueError("Number of lanes must be positive.")

        if lane_capacity <= 0:
            raise ValueError("Lane capacity must be positive.")

        if free_flow_speed <= 0:
            raise ValueError("Free-flow speed must be positive.")

        if jam_density <= 0:
            raise ValueError("Jam density must be positive.")

        # set link parameters
        self.length: float = length  # in kilometers
        self.lanes: int = lanes  # number of lanes
        self.lane_capacity: float = lane_capacity  # in vehicles per hour per lane
        self.vf: float = free_flow_speed  # in kilometers per hour
        self.rho_jam: float = jam_density  # in vehicles per kilometer per lane

        # identifier
        self.id: str = id if id is not None else str(uuid.uuid4())

        # optional start / end node identifiers to be set during connection
        self.origin_node_id: str | None = origin_node_id
        self.destination_node_id: str | None = destination_node_id

        # bidirectional linked list structure
        self._head: Optional[Cell] = None
        self._tail: Optional[Cell] = None
        self._cell_count: int = 0

    def add_cell(
        self,
        length: float,
        upcoming_lane_drop: int = 0,
    ) -> Cell:
        """Create and append a new `Cell` to the motorway link.

        Constructs a `Cell` with the supplied geometric parameters and
        appends it to the downstream end of the link. If the link already
        contains cells, upstream/downstream pointers are updated so the new
        cell is connected in the bidirectional linked list.

        Args:
            length (float): Cell length in kilometers (must be positive).
            upcoming_lane_drop (int): Number of lanes dropping downstream
                of this cell (default: 0).

        Returns:
            Cell: The newly created `Cell` instance.

        Raises:
            ValueError: If `length` is non-positive.
        """

        # check if the motorway link already contains an upstream cell and if there is a lane drop
        new_cell = Cell(length=length, upcoming_lane_drop=upcoming_lane_drop)

        # link the new cell into the bidirectional linked list
        if self._head is None:
            # first cell in motorway link
            self._head = new_cell
            self._tail = new_cell
        elif self._tail is None:
            # should not happen - if head is defined, tail should be too
            raise RuntimeError("Motorway link linked list is in an invalid state")
        else:
            # append to end of list
            self._tail.downstream = new_cell
            new_cell.upstream = self._tail
            self._tail = new_cell

        self._cell_count += 1

        return new_cell

    def partition_link(
        self, preferred_cell_size: float, dt: float, upcoming_lane_drop: int = 0
    ) -> None:
        """Partition the motorway link into a sequence of `Cell` objects.

        Cells are created to approximate `preferred_cell_size` while ensuring
        the Courant-Friedrichs-Lewy (CFL) condition is satisfied for the
        link's free-flow speed given the timestep `dt`. The last cell may be
        slightly longer to match the total link length. `dt` is expected in
        the same time units as `vf` (e.g. hours if `vf` is km/h).

        Args:
            preferred_cell_size (float): Preferred cell length in kilometers.
            dt (float): Simulation timestep (same time units as `vf`).
            upcoming_lane_drop (int): Number of lanes dropping downstream of
                the last cell (default: 0).
        """

        if preferred_cell_size <= 0:
            raise ValueError("Preferred cell size must be positive.")

        # clear existing cells
        self._head = None
        self._tail = None
        self._cell_count = 0

        # determine maximum allowable cell size from CFL condition
        min_cell_length = self.vf * dt
        valid_cell_size = max(preferred_cell_size, min_cell_length + 0.001)
        num_cells = int(self.length // valid_cell_size)

        if num_cells == 0:
            raise ValueError(
                f"Motorway link too short to fit a single segment. Segment length: {self.length} km, minimum cell size: {valid_cell_size} km"
            )

        # compute the homogeneous cell size (as close as possible to the preferred one; rounded to meters)
        balanced_cell_size = round(self.length / num_cells, 3)

        # add cells of the preferred length (or as close as possible to satisfy CFL condition)
        for _ in range(num_cells - 1):
            self.add_cell(length=balanced_cell_size)

        # add the last cell with the optional lane drop parameter to the network
        # the length of the last cell is determined by the remaining length of the link
        last_cell_size = self.length - balanced_cell_size * (num_cells - 1)
        self.add_cell(length=last_cell_size, upcoming_lane_drop=upcoming_lane_drop)

    def __len__(self) -> int:
        """Return the number of cells in the motorway link."""
        return self._cell_count

    def __iter__(self):
        """Iterate over cells from upstream to downstream."""
        current = self._head
        while current is not None:
            yield current
            current = current.downstream

    def first_cell(self) -> Optional[Cell]:
        """Return the first (most upstream) cell, or None if motorway link is empty."""
        return self._head

    def last_cell(self) -> Optional[Cell]:
        """Return the last (most downstream) cell, or None if motorway link is empty."""
        return self._tail

    def enumerate_cells(self):
        """Iterate over (index, cell) tuples from upstream to downstream."""
        for i, cell in enumerate(self):
            yield i, cell

    def get_cell(self, index: int) -> Cell:
        """Get cell at specified index by traversing linked list.

        Args:
            index: Zero-based index of cell (0 = first/upstream cell).

        Returns:
            Cell at the specified index.

        Raises:
            IndexError: If index is out of bounds.
        """
        if index < 0 or index >= self._cell_count:
            raise IndexError(f"Cell index {index} out of range [0, {self._cell_count})")

        current = self._head
        for _ in range(index):
            if current is None:
                raise IndexError(f"Cell index {index} out of range")
            current = current.downstream

        if current is None:
            raise IndexError(f"Cell index {index} out of range")

        return current

    # TODO: remove code once it has been replaced through appropriate implementation on network level
    # def plot_simulation_results(
    #     self,
    #     time: NDArray[np.float64],
    #     flow: NDArray[np.float64],
    #     density: NDArray[np.float64],
    #     speed: NDArray[np.float64],
    #     mainline_demand_func: Callable[[float], float],
    #     input_flow: NDArray[np.float64],
    #     input_queue: NDArray[np.float64],
    #     onramp_demand_func: Callable[[float, int], NDArray[np.float64]],
    #     onramp_flow: NDArray[np.float64],
    #     onramp_queue: NDArray[np.float64],
    #     offramp_flow: NDArray[np.float64],
    # ) -> None:
    #     """Plot comprehensive simulation results for the motorway link.

    #     Produces multiple figures showing density, flow, speed, input and
    #     onramp demands/flows/queues and 3D surface visualizations. The
    #     provided arrays must match the motorway link's number of cells and the
    #     supplied `time` vector.

    #     Args:
    #         time: 1-D array of time points (hours).
    #         flow: 2-D array of flows per cell and time (veh/h), shape
    #             `(num_cells, time_steps)`.
    #         density: 2-D array of densities per cell and time
    #             (veh/km/lane), shape `(num_cells, time_steps)`.
    #         speed: 2-D array of speeds per cell and time (km/h), shape
    #             `(num_cells, time_steps)`.
    #         mainline_demand_func: Callable that returns mainline demand
    #             given time (hours).
    #         input_flow: 1-D array of input flows over time (veh/h).
    #         input_queue: 1-D array of input queue lengths over time (veh).
    #         onramp_demand_func: Callable that returns onramp demand array
    #             given time (hours) and number of cells.
    #         onramp_flow: 2-D array of onramp flows per cell and time (veh/h).
    #         onramp_queue: 2-D array of onramp queue lengths per cell and time
    #             (veh).

    #     Returns:
    #         None. Shows Matplotlib figures when called.
    #     """

    #     num_cells = len(self)
    #     time_seconds = time * 3600

    #     # calculate actual simulation duration
    #     actual_duration_seconds = time[-1] * 3600

    #     # prepare the demand arrays for the onramps and the mainline
    #     input_demand = np.array([mainline_demand_func(t) for t in time])
    #     onramp_demand = np.array([onramp_demand_func(t, num_cells) for t in time]).T

    #     # calculate max values for proper y-axis scaling
    #     max_density = np.max(density) * 1.1  # 10% margin
    #     max_speed = np.max(speed) * 1.1  # 10% margin
    #     max_input_demand = (
    #         np.max(input_demand) * 1.1 if np.max(input_demand) > 0 else 2500
    #     )
    #     max_input_flow = np.max(input_flow) * 1.1 if np.max(input_flow) > 0 else 2500
    #     max_input_queue = np.max(input_queue) * 1.1 if np.max(input_queue) > 0 else 100
    #     max_onramp_demand = (
    #         np.max(onramp_demand) * 1.1 if np.max(onramp_demand) > 0 else 5000
    #     )
    #     max_onramp_flow = np.max(onramp_flow) * 1.1 if np.max(onramp_flow) > 0 else 2500
    #     max_onramp_queue = (
    #         np.max(onramp_queue) * 1.1 if np.max(onramp_queue) > 0 else 100
    #     )

    #     # Figure 1: Vehicle Density (dynamic grid based on number of cells)
    #     ncols = 3
    #     nrows = math.ceil(num_cells / ncols)
    #     fig1, axes1 = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    #     fig1.suptitle("Vehicle Density", fontsize=14, fontweight="bold")
    #     axes1 = np.array(axes1).flatten()

    #     for i, cell in self.enumerate_cells():
    #         axes1[i].plot(time_seconds, density[i, :], linewidth=1.5)
    #         axes1[i].axhline(cell.rho_jam, color="red", linestyle="--", linewidth=1)
    #         axes1[i].set_ylim([0, max(cell.rho_jam * 1.1, max_density)])
    #         axes1[i].set_xlim([0, actual_duration_seconds])
    #         axes1[i].set_xlabel("time (s)")
    #         axes1[i].set_ylabel("density (veh/km/lane)")
    #         axes1[i].grid(True)
    #         axes1[i].set_title(f"Cell {i + 1}")

    #     # hide any unused axes
    #     for ax in axes1[num_cells:]:
    #         ax.set_visible(False)

    #     plt.tight_layout()

    #     # Figure 2: Vehicle Flow (dynamic grid)
    #     fig2, axes2 = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    #     fig2.suptitle("Vehicle Cell Outflow", fontsize=14, fontweight="bold")
    #     axes2 = np.array(axes2).flatten()

    #     for i, cell in self.enumerate_cells():
    #         Qc = cell.Qc
    #         # mainline outflow
    #         axes2[i].plot(
    #             time_seconds[:-1], flow[i, :-1], linewidth=1.5, label="Cell outflow"
    #         )
    #         axes2[i].axhline(Qc, color="red", linestyle="--", linewidth=1)

    #         has_offramp = cell.offramp is not None
    #         has_onramp = cell.onramp is not None
    #         max_off = 0
    #         max_on = 0
    #         max_input = 0

    #         # if an offramp exists for this cell, plot its outflow and the total outflow
    #         if has_offramp:
    #             axes2[i].plot(
    #                 time_seconds[:-1],
    #                 offramp_flow[i, :-1],
    #                 linewidth=1.2,
    #                 color="tab:orange",
    #                 label="Offramp outflow",
    #             )

    #             total_outflow = flow[i, :-1] + offramp_flow[i, :-1]
    #             axes2[i].plot(
    #                 time_seconds[:-1],
    #                 total_outflow,
    #                 linestyle="--",
    #                 linewidth=1.0,
    #                 color="gray",
    #                 label="Total outflow",
    #             )
    #             max_off = (
    #                 np.max(offramp_flow[i, :-1]) * 1.05
    #                 if np.max(offramp_flow[i, :-1]) > 0
    #                 else 0
    #             )

    #         # if an onramp exists for this cell, plot its flow on the same axes
    #         if has_onramp:
    #             axes2[i].plot(
    #                 time_seconds[:-1],
    #                 onramp_flow[i, :-1],
    #                 linewidth=1.2,
    #                 color="green",
    #                 label="Onramp inflow",
    #             )
    #             max_on = (
    #                 np.max(onramp_flow[i, :-1]) * 1.05
    #                 if np.max(onramp_flow[i, :-1]) > 0
    #                 else 0
    #             )

    #         # for the first cell, plot the input flow
    #         if i == 0:
    #             axes2[i].plot(
    #                 time_seconds[:-1],
    #                 input_flow[:-1],
    #                 linewidth=1.2,
    #                 color="green",
    #                 label="Input flow",
    #             )
    #             max_input = (
    #                 np.max(input_flow[:-1]) * 1.05 if np.max(input_flow[:-1]) > 0 else 0
    #             )

    #         # determine y-limit including possible offramp values
    #         axes2[i].set_ylim(
    #             [
    #                 0,
    #                 max(
    #                     Qc * 1.05,
    #                     np.max(flow[i, :-1]) * 1.05,
    #                     max_off,
    #                     max_on,
    #                     max_input,
    #                 ),
    #             ]
    #         )
    #         axes2[i].set_xlim([0, actual_duration_seconds])
    #         axes2[i].set_xlabel("time (s)")
    #         axes2[i].set_ylabel("flow (veh/h)")
    #         axes2[i].grid(True)
    #         axes2[i].set_title(f"Cell {i + 1}")

    #         # compact legend if any extra inflows/outflows were plotted
    #         if has_offramp or has_onramp or i == 0:
    #             axes2[i].legend(fontsize="small", frameon=False, loc="upper right")

    #     for ax in axes2[num_cells:]:
    #         ax.set_visible(False)

    #     plt.tight_layout()

    #     # Figure 3: Vehicle Speed (dynamic grid)
    #     fig3, axes3 = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    #     fig3.suptitle("Vehicle Speed", fontsize=14, fontweight="bold")
    #     axes3 = np.array(axes3).flatten()

    #     for i, cell in self.enumerate_cells():
    #         vf_cell = cell.vf
    #         axes3[i].plot(time_seconds[:-1], speed[i, :-1], linewidth=1.5)
    #         axes3[i].axhline(vf_cell, color="red", linestyle="--", linewidth=1)
    #         axes3[i].set_ylim([0, max(vf_cell * 1.05, max_speed)])
    #         axes3[i].set_xlim([0, actual_duration_seconds])
    #         axes3[i].set_xlabel("time (s)")
    #         axes3[i].set_ylabel("speed (km/h)")
    #         axes3[i].grid(True)
    #         axes3[i].set_title(f"Cell {i + 1}")

    #     for ax in axes3[num_cells:]:
    #         ax.set_visible(False)

    #     plt.tight_layout()

    #     # Figure 4: Input and Onramp Flows & Queues (combined demand+flow)
    #     # determine which cells have onramps
    #     onramp_cells = [
    #         i
    #         for i, c in self.enumerate_cells()
    #         if getattr(c, "onramp", None) is not None
    #     ]
    #     # rows: 1 for input, plus one per onramp
    #     rows = 1 + len(onramp_cells)
    #     ncols_4 = 2  # combined demand+flow, and queue
    #     fig4, axes4 = plt.subplots(rows, ncols_4, figsize=(6 * ncols_4, 3 * rows))
    #     fig4.suptitle("Input and Onramp Flows & Queues", fontsize=14, fontweight="bold")

    #     # normalize axes4 indexing to 2D
    #     axes4 = np.array(axes4).reshape(rows, ncols_4)
    #     axes4[0, 0].plot(
    #         time_seconds[:-1], input_demand[:-1], linewidth=1.5, label="Input Demand"
    #     )
    #     axes4[0, 0].plot(
    #         time_seconds[:-1], input_flow[:-1], linewidth=1.5, label="Input Flow"
    #     )
    #     axes4[0, 0].grid(True)
    #     axes4[0, 0].set_xlim([0, actual_duration_seconds])
    #     axes4[0, 0].set_ylim([0, max(max_input_demand, max_input_flow)])
    #     axes4[0, 0].set_xlabel("time (s)")
    #     axes4[0, 0].set_ylabel("veh/h")
    #     axes4[0, 0].set_title("Input Demand & Flow")
    #     axes4[0, 0].legend(fontsize="small", ncol=2, frameon=False)

    #     axes4[0, 1].plot(
    #         time_seconds[:-1], input_queue[:-1], linewidth=1.5, color="tab:gray"
    #     )
    #     axes4[0, 1].grid(True)
    #     axes4[0, 1].set_xlim([0, actual_duration_seconds])
    #     axes4[0, 1].set_ylim([0, max_input_queue])
    #     axes4[0, 1].set_xlabel("time (s)")
    #     axes4[0, 1].set_ylabel("Queue (veh)")
    #     axes4[0, 1].set_title("Input Queue")

    #     # for each onramp cell, plot combined demand+flow and queue in its own row
    #     for row_idx, cell_idx in enumerate(onramp_cells, start=1):
    #         # demand and flow combined
    #         max_d = (
    #             np.max(onramp_demand[cell_idx, :]) * 1.1
    #             if np.max(onramp_demand[cell_idx, :]) > 0
    #             else max_onramp_demand
    #         )
    #         max_f = (
    #             np.max(onramp_flow[cell_idx, :]) * 1.1
    #             if np.max(onramp_flow[cell_idx, :]) > 0
    #             else max_onramp_flow
    #         )
    #         combined_max = max(max_d, max_f)

    #         axes4[row_idx, 0].plot(
    #             time_seconds[:-1],
    #             onramp_demand[cell_idx, :-1],
    #             linewidth=1.5,
    #             label="Demand",
    #         )
    #         axes4[row_idx, 0].plot(
    #             time_seconds[:-1],
    #             onramp_flow[cell_idx, :-1],
    #             linewidth=1.5,
    #             label="Flow",
    #         )
    #         axes4[row_idx, 0].grid(True)
    #         axes4[row_idx, 0].set_xlim([0, actual_duration_seconds])
    #         axes4[row_idx, 0].set_ylim([0, combined_max])
    #         axes4[row_idx, 0].set_xlabel("time (s)")
    #         axes4[row_idx, 0].set_ylabel("veh/h")
    #         axes4[row_idx, 0].set_title(f"Onramp Demand & Flow (Cell {cell_idx + 1})")
    #         axes4[row_idx, 0].legend(fontsize="small", ncol=2, frameon=False)

    #         # onramp queue
    #         max_q = (
    #             np.max(onramp_queue[cell_idx, :]) * 1.1
    #             if np.max(onramp_queue[cell_idx, :]) > 0
    #             else max_onramp_queue
    #         )
    #         axes4[row_idx, 1].plot(
    #             time_seconds[:-1],
    #             onramp_queue[cell_idx, :-1],
    #             linewidth=1.5,
    #             color="tab:gray",
    #         )
    #         axes4[row_idx, 1].grid(True)
    #         axes4[row_idx, 1].set_xlim([0, actual_duration_seconds])
    #         axes4[row_idx, 1].set_ylim([0, max_q])
    #         axes4[row_idx, 1].set_xlabel("time (s)")
    #         axes4[row_idx, 1].set_ylabel("Queue (veh)")
    #         axes4[row_idx, 1].set_title(f"Onramp Queue (Cell {cell_idx + 1})")

    #     plt.tight_layout()

    #     # Figure 5: 3D Surface Plots
    #     fig5 = plt.figure(figsize=(18, 6))
    #     fig5.suptitle("3D Visualization", fontsize=14, fontweight="bold")

    #     # create meshgrid for 3D plots
    #     X_full, Y_full = np.meshgrid(time_seconds, np.arange(1, num_cells + 1))
    #     X_truncated, Y_truncated = np.meshgrid(
    #         time_seconds[:-1], np.arange(1, num_cells + 1)
    #     )

    #     # 3D density plot
    #     max_rho_jam = max(cell.rho_jam for cell in self)
    #     ax1 = fig5.add_subplot(1, 3, 1, projection="3d")
    #     ax1.plot_surface(
    #         X_full, Y_full, density, cmap="viridis", edgecolor="none", alpha=0.9
    #     )
    #     ax1.view_init(elev=30, azim=-37.5)
    #     ax1.set_xlabel("time (s)", rotation=30)
    #     ax1.set_ylabel("Cell", rotation=-37.5)
    #     ax1.set_zlabel("density (veh/km/lane)")
    #     ax1.set_xlim([0, actual_duration_seconds])
    #     ax1.set_ylim([1, num_cells])
    #     ax1.set_zlim([0, max(max_rho_jam * 1.1, max_density)])

    #     # 3D flow plot
    #     ax2 = fig5.add_subplot(1, 3, 2, projection="3d")
    #     ax2.plot_surface(
    #         X_truncated,
    #         Y_truncated,
    #         flow[:, :-1],
    #         cmap="viridis",
    #         edgecolor="none",
    #         alpha=0.9,
    #     )
    #     ax2.view_init(elev=30, azim=-37.5)
    #     ax2.set_xlabel("time (s)", rotation=30)
    #     ax2.set_ylabel("Cell", rotation=-37.5)
    #     ax2.set_zlabel("flow (veh/h)")
    #     ax2.set_xlim([0, actual_duration_seconds])
    #     ax2.set_ylim([1, num_cells])
    #     max_capacity = max(cell.Qc for cell in self)
    #     ax2.set_zlim([0, max_capacity])

    #     # 3D speed plot
    #     max_vf = max(cell.vf for cell in self)
    #     ax3 = fig5.add_subplot(1, 3, 3, projection="3d")
    #     ax3.plot_surface(
    #         X_truncated,
    #         Y_truncated,
    #         speed[:, :-1],
    #         cmap="viridis",
    #         edgecolor="none",
    #         alpha=0.9,
    #     )
    #     ax3.view_init(elev=30, azim=-37.5)
    #     ax3.set_xlabel("time (s)", rotation=30)
    #     ax3.set_ylabel("Cell", rotation=-37.5)
    #     ax3.set_zlabel("speed (km/h)")
    #     ax3.set_xlim([0, actual_duration_seconds])
    #     ax3.set_ylim([1, num_cells])
    #     ax3.set_zlim([0, max(max_vf * 1.1, max_speed)])
    #     plt.tight_layout()

    #     # show plots
    #     plt.show()
