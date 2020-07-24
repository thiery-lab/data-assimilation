"""Progress bar for tracking progress of long iterative tasks.

Heavily influenced in design by the `tqdm` package (https://github.com/tqdm/tqdm) but
with simplified Jupyter notebook support without need for `ipywidgets`.
"""

from typing import Tuple, Sequence, Optional, TextIO
import html
import sys
from timeit import default_timer as timer

try:
    from IPython import get_ipython
    from IPython.display import display as ipython_display

    IPYTHON_AVAILABLE = True
except ImportError:
    IPYTHON_AVAILABLE = False

try:
    import google.colab

    ON_COLAB = True
except ImportError:
    ON_COLAB = False


def _in_zmq_interactive_shell() -> bool:
    """Check if in interactive ZMQ shell which supports updateable displays"""
    if not IPYTHON_AVAILABLE:
        return False
    elif ON_COLAB:
        return True
    else:
        try:
            shell = get_ipython().__class__.__name__
            if shell == "ZMQInteractiveShell":
                return True
            elif shell == "TerminalInteractiveShell":
                return False
            else:
                return False
        except NameError:
            return False


def _create_display(obj: object, position: Tuple[int, int]):
    """Create an updateable display object.

    Args:
        obj: Initial object to display.
        position: Tuple specifying position of display within
            a sequence of displays with first entry corresponding to the
            zero-indexed position and the second entry the total number of
            displays.

    Returns:
        Object with `update` method to update displayed content.
    """
    if _in_zmq_interactive_shell():
        return ipython_display(obj, display_id=True)
    else:
        return FileDisplay(position)


def _format_time(total_seconds: float) -> str:
    """Format a time interval in seconds as a colon-delimited string [h:]m:s"""
    total_mins, seconds = divmod(int(total_seconds), 60)
    hours, mins = divmod(total_mins, 60)
    if hours != 0:
        return f"{hours:d}:{mins:02d}:{seconds:02d}"
    else:
        return f"{mins:02d}:{seconds:02d}"


class ProgressBar:
    """Iterable object for tracking progress of an iterative task.

    Implements both string and HTML representations to allow richer
    display in interfaces which support HTML output, for example Jupyter
    notebooks or interactive terminals.
    """

    GLYPHS = " ▏▎▍▌▋▊▉█"
    """Characters used to create string representation of progress bar."""

    def __init__(
        self,
        sequence: Sequence,
        description: Optional[str] = None,
        position: Tuple[int, int] = (0, 1),
        displays: Optional[Sequence] = None,
        n_col: int = 10,
        unit: str = "it",
        min_refresh_time: float = 0.25,
    ):
        """
        Args:
            sequence: Sequence to iterate over. Must be iterable AND have a defined
                length such that `len(sequence)` is valid.
            description: Description of task to prefix progress  bar with.
            position: Tuple specifying position of progress bar within a sequence with
                first entry corresponding to zero-indexed position and the second entry
                the total number of progress bars.
            displays: Sequence of objects to use to display visual representation(s) of
                progress bar. Each object much have an `update` method which will be
                passed a single argument corresponding to the current progress bar.
            n_col: Number of columns (characters) to use in string representation of
                progress bar.
            unit: String describing unit of per-iteration tasks.
            min_referesh_time : Minimum time in seconds between each refresh of progress
                bar visual representation.
        """
        self._sequence = sequence
        self._description = description
        self._position = position
        self._active = False
        self._n_iter = len(sequence)
        self._n_col = n_col
        self._unit = unit
        self._counter = 0
        self._start_time = None
        self._elapsed_time = 0
        self._displays = displays
        self._min_refresh_time = min_refresh_time

    @property
    def sequence(self) -> Sequence:
        """Sequence iterated over."""
        return self._sequence

    @sequence.setter
    def sequence(self, value: Sequence):
        if self._active:
            raise RuntimeError("Cannot set sequence of active progress bar.")
        else:
            self._sequence = value
            self._n_iter = len(value)

    @property
    def n_iter(self) -> int:
        return self._n_iter

    def __iter__(self):
        for i, val in enumerate(self.sequence):
            yield val
            self.update(i + 1, refresh=True)

    def __len__(self):
        return self._n_iter

    @property
    def description(self) -> str:
        """"Description of task being tracked."""
        return self._description

    @property
    def counter(self) -> int:
        """Progress iteration count."""
        return self._counter

    @counter.setter
    def counter(self, value: int):
        self._counter = max(0, min(value, self.n_iter))

    @property
    def prop_complete(self) -> float:
        """Proportion complete (float value in [0, 1])."""
        return self.counter / self.n_iter

    @property
    def perc_complete(self) -> str:
        """Percentage complete formatted as string."""
        return f"{int(self.prop_complete * 100):3d}%"

    @property
    def elapsed_time(self) -> str:
        """Elapsed time formatted as string."""
        return _format_time(self._elapsed_time)

    @property
    def iter_rate(self) -> str:
        """Mean iteration rate if ≥ 1 `it/s` or reciprocal `s/it` as string."""
        if self.prop_complete == 0:
            return "?"
        else:
            mean_time = self._elapsed_time / self.counter
            return (
                f"{mean_time:.2f}s/{self._unit}"
                if mean_time > 1
                else f"{1/mean_time:.2f}{self._unit}/s"
            )

    @property
    def est_remaining_time(self) -> str:
        """Estimated remaining time to completion formatted as string."""
        if self.prop_complete == 0:
            return "?"
        else:
            return _format_time((1 / self.prop_complete - 1) * self._elapsed_time)

    @property
    def n_block_filled(self) -> int:
        """Number of filled blocks in progress bar."""
        return int(self._n_col * self.prop_complete)

    @property
    def n_block_empty(self) -> int:
        """Number of empty blocks in progress bar."""
        return self._n_col - self.n_block_filled

    @property
    def prop_partial_block(self) -> float:
        """Proportion filled in partial block in progress bar."""
        return self._n_col * self.prop_complete - self.n_block_filled

    @property
    def filled_blocks(self) -> str:
        """Filled blocks string."""
        return self.GLYPHS[-1] * self.n_block_filled

    @property
    def empty_blocks(self) -> str:
        """Empty blocks string."""
        if self.prop_partial_block == 0:
            return self.GLYPHS[0] * self.n_block_empty
        else:
            return self.GLYPHS[0] * (self.n_block_empty - 1)

    @property
    def partial_block(self) -> str:
        """Partial block character."""
        if self.prop_partial_block == 0:
            return ""
        else:
            return self.GLYPHS[int(len(self.GLYPHS) * self.prop_partial_block)]

    @property
    def progress_bar(self) -> str:
        """Progress bar string."""
        return f"|{self.filled_blocks}{self.partial_block}{self.empty_blocks}|"

    @property
    def bar_color(self) -> str:
        """CSS color property for HTML progress bar."""
        if self.counter == self.n_iter:
            return "var(--jp-success-color1, #4caf50)"
        elif self._active:
            return "var(--jp-brand-color1, #2196f3)"
        else:
            return "var(--jp-error-color1, #f44336)"

    @property
    def prefix(self) -> str:
        """Text to prefix progress bar with."""
        return (
            f'{self.description + ": "if self.description else ""}'
            f"{self.perc_complete}"
        )

    @property
    def postfix(self) -> str:
        """Text to postfix progress bar with."""
        return (
            f"{self.counter}/{self.n_iter} "
            f"[{self.elapsed_time}<{self.est_remaining_time}, {self.iter_rate}]"
        )

    def reset(self):
        """Reset progress bar state."""
        self._counter = 0
        self._start_time = timer()
        self._last_refresh_time = -float("inf")

    def update(self, iter_count: int, refresh: bool = True):
        """Update progress bar state.

        Args:
            iter_count: New value for iteration counter.
            refresh: Whether to refresh display(s).
        """
        if iter_count == 0:
            self.reset()
        else:
            self.counter = iter_count
            self._elapsed_time = timer() - self._start_time
        if (
            refresh
            and iter_count == self.n_iter
            or (timer() - self._last_refresh_time > self._min_refresh_time)
        ):
            self.refresh()
            self._last_refresh_time = timer()

    def refresh(self):
        """Refresh visual display(s) of progress bar."""
        for display in self._displays:
            display.update(self)

    def __str__(self):
        return f"{self.prefix}{self.progress_bar}{self.postfix}"

    def __repr__(self):
        return self.__str__()

    def _repr_html_(self):
        return f"""
        <div style="line-height: 28px; width: 100%; display: flex;
                    flex-flow: row wrap; align-items: center;
                    position: relative; margin: 2px;">
          <label style="margin-right: 8px; flex-shrink: 0;
                        font-size: var(--jp-code-font-size, 13px);
                        font-family: var(--jp-code-font-family, monospace);">
            {html.escape(self.prefix).replace(' ', '&nbsp;')}
          </label>
          <div role="progressbar" aria-valuenow="{self.prop_complete}"
               aria-valuemin="0" aria-valuemax="1"
               style="position: relative; flex-grow: 1; align-self: stretch;
                      margin-top: 4px; margin-bottom: 4px;  height: initial;
                      background-color: #eee;">
            <div style="background-color: {self.bar_color}; position: absolute;
                        bottom: 0; left: 0; width: {self.perc_complete};
                        height: 100%;"></div>
          </div>
          <div style="margin-left: 8px; flex-shrink: 0;
                      font-family: var(--jp-code-font-family, monospace);
                      font-size: var(--jp-code-font-size, 13px);">
            {html.escape(self.postfix)}
          </div>
        </div>
        """

    def __enter__(self):
        self._active = True
        self.reset()
        if self._displays is None:
            self._displays = [_create_display(self, self._position)]
        return self

    def __exit__(self, *args):
        self._active = False
        if self.counter != self.n_iter:
            self.refresh()
        return False


class FileDisplay:
    """Use file which supports ANSI escape sequences as an updatable display"""

    CURSOR_UP = "\x1b[A"
    """ANSI escape sequence to move cursor up one line."""

    CURSOR_DOWN = "\x1b[B"
    """ANSI escape sequence to move cursor down one line."""

    def __init__(
        self, position: Tuple[int, int] = (0, 1), file: Optional[TextIO] = None
    ):
        """
        Args:
            position: Tuple specifying position of display line within a sequence lines
                with first entry corresponding to zero-indexed line and the second entry
                the total number of lines.
            file: File object to write updates to. Must support ANSI escape sequences
                `\\x1b[A}` (cursor up) and `\\x1b[B` (cursor down) for manipulating
                write position. Defaults to `sys.stdout` if `None`.
        """
        self._position = position
        self._file = file if file is not None else sys.stdout
        self._last_string_length = 0
        if self._position[0] == 0:
            self._file.write("\n" * self._position[1])
        self._file.flush()

    def _move_line(self, offset: int):
        self._file.write(self.CURSOR_DOWN * offset + self.CURSOR_UP * -offset)
        self._file.flush()

    def update(self, obj):
        self._move_line(self._position[0] - self._position[1])
        string = str(obj)
        self._file.write(f"{string: <{self._last_string_length}}\r")
        self._last_string_length = len(string)
        self._move_line(self._position[1] - self._position[0])
        self._file.flush()
