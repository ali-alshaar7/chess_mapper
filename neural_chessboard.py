import math
import scipy.cluster
import scipy, cv2, pyclipper, numpy as np
import matplotlib.path, matplotlib.pyplot as plt
import matplotlib.path as mplPath
import collections, itertools, random, math
from copy import copy
from config import root_direc
import functools, os, re

na = np.array
from keras.models import model_from_json

"""
https://github.com/maciejczyzewski/neural-chessboard
"""
#algorithm for initially centering chessbaord

NC_SLID_CLAHE = [[3,   (2, 6),    5], # @1
		         [3,   (6, 2),    5], # @2
				 [5,   (3, 3),    5], # @3
				 [0,   (0, 0),    0]] # EE

__laps_model = root_direc + r'\board_centering\laps.model.json'
__laps_weights = root_direc + r'\board_centering\laps.weights.h5'
NC_LAPS_MODEL = model_from_json(open(__laps_model, 'r').read())
NC_LAPS_MODEL.load_weights(__laps_weights)

"""
https://github.com/ideasman42/isect_segments-bentley_ottmann
"""

# BentleyOttmann sweep-line implementation
# (for finding all intersections in a set of line segments)

__all__ = (
    "isect_segments",
    "isect_polygon",

    # for testing only (correct but slow)
    "isect_segments__naive",
    "isect_polygon__naive",
    )

# ----------------------------------------------------------------------------
# Main Poly Intersection

# Defines to change behavior.
#
# Whether to ignore intersections of line segments when both
# their end points form the intersection point.
USE_IGNORE_SEGMENT_ENDINGS = True

USE_DEBUG = False # FIXME

USE_VERBOSE = False

# checks we should NOT need,
# but do them in case we find a test-case that fails.
USE_PARANOID = False

# Support vertical segments,
# (the bentley-ottmann method doesn't support this).
# We use the term 'START_VERTICAL' for a vertical segment,
# to differentiate it from START/END/INTERSECTION
USE_VERTICAL = True
# end defines!
# ------------

# ---------
# Constants
X, Y = 0, 1
EPS = 1e-10
EPS_SQ = EPS * EPS
INF = float("inf")


class Event:
    __slots__ = (
        "type",
        "point",
        "segment",

        # this is just cache,
        # we may remove or calculate slope on the fly
        "slope",
        "span",
        ) + (() if not USE_DEBUG else (
         # debugging only
        "other",
        "in_sweep",
        ))

    class Type:
        END = 0
        INTERSECTION = 1
        START = 2
        if USE_VERTICAL:
            START_VERTICAL = 3

    def __init__(self, type, point, segment, slope):
        assert(isinstance(point, tuple))
        self.type = type
        self.point = point
        self.segment = segment

        # will be None for INTERSECTION
        self.slope = slope
        if segment is not None:
            self.span = segment[1][X] - segment[0][X]

        if USE_DEBUG:
            self.other = None
            self.in_sweep = False

    def is_vertical(self):
        return self.segment[0][X] == self.segment[1][X]

    def y_intercept_x(self, x: float):
        # vertical events only for comparison (above_all check)
        # never added into the binary-tree its self
        if USE_VERTICAL:
            if self.is_vertical():
                return None

        if x <= self.segment[0][X]:
            return self.segment[0][Y]
        elif x >= self.segment[1][X]:
            return self.segment[1][Y]

        # use the largest to avoid float precision error with nearly vertical lines.
        delta_x0 = x - self.segment[0][X]
        delta_x1 = self.segment[1][X] - x
        if delta_x0 > delta_x1:
            ifac = delta_x0 / self.span
            fac = 1.0 - ifac
        else:
            fac = delta_x1 / self.span
            ifac = 1.0 - fac
        assert(fac <= 1.0)
        return (self.segment[0][Y] * fac) + (self.segment[1][Y] * ifac)

    @staticmethod
    def Compare(sweep_line, this, that):
        if this is that:
            return 0
        if USE_DEBUG:
            if this.other is that:
                return 0
        current_point_x = sweep_line._current_event_point_x
        ipthis = this.y_intercept_x(current_point_x)
        ipthat = that.y_intercept_x(current_point_x)
        # print(ipthis, ipthat)
        if USE_VERTICAL:
            if ipthis is None:
                ipthis = this.point[Y]
            if ipthat is None:
                ipthat = that.point[Y]

        delta_y = ipthis - ipthat

        assert((delta_y < 0.0) == (ipthis < ipthat))
        # NOTE, VERY IMPORTANT TO USE EPSILON HERE!
        # otherwise w/ float precision errors we get incorrect comparisons
        # can get very strange & hard to debug output without this.
        if abs(delta_y) > EPS:
            return -1 if (delta_y < 0.0) else 1
        else:
            this_slope = this.slope
            that_slope = that.slope
            if this_slope != that_slope:
                if sweep_line._before:
                    return -1 if (this_slope > that_slope) else 1
                else:
                    return 1 if (this_slope > that_slope) else -1

        delta_x_p1 = this.segment[0][X] - that.segment[0][X]
        if delta_x_p1 != 0.0:
            return -1 if (delta_x_p1 < 0.0) else 1

        delta_x_p2 = this.segment[1][X] - that.segment[1][X]
        if delta_x_p2 != 0.0:
            return -1 if (delta_x_p2 < 0.0) else 1

        return 0

    def __repr__(self):
        return ("Event(0x%x, s0=%r, s1=%r, p=%r, type=%d, slope=%r)" % (
            id(self),
            self.segment[0], self.segment[1],
            self.point,
            self.type,
            self.slope,
            ))


class SweepLine:
    __slots__ = (
        # A map holding all intersection points mapped to the Events
        # that form these intersections.
        # {Point: set(Event, ...), ...}
        "intersections",
        "queue",

        # Events (sorted set of ordered events, no values)
        #
        # note: START & END events are considered the same so checking if an event is in the tree
        # will return true if its opposite side is found.
        # This is essential for the algorithm to work, and why we don't explicitly remove START events.
        # Instead, the END events are never added to the current sweep, and removing them also removes the start.
        "_events_current_sweep",
        # The point of the current Event.
        "_current_event_point_x",
        # A flag to indicate if we're slightly before or after the line.
        "_before",
        )

    def __init__(self):
        self.intersections = {}

        self._current_event_point_x = None
        self._events_current_sweep = RBTree(cmp=Event.Compare, cmp_data=self)
        self._before = True

    def get_intersections(self):
        return list(self.intersections.keys())

    # Checks if an intersection exists between two Events 'a' and 'b'.
    def _check_intersection(self, a: Event, b: Event):
        # Return immediately in case either of the events is null, or
        # if one of them is an INTERSECTION event.
        if ((a is None or b is None) or
                (a.type == Event.Type.INTERSECTION) or
                (b.type == Event.Type.INTERSECTION)):

            return

        if a is b:
            return

        # Get the intersection point between 'a' and 'b'.
        p = isect_seg_seg_v2_point(
                a.segment[0], a.segment[1],
                b.segment[0], b.segment[1])

        # No intersection exists.
        if p is None:
            return

        # If the intersection is formed by both the segment endings, AND
        # USE_IGNORE_SEGMENT_ENDINGS is true,
        # return from this method.
        if USE_IGNORE_SEGMENT_ENDINGS:
            if ((len_squared_v2v2(p, a.segment[0]) < EPS_SQ or
                 len_squared_v2v2(p, a.segment[1]) < EPS_SQ) and
                (len_squared_v2v2(p, b.segment[0]) < EPS_SQ or
                 len_squared_v2v2(p, b.segment[1]) < EPS_SQ)):

                return

        # Add the intersection.
        events_for_point = self.intersections.pop(p, set())
        is_new = len(events_for_point) == 0
        events_for_point.add(a)
        events_for_point.add(b)
        self.intersections[p] = events_for_point

        # If the intersection occurs to the right of the sweep line, OR
        # if the intersection is on the sweep line and it's above the
        # current event-point, add it as a new Event to the queue.
        if is_new and p[X] >= self._current_event_point_x:
            event_isect = Event(Event.Type.INTERSECTION, p, None, None)
            self.queue.offer(p, event_isect)

    def _sweep_to(self, p):
        if p[X] == self._current_event_point_x:
            # happens in rare cases,
            # we can safely ignore
            return

        self._current_event_point_x = p[X]

    def insert(self, event):
        assert(event not in self._events_current_sweep)
        assert(event.type != Event.Type.START_VERTICAL)
        if USE_DEBUG:
            assert(event.in_sweep == False)
            assert(event.other.in_sweep == False)

        self._events_current_sweep.insert(event, None)

        if USE_DEBUG:
            event.in_sweep = True
            event.other.in_sweep = True

    def remove(self, event):
        try:
            self._events_current_sweep.remove(event)
            if USE_DEBUG:
                assert(event.in_sweep == True)
                assert(event.other.in_sweep == True)
                event.in_sweep = False
                event.other.in_sweep = False
            return True
        except KeyError:
            if USE_DEBUG:
                assert(event.in_sweep == False)
                assert(event.other.in_sweep == False)
            return False

    def above(self, event):
        return self._events_current_sweep.succ_key(event, None)

    def below(self, event):
        return self._events_current_sweep.prev_key(event, None)

    '''
    def above_all(self, event):
        while True:
            event = self.above(event)
            if event is None:
                break
            yield event
    '''

    def above_all(self, event):
        # assert(event not in self._events_current_sweep)
        return self._events_current_sweep.key_slice(event, None, reverse=False)

    def handle(self, p, events_current):
        if len(events_current) == 0:
            return
        # done already
        # self._sweep_to(events_current[0])
        assert(p[0] == self._current_event_point_x)

        if not USE_IGNORE_SEGMENT_ENDINGS:
            if len(events_current) > 1:
                for i in range(0, len(events_current) - 1):
                    for j in range(i + 1, len(events_current)):
                        self._check_intersection(
                                events_current[i], events_current[j])

        for e in events_current:
            self.handle_event(e)

    def handle_event(self, event):
        t = event.type
        if t == Event.Type.START:
            # print("  START")
            self._before = False
            self.insert(event)

            e_above = self.above(event)
            e_below = self.below(event)

            self._check_intersection(event, e_above)
            self._check_intersection(event, e_below)
            if USE_PARANOID:
                self._check_intersection(e_above, e_below)

        elif t == Event.Type.END:
            # print("  END")
            self._before = True

            e_above = self.above(event)
            e_below = self.below(event)

            self.remove(event)

            self._check_intersection(e_above, e_below)
            if USE_PARANOID:
                self._check_intersection(event, e_above)
                self._check_intersection(event, e_below)

        elif t == Event.Type.INTERSECTION:
            # print("  INTERSECTION")
            self._before = True
            event_set = self.intersections[event.point]
            # note: events_current aren't sorted.
            reinsert_stack = []  # Stack
            for e in event_set:
                # If we the Event was not already removed,
                # we want to insert it later on.
                if self.remove(e):
                    reinsert_stack.append(e)
            self._before = False

            # Insert all Events that we were able to remove.
            while reinsert_stack:
                e = reinsert_stack.pop()

                self.insert(e)

                e_above = self.above(e)
                e_below = self.below(e)

                self._check_intersection(e, e_above)
                self._check_intersection(e, e_below)
                if USE_PARANOID:
                    self._check_intersection(e_above, e_below)
        elif (USE_VERTICAL and
                (t == Event.Type.START_VERTICAL)):

            # just check sanity
            assert(event.segment[0][X] == event.segment[1][X])
            assert(event.segment[0][Y] <= event.segment[1][Y])

            # In this case we only need to find all segments in this span.
            y_above_max = event.segment[1][Y]

            # self.insert(event)
            for e_above in self.above_all(event):
                if e_above.type == Event.Type.START_VERTICAL:
                    continue
                y_above = e_above.y_intercept_x(
                        self._current_event_point_x)
                if USE_IGNORE_SEGMENT_ENDINGS:
                    if y_above >= y_above_max:
                        break
                else:
                    if y_above > y_above_max:
                        break

                # We know this intersects,
                # so we could use a faster function now:
                # ix = (self._current_event_point_x, y_above)
                # ...however best use existing functions
                # since it does all sanity checks on endpoints... etc.
                self._check_intersection(event, e_above)

            # self.remove(event)


class EventQueue:
    __slots__ = (
        # note: we only ever pop_min, this could use a 'heap' structure.
        # The sorted map holding the points -> event list
        # [Point: Event] (tree)
        "events_scan",
        )

    def __init__(self, segments, line: SweepLine):
        self.events_scan = RBTree()
        # segments = [s for s in segments if s[0][0] != s[1][0] and s[0][1] != s[1][1]]

        for s in segments:
            assert(s[0][X] <= s[1][X])

            slope = slope_v2v2(*s)

            if s[0] == s[1]:
                pass
            elif USE_VERTICAL and (s[0][X] == s[1][X]):
                e_start = Event(Event.Type.START_VERTICAL, s[0], s, slope)

                if USE_DEBUG:
                    e_start.other = e_start  # FAKE, avoid error checking

                self.offer(s[0], e_start)
            else:
                e_start = Event(Event.Type.START, s[0], s, slope)
                e_end   = Event(Event.Type.END,   s[1], s, slope)

                if USE_DEBUG:
                    e_start.other = e_end
                    e_end.other = e_start

                self.offer(s[0], e_start)
                self.offer(s[1], e_end)

        line.queue = self

    def offer(self, p, e: Event):
        """
        Offer a new event ``s`` at point ``p`` in this queue.
        """
        existing = self.events_scan.setdefault(
                p, ([], [], [], []) if USE_VERTICAL else
                   ([], [], []))
        # Can use double linked-list for easy insertion at beginning/end
        '''
        if e.type == Event.Type.END:
            existing.insert(0, e)
        else:
            existing.append(e)
        '''

        existing[e.type].append(e)

    # return a set of events
    def poll(self):
        """
        Get, and remove, the first (lowest) item from this queue.
        :return: the first (lowest) item from this queue.
        :rtype: Point, Event pair.
        """
        assert(len(self.events_scan) != 0)
        p, events_current = self.events_scan.pop_min()
        return p, events_current


def isect_segments(segments) -> list:
    # order points left -> right
    segments = [
        # in nearly all cases, comparing X is enough,
        # but compare Y too for vertical lines
        (s[0], s[1]) if (s[0] <= s[1]) else
        (s[1], s[0])
        for s in segments]

    sweep_line = SweepLine()
    queue = EventQueue(segments, sweep_line)

    while len(queue.events_scan) > 0:
        if USE_VERBOSE:
            print(len(queue.events_scan), sweep_line._current_event_point_x)
        p, e_ls = queue.poll()
        for events_current in e_ls:
            if events_current:
                sweep_line._sweep_to(p)
                sweep_line.handle(p, events_current)

    return sweep_line.get_intersections()


def isect_polygon(points) -> list:
    n = len(points)
    segments = [
        (tuple(points[i]), tuple(points[(i + 1) % n]))
        for i in range(n)]
    return isect_segments(segments)


# ----------------------------------------------------------------------------
# 2D math utilities


def slope_v2v2(p1, p2):
    if p1[X] == p2[X]:
        if p1[Y] < p2[Y]:
            return INF
        else:
            return -INF
    else:
        return (p2[Y] - p1[Y]) / (p2[X] - p1[X])


def sub_v2v2(a, b):
    return (
        a[0] - b[0],
        a[1] - b[1])


def dot_v2v2(a, b):
    return (
        (a[0] * b[0]) +
        (a[1] * b[1]))


def len_squared_v2v2(a, b):
    c = sub_v2v2(a, b)
    return dot_v2v2(c, c)


def line_point_factor_v2(p, l1, l2, default=0.0):
    u = sub_v2v2(l2, l1)
    h = sub_v2v2(p, l1)
    dot = dot_v2v2(u, u)
    return (dot_v2v2(u, h) / dot) if dot != 0.0 else default


def isect_seg_seg_v2_point(v1, v2, v3, v4, bias=0.0):
    # Only for predictability and hashable point when same input is given
    if v1 > v2:
        v1, v2 = v2, v1
    if v3 > v4:
        v3, v4 = v4, v3

    if (v1, v2) > (v3, v4):
        v1, v2, v3, v4 = v3, v4, v1, v2

    div = (v2[0] - v1[0]) * (v4[1] - v3[1]) - (v2[1] - v1[1]) * (v4[0] - v3[0])
    if div == 0.0:
        return None

    vi = (((v3[0] - v4[0]) *
           (v1[0] * v2[1] - v1[1] * v2[0]) - (v1[0] - v2[0]) *
           (v3[0] * v4[1] - v3[1] * v4[0])) / div,
          ((v3[1] - v4[1]) *
           (v1[0] * v2[1] - v1[1] * v2[0]) - (v1[1] - v2[1]) *
           (v3[0] * v4[1] - v3[1] * v4[0])) / div,
          )

    fac = line_point_factor_v2(vi, v1, v2, default=-1.0)
    if fac < 0.0 - bias or fac > 1.0 + bias:
        return None

    fac = line_point_factor_v2(vi, v3, v4, default=-1.0)
    if fac < 0.0 - bias or fac > 1.0 + bias:
        return None

    # vi = round(vi[X], 8), round(vi[Y], 8)
    return vi


# ----------------------------------------------------------------------------
# Simple naive line intersect, (for testing only)


def isect_segments__naive(segments) -> list:
    """
    Brute force O(n2) version of ``isect_segments`` for test validation.
    """
    isect = []

    # order points left -> right
    segments = [
        (s[0], s[1]) if s[0][X] <= s[1][X] else
        (s[1], s[0])
        for s in segments]

    n = len(segments)

    for i in range(n):
        a0, a1 = segments[i]
        for j in range(i + 1, n):
            b0, b1 = segments[j]
            if a0 not in (b0, b1) and a1 not in (b0, b1):
                ix = isect_seg_seg_v2_point(a0, a1, b0, b1)
                if ix is not None:
                    # USE_IGNORE_SEGMENT_ENDINGS handled already
                    isect.append(ix)

    return isect


def isect_polygon__naive(points) -> list:
    """
    Brute force O(n2) version of ``isect_polygon`` for test validation.
    """
    isect = []

    n = len(points)

    for i in range(n):
        a0, a1 = points[i], points[(i + 1) % n]
        for j in range(i + 1, n):
            b0, b1 = points[j], points[(j + 1) % n]
            if a0 not in (b0, b1) and a1 not in (b0, b1):
                ix = isect_seg_seg_v2_point(a0, a1, b0, b1)
                if ix is not None:

                    if USE_IGNORE_SEGMENT_ENDINGS:
                        if ((len_squared_v2v2(ix, a0) < EPS_SQ or
                             len_squared_v2v2(ix, a1) < EPS_SQ) and
                            (len_squared_v2v2(ix, b0) < EPS_SQ or
                             len_squared_v2v2(ix, b1) < EPS_SQ)):
                            continue

                    isect.append(ix)

    return isect


# ----------------------------------------------------------------------------
# Inline Libs
#
# bintrees: 2.0.2, extracted from:
# http://pypi.python.org/pypi/bintrees
#
# - Removed unused functions, such as slicing and range iteration.
# - Added 'cmp' and and 'cmp_data' arguments,
#   so we can define our own comparison that takes an arg.
#   Needed for sweep-line.
# - Added support for 'default' arguments for prev_item/succ_item,
#   so we can avoid exception handling.

# -------
# ABCTree

from operator import attrgetter
_sentinel = object()


class _ABCTree(object):
    def __init__(self, items=None, cmp=None, cmp_data=None):
        """T.__init__(...) initializes T; see T.__class__.__doc__ for signature"""
        self._root = None
        self._count = 0
        if cmp is None:
            def cmp(cmp_data, a, b):
                if a < b:
                    return -1
                elif a > b:
                    return 1
                else:
                    return 0
        self._cmp = cmp
        self._cmp_data = cmp_data
        if items is not None:
            self.update(items)

    def clear(self):
        """T.clear() -> None.  Remove all items from T."""
        def _clear(node):
            if node is not None:
                _clear(node.left)
                _clear(node.right)
                node.free()
        _clear(self._root)
        self._count = 0
        self._root = None

    @property
    def count(self):
        """Get items count."""
        return self._count

    def get_value(self, key):
        node = self._root
        while node is not None:
            cmp = self._cmp(self._cmp_data, key, node.key)
            if cmp == 0:
                return node.value
            elif cmp < 0:
                node = node.left
            else:
                node = node.right
        raise KeyError(str(key))

    def pop_item(self):
        """T.pop_item() -> (k, v), remove and return some (key, value) pair as a
        2-tuple; but raise KeyError if T is empty.
        """
        if self.is_empty():
            raise KeyError("pop_item(): tree is empty")
        node = self._root
        while True:
            if node.left is not None:
                node = node.left
            elif node.right is not None:
                node = node.right
            else:
                break
        key = node.key
        value = node.value
        self.remove(key)
        return key, value
    popitem = pop_item  # for compatibility  to dict()

    def min_item(self):
        """Get item with min key of tree, raises ValueError if tree is empty."""
        if self.is_empty():
            raise ValueError("Tree is empty")
        node = self._root
        while node.left is not None:
            node = node.left
        return node.key, node.value

    def max_item(self):
        """Get item with max key of tree, raises ValueError if tree is empty."""
        if self.is_empty():
            raise ValueError("Tree is empty")
        node = self._root
        while node.right is not None:
            node = node.right
        return node.key, node.value

    def succ_item(self, key, default=_sentinel):
        """Get successor (k,v) pair of key, raises KeyError if key is max key
        or key does not exist. optimized for pypy.
        """
        # removed graingets version, because it was little slower on CPython and much slower on pypy
        # this version runs about 4x faster with pypy than the Cython version
        # Note: Code sharing of succ_item() and ceiling_item() is possible, but has always a speed penalty.
        node = self._root
        succ_node = None
        while node is not None:
            cmp = self._cmp(self._cmp_data, key, node.key)
            if cmp == 0:
                break
            elif cmp < 0:
                if (succ_node is None) or self._cmp(self._cmp_data, node.key, succ_node.key) < 0:
                    succ_node = node
                node = node.left
            else:
                node = node.right

        if node is None:  # stay at dead end
            if default is _sentinel:
                raise KeyError(str(key))
            return default
        # found node of key
        if node.right is not None:
            # find smallest node of right subtree
            node = node.right
            while node.left is not None:
                node = node.left
            if succ_node is None:
                succ_node = node
            elif self._cmp(self._cmp_data, node.key, succ_node.key) < 0:
                succ_node = node
        elif succ_node is None:  # given key is biggest in tree
            if default is _sentinel:
                raise KeyError(str(key))
            return default
        return succ_node.key, succ_node.value

    def prev_item(self, key, default=_sentinel):
        """Get predecessor (k,v) pair of key, raises KeyError if key is min key
        or key does not exist. optimized for pypy.
        """
        # removed graingets version, because it was little slower on CPython and much slower on pypy
        # this version runs about 4x faster with pypy than the Cython version
        # Note: Code sharing of prev_item() and floor_item() is possible, but has always a speed penalty.
        node = self._root
        prev_node = None

        while node is not None:
            cmp = self._cmp(self._cmp_data, key, node.key)
            if cmp == 0:
                break
            elif cmp < 0:
                node = node.left
            else:
                if (prev_node is None) or self._cmp(self._cmp_data, prev_node.key, node.key) < 0:
                    prev_node = node
                node = node.right

        if node is None:  # stay at dead end (None)
            if default is _sentinel:
                raise KeyError(str(key))
            return default
        # found node of key
        if node.left is not None:
            # find biggest node of left subtree
            node = node.left
            while node.right is not None:
                node = node.right
            if prev_node is None:
                prev_node = node
            elif self._cmp(self._cmp_data, prev_node.key, node.key) < 0:
                prev_node = node
        elif prev_node is None:  # given key is smallest in tree
            if default is _sentinel:
                raise KeyError(str(key))
            return default
        return prev_node.key, prev_node.value

    def __repr__(self):
        """T.__repr__(...) <==> repr(x)"""
        tpl = "%s({%s})" % (self.__class__.__name__, '%s')
        return tpl % ", ".join(("%r: %r" % item for item in self.items()))

    def __contains__(self, key):
        """k in T -> True if T has a key k, else False"""
        try:
            self.get_value(key)
            return True
        except KeyError:
            return False

    def __len__(self):
        """T.__len__() <==> len(x)"""
        return self.count

    def is_empty(self):
        """T.is_empty() -> False if T contains any items else True"""
        return self.count == 0

    def set_default(self, key, default=None):
        """T.set_default(k[,d]) -> T.get(k,d), also set T[k]=d if k not in T"""
        try:
            return self.get_value(key)
        except KeyError:
            self.insert(key, default)
            return default
    setdefault = set_default  # for compatibility to dict()

    def get(self, key, default=None):
        """T.get(k[,d]) -> T[k] if k in T, else d.  d defaults to None."""
        try:
            return self.get_value(key)
        except KeyError:
            return default

    def pop(self, key, *args):
        """T.pop(k[,d]) -> v, remove specified key and return the corresponding value.
        If key is not found, d is returned if given, otherwise KeyError is raised
        """
        if len(args) > 1:
            raise TypeError("pop expected at most 2 arguments, got %d" % (1 + len(args)))
        try:
            value = self.get_value(key)
            self.remove(key)
            return value
        except KeyError:
            if len(args) == 0:
                raise
            else:
                return args[0]

    def prev_key(self, key, default=_sentinel):
        """Get predecessor to key, raises KeyError if key is min key
        or key does not exist.
        """
        item = self.prev_item(key, default)
        return default if item is default else item[0]

    def succ_key(self, key, default=_sentinel):
        """Get successor to key, raises KeyError if key is max key
        or key does not exist.
        """
        item = self.succ_item(key, default)
        return default if item is default else item[0]

    def pop_min(self):
        """T.pop_min() -> (k, v), remove item with minimum key, raise ValueError
        if T is empty.
        """
        item = self.min_item()
        self.remove(item[0])
        return item

    def pop_max(self):
        """T.pop_max() -> (k, v), remove item with maximum key, raise ValueError
        if T is empty.
        """
        item = self.max_item()
        self.remove(item[0])
        return item

    def min_key(self):
        """Get min key of tree, raises ValueError if tree is empty. """
        return self.min_item()[0]

    def max_key(self):
        """Get max key of tree, raises ValueError if tree is empty. """
        return self.max_item()[0]

    def key_slice(self, start_key, end_key, reverse=False):
        """T.key_slice(start_key, end_key) -> key iterator:
        start_key <= key < end_key.
        Yields keys in ascending order if reverse is False else in descending order.
        """
        return (k for k, v in self.iter_items(start_key, end_key, reverse=reverse))

    def iter_items(self,  start_key=None, end_key=None, reverse=False):
        """Iterates over the (key, value) items of the associated tree,
        in ascending order if reverse is True, iterate in descending order,
        reverse defaults to False"""
        # optimized iterator (reduced method calls) - faster on CPython but slower on pypy

        if self.is_empty():
            return []
        if reverse:
            return self._iter_items_backward(start_key, end_key)
        else:
            return self._iter_items_forward(start_key, end_key)

    def _iter_items_forward(self, start_key=None, end_key=None):
        for item in self._iter_items(left=attrgetter("left"), right=attrgetter("right"),
                                     start_key=start_key, end_key=end_key):
            yield item

    def _iter_items_backward(self, start_key=None, end_key=None):
        for item in self._iter_items(left=attrgetter("right"), right=attrgetter("left"),
                                     start_key=start_key, end_key=end_key):
            yield item

    def _iter_items(self, left=attrgetter("left"), right=attrgetter("right"), start_key=None, end_key=None):
        node = self._root
        stack = []
        go_left = True
        in_range = self._get_in_range_func(start_key, end_key)

        while True:
            if left(node) is not None and go_left:
                stack.append(node)
                node = left(node)
            else:
                if in_range(node.key):
                    yield node.key, node.value
                if right(node) is not None:
                    node = right(node)
                    go_left = True
                else:
                    if not len(stack):
                        return  # all done
                    node = stack.pop()
                    go_left = False

    def _get_in_range_func(self, start_key, end_key):
        if start_key is None and end_key is None:
            return lambda x: True
        else:
            if start_key is None:
                start_key = self.min_key()
            if end_key is None:
                return (lambda x: self._cmp(self._cmp_data, start_key, x) <= 0)
            else:
                return (lambda x: self._cmp(self._cmp_data, start_key, x) <= 0 and
                        self._cmp(self._cmp_data, x, end_key) < 0)


# ------
# RBTree

class Node(object):
    """Internal object, represents a tree node."""
    __slots__ = ['key', 'value', 'red', 'left', 'right']

    def __init__(self, key=None, value=None):
        self.key = key
        self.value = value
        self.red = True
        self.left = None
        self.right = None

    def free(self):
        self.left = None
        self.right = None
        self.key = None
        self.value = None

    def __getitem__(self, key):
        """N.__getitem__(key) <==> x[key], where key is 0 (left) or 1 (right)."""
        return self.left if key == 0 else self.right

    def __setitem__(self, key, value):
        """N.__setitem__(key, value) <==> x[key]=value, where key is 0 (left) or 1 (right)."""
        if key == 0:
            self.left = value
        else:
            self.right = value


class RBTree(_ABCTree):
    """
    RBTree implements a balanced binary tree with a dict-like interface.
    see: http://en.wikipedia.org/wiki/Red_black_tree
    """
    @staticmethod
    def is_red(node):
        if (node is not None) and node.red:
            return True
        else:
            return False

    @staticmethod
    def jsw_single(root, direction):
        other_side = 1 - direction
        save = root[other_side]
        root[other_side] = save[direction]
        save[direction] = root
        root.red = True
        save.red = False
        return save

    @staticmethod
    def jsw_double(root, direction):
        other_side = 1 - direction
        root[other_side] = RBTree.jsw_single(root[other_side], other_side)
        return RBTree.jsw_single(root, direction)

    def _new_node(self, key, value):
        """Create a new tree node."""
        self._count += 1
        return Node(key, value)

    def insert(self, key, value):
        """T.insert(key, value) <==> T[key] = value, insert key, value into tree."""
        if self._root is None:  # Empty tree case
            self._root = self._new_node(key, value)
            self._root.red = False  # make root black
            return

        head = Node()  # False tree root
        grand_parent = None
        grand_grand_parent = head
        parent = None  # parent
        direction = 0
        last = 0

        # Set up helpers
        grand_grand_parent.right = self._root
        node = grand_grand_parent.right
        # Search down the tree
        while True:
            if node is None:  # Insert new node at the bottom
                node = self._new_node(key, value)
                parent[direction] = node
            elif RBTree.is_red(node.left) and RBTree.is_red(node.right):  # Color flip
                node.red = True
                node.left.red = False
                node.right.red = False

            # Fix red violation
            if RBTree.is_red(node) and RBTree.is_red(parent):
                direction2 = 1 if grand_grand_parent.right is grand_parent else 0
                if node is parent[last]:
                    grand_grand_parent[direction2] = RBTree.jsw_single(grand_parent, 1 - last)
                else:
                    grand_grand_parent[direction2] = RBTree.jsw_double(grand_parent, 1 - last)

            # Stop if found
            if self._cmp(self._cmp_data, key, node.key) == 0:
                node.value = value  # set new value for key
                break

            last = direction
            direction = 0 if (self._cmp(self._cmp_data, key, node.key) < 0) else 1
            # Update helpers
            if grand_parent is not None:
                grand_grand_parent = grand_parent
            grand_parent = parent
            parent = node
            node = node[direction]

        self._root = head.right  # Update root
        self._root.red = False  # make root black

    def remove(self, key):
        """T.remove(key) <==> del T[key], remove item <key> from tree."""
        if self._root is None:
            raise KeyError(str(key))
        head = Node()  # False tree root
        node = head
        node.right = self._root
        parent = None
        grand_parent = None
        found = None  # Found item
        direction = 1

        # Search and push a red down
        while node[direction] is not None:
            last = direction

            # Update helpers
            grand_parent = parent
            parent = node
            node = node[direction]

            direction = 1 if (self._cmp(self._cmp_data, node.key, key) < 0) else 0

            # Save found node
            if self._cmp(self._cmp_data, key, node.key) == 0:
                found = node

            # Push the red node down
            if not RBTree.is_red(node) and not RBTree.is_red(node[direction]):
                if RBTree.is_red(node[1 - direction]):
                    parent[last] = RBTree.jsw_single(node, direction)
                    parent = parent[last]
                elif not RBTree.is_red(node[1 - direction]):
                    sibling = parent[1 - last]
                    if sibling is not None:
                        if (not RBTree.is_red(sibling[1 - last])) and (not RBTree.is_red(sibling[last])):
                            # Color flip
                            parent.red = False
                            sibling.red = True
                            node.red = True
                        else:
                            direction2 = 1 if grand_parent.right is parent else 0
                            if RBTree.is_red(sibling[last]):
                                grand_parent[direction2] = RBTree.jsw_double(parent, last)
                            elif RBTree.is_red(sibling[1-last]):
                                grand_parent[direction2] = RBTree.jsw_single(parent, last)
                            # Ensure correct coloring
                            grand_parent[direction2].red = True
                            node.red = True
                            grand_parent[direction2].left.red = False
                            grand_parent[direction2].right.red = False

        # Replace and remove if found
        if found is not None:
            found.key = node.key
            found.value = node.value
            parent[int(parent.right is node)] = node[int(node.left is None)]
            node.free()
            self._count -= 1

        # Update root and make it black
        self._root = head.right
        if self._root is not None:
            self._root.red = False
        if not found:
            raise KeyError(str(key))

def slid_canny(img, sigma=0.25):
	"""apply Canny edge detector (automatic thresh)"""
	v = np.median(img)
	img = cv2.medianBlur(img, 5)
	img = cv2.GaussianBlur(img, (7, 7), 2)
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	return cv2.Canny(img, lower, upper)

def slid_detector(img, alfa=150, beta=2):
	"""detect lines using Hough algorithm"""
	__lines, lines = [], cv2.HoughLinesP(img, rho=1, theta=np.pi/360*beta,
		threshold=40, minLineLength=50, maxLineGap=15) # [40, 40, 10]
	if lines is None: return []
	for line in np.reshape(lines, (-1, 4)):
		__lines += [[[int(line[0]), int(line[1])],
			         [int(line[2]), int(line[3])]]]
	return __lines

def slid_clahe(img, limit=2, grid=(3,3), iters=5):
	"""repair using CLAHE algorithm (adaptive histogram equalization)"""
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	for i in range(iters):
		img = cv2.createCLAHE(clipLimit=limit, \
				tileGridSize=grid).apply(img)
	#debug.image(img).save("slid_clahe_@1")
	if limit != 0:
		kernel = np.ones((10, 10), np.uint8)
		img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
		#debug.image(img).save("slid_clahe_@2")
	return img

################################################################################

def pSLID(img, thresh=150):
	"""find all lines using different settings"""
	#print(utils.call("pSLID(img)"))
	segments = []; i = 0
	for key, arr in enumerate(NC_SLID_CLAHE):
		tmp = slid_clahe(img, limit=arr[0], grid=arr[1], iters=arr[2])
		__segments = list(slid_detector(slid_canny(tmp), thresh))
		segments += __segments; i += 1
		print("FILTER: {} {} : {}".format(i, arr, len(__segments)))
		#debug.image(slid_canny(tmp)).lines(__segments).save("pslid_F%d" % i)
	return segments

all_points = []
def SLID(img, segments):
	# FIXME: zrobic 2 rodzaje haszowania (katy + pasy [blad - delta])
	#print(utils.call("SLID(img, segments)"))
	
	global all_points; all_points = []
	pregroup, group, hashmap, raw_lines = [[], []], {}, {}, []

	__cache = {}
	def __dis(a, b):
		idx = hash("__dis" + str(a) + str(b))
		if idx in __cache: return __cache[idx]
		__cache[idx] = np.linalg.norm(na(a)-na(b))
		return __cache[idx]

	X = {}
	def __fi(x):
		if x not in X: X[x] = 0;
		if (X[x] == x or X[x] == 0): X[x] = x
		else:                        X[x] = __fi(X[x])
		return X[x]
	def __un(a, b):
		ia, ib = __fi(a), __fi(b)
		X[ia] = ib; group[ib] |= group[ia]
		#group[ia] = set()
		#group[ia] = set()

	# shortest path // height
	nln = lambda l1, x, dx: \
		np.linalg.norm(np.cross(na(l1[1])-na(l1[0]),
								na(l1[0])-na(   x)))/dx

	def __similar(l1, l2):
		da, db = __dis(l1[0], l1[1]), __dis(l2[0], l2[1])
		# if da > db: l1, l2, da, db = l2, l1, db, da

		d1a, d2a = nln(l1, l2[0], da), nln(l1, l2[1], da)
		d1b, d2b = nln(l2, l1[0], db), nln(l2, l1[1], db)
	
		ds = 0.25 * (d1a + d1b + d2a + d2b) + 0.00001
		#print(da, db, abs(da-db))
		#print(int(da/ds), int(db/ds), "|", int(abs(da-db)), int(da+db),
		#		int(da+db)/(int(abs(da-db))+0.00001))
		alfa = 0.0625 * (da + db) #15
		# FIXME: roznica???
		#if d1 + d2 == 0: d1 += 0.00001 # [FIXME]: divide by 0
		t1 = (da/ds > alfa and db/ds > alfa)
		if not t1: return False # [FIXME]: dist???
		return True

	def __generate(a, b, n):
		points = []; t = 1/n
		for i in range(n):
			x = a[0] + (b[0]-a[0]) * (i * t)
			y = a[1] + (b[1]-a[1]) * (i * t)
			points += [[int(x), int(y)]]
		return points

	def __analyze(group):
		global all_points
		points = []
		for idx in group:
			points += __generate(*hashmap[idx], 10)
		_, radius = cv2.minEnclosingCircle(na(points)); w = radius * (math.pi/2)
		vx, vy, cx, cy = cv2.fitLine(na(points), cv2.DIST_L2, 0, 0.01, 0.01)
		# debug.color()
		all_points += points
		return [[int(cx-vx*w), int(cy-vy*w)], [int(cx+vx*w), int(cy+vy*w)]]

	for l in segments:
		h = hash(str(l))
		t1 = l[0][0] - l[1][0]
		t2 = l[0][1] - l[1][1]
		hashmap[h] = l; group[h] = set([h]); X[h] = h
		if abs(t1) < abs(t2): pregroup[0].append(l)
		else:                 pregroup[1].append(l)

	for lines in pregroup:
		for i in range(len(lines)):
			l1 = lines[i]; h1 = hash(str(l1))
			#print(h1, __fi(h1))
			if (X[h1] != h1): continue
			#if (__fi(h1) != h1): continue
			for j in range(i+1, len(lines)):
				l2 = lines[j]; h2 = hash(str(l2))
				#if (__fi(h2) != h2): continue
				if (X[h2] != h2): continue
				#if (len(group[h2])==0): continue
				if not __similar(l1, l2): continue
				__un(h1, h2) # union & find
				# break # FIXME

	#__d = debug.image(img.shape)
	for i in group:
		#if (__fi(i) != i): continue
		if (X[i] != i): continue
		#if len(group[i]) == 0: continue
		ls = [hashmap[h] for h in group[i]]
		#__d.lines(ls, color=debug.color())
	#__d.save("slid_all_groups")

	for i in group:
		#if (__fi(i) != i): continue
		if (X[i] != i): continue
		#if len(group[i]) == 0: continue
		#if (__fi(i) != i): continue
		raw_lines += [__analyze(group[i])]
	#debug.image(img.shape).lines(raw_lines).save("slid_final")

	return raw_lines

def slid_tendency(raw_lines, s=4): # FIXME: [1.25 -> 2]
	#print(utils.call("slid_tendency(raw_lines)"))
	lines = []; scale = lambda x, y, s: \
		int(x * (1+s)/2 + y * (1-s)/2)
	for a, b in raw_lines:
		# [A] s - scale
		# Xa' = Xa (1+s)/2 + Xb (1-s)/2
		# Ya' = Ya (1+s)/2 + Yb (1-s)/2
		a[0] = scale(a[0], b[0], s)
		a[1] = scale(a[1], b[1], s)
		# [B] s - scale
		# Xb' = Xb (1+s)/2 + Xa (1-s)/2
		# Yb' = Yb (1+s)/2 + Ya (1-s)/2
		b[0] = scale(b[0], a[0], s)
		b[1] = scale(b[1], a[1], s)
		lines += [[a, b]]
	return lines

def laps_intersections(lines):
	"""find all intersections"""
	__lines = [[(a[0], a[1]), (b[0], b[1])] for a, b in lines]
	return isect_segments(__lines)

def laps_cluster(points, max_dist=10):
	"""cluster very similar points"""
	Y = scipy.spatial.distance.pdist(points)
	Z = scipy.cluster.hierarchy.single(Y)
	T = scipy.cluster.hierarchy.fcluster(Z, max_dist, 'distance')
	clusters = collections.defaultdict(list)
	for i in range(len(T)):
		clusters[T[i]].append(points[i])
	clusters = clusters.values()
	clusters = map(lambda arr: (np.mean(np.array(arr)[:,0]),
		                        np.mean(np.array(arr)[:,1])), clusters)
	return list(clusters) # if two points are close, they become one mean point

def laps_detector(img):
	"""determine if that shape is positive"""
	global NC_LAYER

	hashid = str(hash(img.tostring()))

	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	img = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)[1]
	img = cv2.Canny(img, 0, 255)	
	img = cv2.resize(img, (21, 21), interpolation=cv2.INTER_CUBIC)
	
	imgd = img

	X = [np.where(img>int(255/2), 1, 0).ravel()]
	X = X[0].reshape([-1, 21, 21, 1])

	img = cv2.dilate(img, None)
	mask = cv2.copyMakeBorder(img, top=1, bottom=1, left=1, right=1,
		borderType=cv2.BORDER_CONSTANT, value=[255,255,255])
	mask = cv2.bitwise_not(mask); i = 0
	contours, _2 = cv2.findContours(mask,cv2.RETR_EXTERNAL,
				                             cv2.CHAIN_APPROX_NONE)
	
	_c = np.zeros((23,23,3), np.uint8)

	# geometric detector
	for cnt in contours:
		(x,y),radius = cv2.minEnclosingCircle(cnt); x,y=int(x),int(y)
		approx = cv2.approxPolyDP(cnt,0.1*cv2.arcLength(cnt,True),True)
		if len(approx) == 4 and radius < 14:
			cv2.drawContours(_c, [cnt], 0, (0,255,0), 1)
			i += 1
		else:
			cv2.drawContours(_c, [cnt], 0, (0,0,255), 1)
	
	if i == 4: return (True, 1)

	pred = NC_LAPS_MODEL.predict(X)
	a, b = pred[0][0], pred[0][1]
	t = a > b and b < 0.03 and a > 0.975

	# decision
	if t:
		#debug.image(imgd).save("OK" + str(hash(str(imgd))), prefix=False)
		return (True, pred[0])
	else:
		#debug.image(imgd).save("NO" + str(hash(str(imgd))), prefix=False)
		return (False, pred[0])

################################################################################

def LAPS(img, lines, size=10):
	#print(utils.call("LAPS(img, lines)"))

	__points, points = laps_intersections(lines), []
	#debug.image(img).points(__points, size=3).save("laps_in_queue")

	for pt in __points:
		# pixels are in integers
		pt = list(map(int, pt))

		# size of our analysis area
		lx1 = max(0, int(pt[0]-size-1)); lx2 = max(0, int(pt[0]+size))
		ly1 = max(0, int(pt[1]-size)); ly2 = max(0, int(pt[1]+size+1))

		# cropping for detector
		dimg = img[ly1:ly2, lx1:lx2]
		dimg_shape = np.shape(dimg)
		
		# not valid
		if dimg_shape[0] <= 0 or dimg_shape[1] <= 0: continue

		# use neural network
		re_laps = laps_detector(dimg)
		if not re_laps[0]: continue

		# add if okay
		if pt[0] < 0 or pt[1] < 0: continue
		points += [pt]
	points = laps_cluster(points)

	#debug.image(img).points(points, size=5, \
		#color=debug.color()).save("laps_good_points")
	
	return points

def llr_normalize(points): return [[int(a), int(b)] for a, b in points]

def llr_correctness(points, shape):
	__points = []
	for pt in points:
		if pt[0] < 0 or pt[1] < 0 or \
			pt[0] > shape[1] or \
			pt[1] > shape[0]: continue
		__points += [pt]
	return __points

def llr_unique(a):
	indices = sorted(range(len(a)), key=a.__getitem__)
	indices = set(next(it) for k, it in
		itertools.groupby(indices, key=a.__getitem__))
	return [x for i, x in enumerate(a) if i in indices]

def llr_polysort(pts):
	"""sort points clockwise"""
	mlat = sum(x[0] for x in pts) / len(pts)
	mlng = sum(x[1] for x in pts) / len(pts)
	def __sort(x): # main math --> found on MIT site
		return (math.atan2(x[0]-mlat, x[1]-mlng) + \
				2*math.pi)%(2*math.pi)
	pts.sort(key=__sort)
	return pts

def llr_polyscore(cnt, pts, cen, alfa=5, beta=2):
	a = cnt[0]; b = cnt[1]
	c = cnt[2]; d = cnt[3]

	# (1) # za mala powierzchnia
	area = cv2.contourArea(cnt)
	t2 = area < (4 * alfa * alfa) * 5
	if t2: return 0

	gamma = alfa/1.5#alfa**(1/2)
	#print("ALFA", alfa)

	# (2) # za malo punktow
	pco = pyclipper.PyclipperOffset()
	pco.AddPath(cnt, pyclipper.JT_MITER, pyclipper.ET_CLOSEDPOLYGON)
	pcnt = matplotlib.path.Path(pco.Execute(gamma)[0]) # FIXME: alfa/1.5
	wtfs = pcnt.contains_points(pts)
	pts_in = min(np.count_nonzero(wtfs), 49)
	t1 = pts_in < min(len(pts), 49) - 2 * beta - 1
	if t1: return 0

	A = pts_in
	B = area

	# (3)
	# FIXME: punkty za kwadratowosci? (przypadki z L shape)
	nln = lambda l1, x, dx: \
		np.linalg.norm(np.cross(na(l1[1])-na(l1[0]),
								na(l1[0])-na(   x)))/dx
	pcnt_in = []; i = 0
	for pt in wtfs:
		if pt: pcnt_in += [pts[i]]
		i += 1
	def __convex_approx(points, alfa=0.001):
		hull = scipy.spatial.ConvexHull(na(points)).vertices
		cnt = na([points[pt] for pt in hull])
		return cnt
		#approx = cv2.approxPolyDP(cnt,alfa*\
			#	 cv2.arcLength(cnt,True),True)
		#return llr_normalize(itertools.chain(*approx))
	#hull = scipy.spatial.ConvexHull(na(pcnt_in)).vertices
	#cnt_in = na([pcnt_in[pt] for pt in hull])

	cnt_in = __convex_approx(na(pcnt_in))

	points = cnt_in
	x = [p[0] for p in points]          # szukamy punktu
	y = [p[1] for p in points]          # centralnego skupiska
	cen2 = (sum(x) / len(points), \
			sum(y) / len(points))

	G = np.linalg.norm(na(cen)-na(cen2))

	#S = cv2.contourArea(na(cnt_in))
	#if S > B: E += abs(S - B)

	"""
	cnt_in = __convex_approx(na(pcnt_in))
	S = cv2.contourArea(na(cnt_in))
	if S < B: E += abs(S - B)
	cnt_in = __convex_approx(na(list(cnt_in)+list(cnt)))
	S = cv2.contourArea(na(cnt_in))
	if S > B: E += abs(S - B)
	"""

	a = [cnt[0], cnt[1]]
	b = [cnt[1], cnt[2]]
	c = [cnt[2], cnt[3]]
	d = [cnt[3], cnt[0]]
	lns = [a, b, c, d]
	E = 0; F = 0
	for l in lns:
		d = np.linalg.norm(na(l[0])-na(l[1]))
		for p in cnt_in:
			r = nln(l,p,d)
			if r < gamma:
				E += r
				F += 1
	if F == 0: return 0
	E /= F
	# print("PTS_IN", pts_in, "|", "AREA", area, "-->", A/B)
	
	if B == 0 or A == 0: return 0
	
	#C = (E/A+1)**3       # rownosc
	#D = (G/A**(1.5)+1)  # centroid
	#R = (A**4)/(C * (B**2) * D)

	#C = 1+(E/A**2)  # rownosc
	#D = 1+(G/A**2)  # centroid
	#R = (A**4)/(C * (B**2) * D)

	# working
	#C = 1+(E/A**1)  # rownosc
	#D = 1+(G/A**2)  # centroid
	#R = (A**4)/((B**2) * C * D)
	
	C = 1+(E/A)**(1/3)  # rownosc
	D = 1+(G/A)**(1/5)  # centroid
	R = (A**4)/((B**2) * C * D)

	print(R*(10**12), A, "|", B, C, D, "|", E, G)
	
	return R
	#                  R        E        B     A  abs(E-B)
	# 0.0036616950969009555 128126.0 139323.0 41 11197.0
	# 0.00581757739455641   137893.0 145112.5 42 7219.5

################################################################################

# LAPS, SLID

def LLR(img, points, lines):
	#print(utils.call("LLR(img, points, lines)"))
	old = points

	# --- otoczka
	def __convex_approx(points, alfa=0.01):
		hull = scipy.spatial.ConvexHull(na(points)).vertices
		cnt = na([points[pt] for pt in hull])
		approx = cv2.approxPolyDP(cnt,alfa*\
				 cv2.arcLength(cnt,True),True)
		return llr_normalize(itertools.chain(*approx))
	# ---

	# --- geometria
	__cache = {}
	def __dis(a, b):
		idx = hash("__dis" + str(a) + str(b))
		if idx in __cache: return __cache[idx]
		__cache[idx] = np.linalg.norm(na(a)-na(b))
		return __cache[idx]

	nln = lambda l1, x, dx: \
		np.linalg.norm(np.cross(na(l1[1])-na(l1[0]),
								na(l1[0])-na(   x)))/dx
	# ---

	pregroup = [[], []]                   # podzial na 2 grupy (dla ramki)
	S = {}                                # ranking ramek // wraz z wynikiem

	points = llr_correctness(llr_normalize(points), img.shape) # popraw punkty

	# --- clustrowanie
	import sklearn.cluster
	__points = {}; points = llr_polysort(points); __max, __points_max = 0, []
	alfa = math.sqrt(cv2.contourArea(na(points))/49)
	X = sklearn.cluster.DBSCAN(eps=alfa*4).fit(points) # **(1.3)
	for i in range(len(points)): __points[i] = []
	for i in range(len(points)):
		if X.labels_[i] != -1: __points[X.labels_[i]] += [points[i]]
	for i in range(len(points)):
		if len(__points[i]) > __max:
			__max = len(__points[i]); __points_max = __points[i]
	if len(__points) > 0 and len(points) > 49/2: points = __points_max
	print(X.labels_)
	# ---

	# tworzymy zewnetrzny pierscien
	ring = __convex_approx(llr_polysort(points))

	n = len(points); beta = n*(5/100) # beta=n*(100-(skutecznosc LAPS))
	alfa = math.sqrt(cv2.contourArea(na(points))/49) # srednia otoczka siatki

	x = [p[0] for p in points]          # szukamy punktu
	y = [p[1] for p in points]          # centralnego skupiska
	centroid = (sum(x) / len(points), \
			    sum(y) / len(points))

	print(alfa, beta, centroid)

	#        C (x2, y2)        d=(x_1−x_0)^2+(y_1−y_0)^2, t=d_t/d
	#      B (x1, y1)          (x_2,y_2)=(((1−t)x_0+tx_1),((1−t)y_0+ty_1))
	#    .                    t=(x_0-x_2)/(x_0-x_1)
	#  .
	# A (x0, y0)

	def __v(l):
		y_0, x_0 = l[0][0], l[0][1]
		y_1, x_1 = l[1][0], l[1][1]
		
		x_2 = 0;            t=(x_0-x_2)/(x_0-x_1+0.0001)
		a = [int((1-t)*x_0+t*x_1), int((1-t)*y_0+t*y_1)][::-1]

		x_2 = img.shape[0]; t=(x_0-x_2)/(x_0-x_1+0.0001)
		b = [int((1-t)*x_0+t*x_1), int((1-t)*y_0+t*y_1)][::-1]

		poly1 = llr_polysort([[0,0], [0, img.shape[0]], a, b])
		s1 = llr_polyscore(na(poly1), points, centroid, beta=beta, alfa=alfa/2)
		poly2 = llr_polysort([a, b, \
				[img.shape[1],0], [img.shape[1],img.shape[0]]])
		s2 = llr_polyscore(na(poly2), points, centroid, beta=beta, alfa=alfa/2)
		
		return [a, b], s1, s2

	def __h(l):
		x_0, y_0 = l[0][0], l[0][1]
		x_1, y_1 = l[1][0], l[1][1]
		
		x_2 = 0;            t=(x_0-x_2)/(x_0-x_1+0.0001)
		a = [int((1-t)*x_0+t*x_1), int((1-t)*y_0+t*y_1)]

		x_2 = img.shape[1]; t=(x_0-x_2)/(x_0-x_1+0.0001)
		b = [int((1-t)*x_0+t*x_1), int((1-t)*y_0+t*y_1)]

		poly1 = llr_polysort([[0,0], [img.shape[1], 0], a, b])
		s1 = llr_polyscore(na(poly1), points, centroid, beta=beta, alfa=alfa/2)
		poly2 = llr_polysort([a, b, \
				[0, img.shape[0]], [img.shape[1], img.shape[0]]])
		s2 = llr_polyscore(na(poly2), points, centroid, beta=beta, alfa=alfa/2)

		return [a, b], s1, s2

	for l in lines: # bedziemy wszystkie przegladac
		for p in points: # odrzucamy linie ktore nie pasuja
			# (1) linia przechodzi blisko dobrego punktu
			t1 = nln(l, p, __dis(*l)) < alfa
			# (2) linia przechodzi przez srodek skupiska
			t2 = nln(l, centroid, __dis(*l)) > alfa * 2.5 # 3
			# (3) linia nalezy do pierscienia
			# t3 = True if p in ring else False
			if t1 and t2:
			#if (t1 and t2) or (t1 and t3 and t2): # [1 and 2] or [1 and 3 and 2]
				tx, ty = l[0][0]-l[1][0], l[0][1]-l[1][1]
				if abs(tx) < abs(ty): ll, s1, s2 = __v(l); o = 0
				else:                 ll, s1, s2 = __h(l); o = 1
				if s1 == 0 and s2 == 0: continue
				pregroup[o] += [ll]

	pregroup[0] = llr_unique(pregroup[0])
	pregroup[1] = llr_unique(pregroup[1])

	#from laps import laps_intersections
#	debug.image(img) \
	#	.lines(lines, color=(0,0,255)) \
#		.points(laps_intersections(lines), color=(255,0,0), size=2) \
#	.save("llr_debug_1")

	#debug.image(img) \
	#	.points(laps_intersections(lines), color=(0,0,255), size=2) \
	#	.points(old, color=(0,255,0)) \
#	.save("llr_debug_2")

	#debug.image(img) \
	#	.lines(lines, color=(0,0,255)) \
		#.points(points, color=(0,0,255)) \
	#	.points(ring, color=(0,255,0)) \
	#	.points([centroid], color=(255,0,0)) \
#	.save("llr_debug")
	
	#debug.image(img) \
		#.lines(pregroup[0], color=(0,0,255)) \
		#.lines(pregroup[1], color=(255,0,0)) \
	#.save("llr_pregroups")
	
	print("---------------------")
	for v in itertools.combinations(pregroup[0], 2):            # poziome
		for h in itertools.combinations(pregroup[1], 2):        # pionowe
			poly = laps_intersections([v[0], v[1], h[0], h[1]]) # przeciecia
			poly = llr_correctness(poly, img.shape)             # w obrazku
			if len(poly) != 4: continue                         # jesl. nie ma
			poly = na(llr_polysort(llr_normalize(poly)))        # sortuj
			if not cv2.isContourConvex(poly): continue          # wypukly?
			S[-llr_polyscore(poly, points, centroid, \
				beta=beta, alfa=alfa/2)] = poly                 # dodaj

	S = collections.OrderedDict(sorted(S.items()))              # max
	K = next(iter(S))
	print("key --", K)
	four_points = llr_normalize(S[K])               # score

	# XXX: pomijanie warst, lub ich wybor? (jesli mamy juz okay)
	# XXX: wycinanie pod sam koniec? (modul wylicznia ile warstw potrzebnych)

	print("POINTS:", len(points))
	print("LINES:", len(lines))

	#debug.image(img).points(four_points).save("llr_four_points")

	#debug.image(img) \
		#.points(points, color=(0,255,0)) \
 		#.points(four_points, color=(0,0,255)) \
		#.points([centroid], color=(255,0,0)) \
		#.lines([[four_points[0], four_points[1]], [four_points[1], four_points[2]], \
		        #[four_points[2], four_points[3]], [four_points[3], four_points[0]]], \
				#color=(255,255,255)) \
	#.save("llr_debug_3")

	return four_points

def llr_pad(four_points, img):
  #print(utils.call("llr_pad(four_points)"))
  pco = pyclipper.PyclipperOffset()
  pco.AddPath(four_points, pyclipper.JT_MITER, pyclipper.ET_CLOSEDPOLYGON)

  padded = pco.Execute(60)[0]
  #debug.image(img) \
    #.points(four_points, color=(0,0,255)) \
    #.points(padded, color=(0,255,0)) \
    #.lines([[four_points[0], four_points[1]], [four_points[1], four_points[2]], \
            #[four_points[2], four_points[3]], [four_points[3], four_points[0]]], \
        #color=(255,255,255)) \
    #.lines([[padded[0], padded[1]], [padded[1], padded[2]], \
            #[padded[2], padded[3]], [padded[3], padded[0]]], \
        #color=(255,255,255)) \
  #.save("llr_final_pad")

  return pco.Execute(60)[0] # 60,70/75 is best (with buffer/for debug purpose)

def warpImage(img, corners, dp):

    height, width, _ = np.shape(img)
    corners = np.float32([ ( min( width, corners[0][0] + dp) , min(corners[0][1]+ dp, height )) , 
                          ( max(corners[1][0] -dp,0 ) , min(corners[1][1] + dp, height) ) , 
                            ( max(corners[1][0] -dp,0 ) , max(corners[1][0] -dp,0 ) ), 
                          ( min( width, corners[0][0] + dp) , max(corners[1][0] -dp,0 ) ) ])
    target = np.float32([(width,height),(0,height), (0,0),(width,0)])

    M = cv2.getPerspectiveTransform(corners, target)
    	
    out = cv2.warpPerspective(img,M,(width, height),flags=cv2.INTER_LINEAR)
    return out

def makeList(four_points):
  corners = []
  for i in range(0, len(four_points)):
    corners.append((four_points[i][0] , four_points[i][1] ))

  return corners

def image_scale(pts, scale):
	"""scale to original image size"""
	def __loop(x, y): return [x[0] * y, x[1] * y]
	return list(map(functools.partial(__loop, y=1/scale), pts))

def image_resize(img, height=500):
	"""resize image to same normalized area (height**2)"""
	pixels = height * height; shape = list(np.shape(img))
	scale = math.sqrt(float(pixels)/float(shape[0]*shape[1]))
	shape[0] *= scale; shape[1] *= scale
	img = cv2.resize(img, (int(shape[1]), int(shape[0])))
	img_shape = np.shape(img)
	return img, img_shape, scale

def image_transform(img, points, dx, square_length=150):
  """crop original image using perspective warp"""
  board_length = square_length * 8
  def __dis(a, b): return np.linalg.norm(na(a)-na(b))
  def __shi(seq, n=0): return seq[-(n % len(seq)):] + seq[:-(n % len(seq))]
  best_idx, best_val = 0, 10**6
  for idx, val in enumerate(points):
    val = __dis(val, [0, 0])
    if val < best_val:
      best_idx, best_val = idx, val
  pts1 = np.float32(__shi(points, 4 - best_idx))
  pts2 = np.float32([[dx[0], dx[2]], [board_length - dx[1], dx[2]], \
      [board_length - dx[1], board_length - dx[3]], [dx[0], board_length - dx[3]]])
  M = cv2.getPerspectiveTransform(pts1, pts2)

  """
  for i in range(0,4):
    print(M.dot(np.array([pts1[i][0], pts1[i][1], 1])), np.float32([pts2[i][0], pts2[i][1], 1]) )
  """

  W = cv2.warpPerspective(img, M, (board_length, board_length))

  return W, M

def crop(img, pts, dx):
		"""crop using 4 points transform"""
		pts_orig = image_scale(pts, 1)
		img_crop = image_transform(img, pts_orig, dx)
		return img_crop

