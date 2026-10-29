"""Native tree data structure for HERACLES.

Provides ``Tree``, ``Node``, ``Taxon``, ``TaxonNamespace``, an iterative
Newick parser/writer, a pure-Python neighbour joining and the bipartition
primitives behind the Robinson-Foulds distance. Method names follow dendropy,
so code written for dendropy trees mostly works unchanged.

Reading, writing, rooting and comparing trees is in :mod:`heracles.trees.utils`;
start there unless you are working on the data structure itself.
"""

from __future__ import annotations

import numpy as np


__all__ = [
    'quote_label',
    'Taxon',
    'TaxonNamespace',
    'Node',
    'Tree',
    'native_nj',
    'symmetric_difference',
]


# Taxon / TaxonNamespace

class Taxon:
    """A leaf taxon, identified by its label."""
    __slots__ = ("label",)

    def __init__(self, label=None):
        self.label = label

    def __repr__(self):
        return f"Taxon({self.label!r})"


class TaxonNamespace:
    """Ordered collection of :class:`Taxon` objects (leaf-label registry)."""

    def __init__(self, labels=None):
        self._taxa: list[Taxon] = []
        self._map: dict[str, Taxon] = {}
        if labels is not None:
            for lab in labels:
                self.require_taxon(str(lab))

    def require_taxon(self, label: str) -> Taxon:
        """Return the taxon with *label*, creating and registering it if absent."""
        t = self._map.get(label)
        if t is not None:
            return t
        t = Taxon(label=label)
        self._taxa.append(t)
        self._map[label] = t
        return t

    def labels(self) -> list[str]:
        """Taxon labels in registration order."""
        return [t.label for t in self._taxa]

    def __iter__(self):
        return iter(self._taxa)

    def __len__(self):
        return len(self._taxa)

    def __contains__(self, item):
        if isinstance(item, Taxon):
            return item.label in self._map
        return str(item) in self._map


# Edge / Node

class _Edge:
    """The edge above a node; holds only its length."""
    __slots__ = ("length",)

    def __init__(self, length=None):
        self.length = length


class Node:
    """A tree node; leaves carry a ``taxon``, internal nodes an optional ``label``."""
    __slots__ = ("taxon", "parent_node", "_children", "edge", "annotations", "label")

    def __init__(self):
        self.taxon: Taxon | None = None
        self.parent_node: Node | None = None
        self._children: list[Node] = []
        self.edge = _Edge()
        self.annotations: dict = {}
        self.label: str | None = None

    @property
    def edge_length(self):
        """Length of the edge above this node (None if unset)."""
        return self.edge.length

    @edge_length.setter
    def edge_length(self, value):
        self.edge.length = value

    def child_nodes(self) -> list[Node]:
        """List of child nodes (a copy)."""
        return list(self._children)

    def is_leaf(self) -> bool:
        """True if the node has no children."""
        return len(self._children) == 0

    def add_child(self, node: Node):
        """Append *node* as the last child and set its parent."""
        node.parent_node = self
        self._children.append(node)

    def remove_child(self, node: Node):
        """Detach child *node* and clear its parent."""
        self._children.remove(node)
        node.parent_node = None

    def leaf_iter(self):
        """Iterate over the leaves below this node, left to right."""
        stack = [self]
        while stack:
            node = stack.pop()
            if node.is_leaf():
                yield node
            else:
                stack.extend(reversed(node._children))

    def __repr__(self):
        lab = self.taxon.label if self.taxon else "<internal>"
        return f"Node({lab!r})"


# Tree

class Tree:
    """A tree rooted at ``seed_node``, with leaf taxa in ``taxon_namespace``."""
    def __init__(self, taxon_namespace=None):
        self.seed_node = Node()
        self.taxon_namespace: TaxonNamespace = taxon_namespace if taxon_namespace is not None else TaxonNamespace()
        self.is_rooted = True

    # Newick input and output

    @classmethod
    def get(cls, *, data=None, schema="newick", **kwargs):
        """Parse a Newick string (*data*); accepts ``taxon_namespace`` and ``rooting``."""
        if schema != "newick":
            raise ValueError(f"unsupported schema: {schema!r}")
        return _parse_newick(data, **kwargs)

    def as_string(self, *, schema="newick", **kwargs) -> str:
        """Newick string of the tree, ending in ``;`` and a newline."""
        if schema != "newick":
            raise ValueError(f"unsupported schema: {schema!r}")
        return _write_newick(self, **kwargs)

    # iterators

    def preorder_node_iter(self):
        """Iterate over nodes in preorder (parents before children)."""
        stack = [self.seed_node]
        while stack:
            node = stack.pop()
            yield node
            for i in range(len(node._children) - 1, -1, -1):
                stack.append(node._children[i])

    def postorder_node_iter(self):
        """Iterate over nodes in postorder (children before parents)."""
        stack1 = [self.seed_node]
        stack2: list[Node] = []
        while stack1:
            node = stack1.pop()
            stack2.append(node)
            stack1.extend(node._children)
        return reversed(stack2)

    def leaf_node_iter(self):
        """Iterate over leaves in preorder."""
        for node in self.preorder_node_iter():
            if node.is_leaf():
                yield node

    def leaf_nodes(self) -> list[Node]:
        """List of leaves in preorder."""
        return list(self.leaf_node_iter())

    def preorder_edge_iter(self):
        """Iterate over the edges of non-root nodes in preorder."""
        for node in self.preorder_node_iter():
            if node.parent_node is not None:
                yield node.edge

    def preorder_internal_node_iter(self):
        """Iterate over internal nodes in preorder."""
        for node in self.preorder_node_iter():
            if not node.is_leaf():
                yield node

    def internal_edges(self, exclude_seed_edge=False) -> list[_Edge]:
        """Edges above internal non-root nodes."""
        out: list[_Edge] = []
        for node in self.preorder_node_iter():
            if node.is_leaf():
                continue
            if node is self.seed_node and exclude_seed_edge:
                continue
            if node.parent_node is not None:
                out.append(node.edge)
        return out

    def find_node_with_taxon_label(self, label) -> Node | None:
        """First node whose taxon has *label*, or None."""
        for node in self.preorder_node_iter():
            if node.taxon is not None and node.taxon.label == label:
                return node
        return None

    def taxon_namespace_scoped_copy(self) -> Tree:
        """Deep copy of the nodes that shares this tree's taxon namespace."""
        new = Tree(taxon_namespace=self.taxon_namespace)
        new.is_rooted = self.is_rooted
        node_map: dict[int, Node] = {}
        for old in self.preorder_node_iter():
            nd = Node()
            nd.taxon = old.taxon
            nd.label = old.label
            nd.edge.length = old.edge.length
            nd.annotations = dict(old.annotations)
            node_map[id(old)] = nd
            if old.parent_node is not None:
                node_map[id(old.parent_node)].add_child(nd)
        new.seed_node = node_map[id(self.seed_node)]
        return new

    # topology operations

    def to_outgroup_position(self, outgroup_node: Node, **_kwargs):
        """Reroot the tree so *outgroup_node* is a direct child of the root."""
        if outgroup_node is self.seed_node:
            return
        path: list[Node] = []
        n = outgroup_node
        while n is not None:
            path.append(n)
            n = n.parent_node
        for i in range(len(path) - 1, 1, -1):
            parent, child = path[i], path[i - 1]
            parent.remove_child(child)
            child.add_child(parent)
            parent.edge.length = child.edge.length
        new_root = path[1]
        new_root.parent_node = None
        new_root.edge.length = None
        self.seed_node = new_root

    def prune_taxa_with_labels(self, labels, suppress_unifurcations=True):
        """Remove the leaves with these labels, in place."""
        label_set = set(str(lab) for lab in labels)
        for leaf in list(self.leaf_node_iter()):
            if leaf.taxon is not None and leaf.taxon.label in label_set:
                if leaf.parent_node is not None:
                    leaf.parent_node.remove_child(leaf)
        if suppress_unifurcations:
            self._suppress_unifurcations()

    def prune_taxa(self, taxa, suppress_unifurcations=True):
        """Remove the leaves of these taxa, in place."""
        self.prune_taxa_with_labels([t.label for t in taxa],
                                    suppress_unifurcations=suppress_unifurcations)

    def _suppress_unifurcations(self):
        """Remove single-child nodes, merging their edge lengths."""
        for node in list(self.postorder_node_iter()):
            if len(node._children) != 1:
                continue
            child = node._children[0]
            child.edge.length = (child.edge.length or 0.0) + (node.edge.length or 0.0)
            parent = node.parent_node
            if parent is not None:
                idx = parent._children.index(node)
                parent._children[idx] = child
                child.parent_node = parent
            else:
                child.parent_node = None
                child.edge.length = None
                self.seed_node = child

    def deroot(self):
        """Collapse a basal bifurcation into a trifurcation (unroot).

        The internal child of the root is dissolved and its children attached
        to the root; the other child keeps the combined length of the two root
        edges.  A root with two leaf children (a two-taxon tree) is left as is.
        """
        root = self.seed_node
        if len(root._children) != 2:
            self.is_rooted = False
            return
        keep, dissolve = root._children
        if dissolve.is_leaf():
            if keep.is_leaf():
                self.is_rooted = False
                return
            keep, dissolve = dissolve, keep
        combined = (keep.edge.length or 0.0) + (dissolve.edge.length or 0.0)
        keep.edge.length = combined
        root.remove_child(dissolve)
        for gc in list(dissolve._children):
            dissolve.remove_child(gc)
            root.add_child(gc)
        self.is_rooted = False

    def suppress_unifurcations(self):
        """Remove single-child nodes in place, merging their edge lengths."""
        self._suppress_unifurcations()

    def calc_node_root_distances(self, return_leaf_distances_only=False):
        """Cumulative branch-length distance from each node to the root."""
        rd = {}
        dists = []
        for node in self.preorder_node_iter():
            if node.parent_node is None:
                rd[id(node)] = 0.0
            else:
                rd[id(node)] = rd[id(node.parent_node)] + (node.edge.length or 0.0)
            if return_leaf_distances_only and not node.is_leaf():
                continue
            dists.append(rd[id(node)])
        return dists

    def update_bipartitions(self, **_kwargs):
        """Does nothing; accepted so code written for dendropy trees runs unchanged."""
        pass


# Newick parser (iterative, no recursion limit)

def _parse_newick(data: str, **kwargs) -> Tree:
    """Iterative Newick parser behind :meth:`Tree.get`."""
    taxon_namespace: TaxonNamespace | None = kwargs.get("taxon_namespace")
    rooting = kwargs.get("rooting")

    data = data.strip()

    is_rooted = True
    while data.startswith("["):
        end = data.index("]") + 1
        tag = data[1:end - 1].strip()
        if tag == "&R":
            is_rooted = True
        elif tag == "&U":
            is_rooted = False
        data = data[end:].lstrip()

    if rooting == "force-rooted":
        is_rooted = True
    elif rooting == "force-unrooted":
        is_rooted = False

    while data.endswith(";"):
        data = data[:-1].rstrip()

    tns = taxon_namespace if taxon_namespace is not None else TaxonNamespace()
    tree = Tree(taxon_namespace=tns)
    tree.is_rooted = is_rooted

    if not data:
        return tree

    # Tokenising helpers; pos is a one-element list so the closures can advance it.

    pos = [0]
    n = len(data)

    def _skip():
        p = pos[0]
        while p < n:
            c = data[p]
            if c in " \t\n\r":
                p += 1
            elif c == "[":
                d = 1
                p += 1
                while p < n and d:
                    if data[p] == "[":
                        d += 1
                    elif data[p] == "]":
                        d -= 1
                    p += 1
            else:
                break
        pos[0] = p

    def _read_label():
        _skip()
        p = pos[0]
        if p >= n or data[p] in "(),:;":
            return None
        if data[p] == "'":
            p += 1
            parts: list[str] = []
            while p < n:
                if data[p] == "'" and p + 1 < n and data[p + 1] == "'":
                    parts.append("'")
                    p += 2
                elif data[p] == "'":
                    p += 1
                    break
                else:
                    parts.append(data[p])
                    p += 1
            pos[0] = p
            return "".join(parts)
        start = p
        while p < n and data[p] not in "(),:;[] \t\n\r'":
            p += 1
        pos[0] = p
        return data[start:p] if p > start else None

    def _read_length():
        _skip()
        p = pos[0]
        if p < n and data[p] == ":":
            p += 1
            while p < n and data[p] in " \t\n\r":
                p += 1
            start = p
            while p < n and data[p] not in "(),:;[] \t\n\r":
                p += 1
            pos[0] = p
            return float(data[start:p])
        return None

    root = tree.seed_node
    _skip()

    if pos[0] >= n:
        return tree

    if data[pos[0]] != "(":
        label = _read_label()
        length = _read_length()
        if label:
            root.taxon = tns.require_taxon(label)
        root.edge.length = length
        return tree

    pos[0] += 1
    stack = [root]

    while stack and pos[0] < n:
        _skip()
        if pos[0] >= n:
            break
        c = data[pos[0]]

        if c == "(":
            pos[0] += 1
            child = Node()
            stack[-1].add_child(child)
            stack.append(child)
        elif c == ")":
            pos[0] += 1
            node = stack.pop()
            _skip()
            label = _read_label()
            length = _read_length()
            if label:
                if node.is_leaf():
                    node.taxon = tns.require_taxon(label)
                else:
                    node.label = label
            node.edge.length = length
        elif c == ",":
            pos[0] += 1
        elif c == ";":
            break
        else:
            child = Node()
            label = _read_label()
            length = _read_length()
            if label:
                child.taxon = tns.require_taxon(label)
            child.edge.length = length
            stack[-1].add_child(child)

    return tree


# Newick writer (iterative)

_NEWICK_SPECIAL = frozenset("(),:;[] \t\n\r'")


def quote_label(label: str) -> str:
    """Quote a label for Newick when it contains a character Newick would mangle."""
    if any(c in _NEWICK_SPECIAL for c in label):
        return "'" + label.replace("'", "''") + "'"
    return label


def _write_newick(tree: Tree, *, suppress_rooting=False,
                  suppress_internal_node_labels=False,
                  suppress_leaf_taxon_labels=False,
                  suppress_internal_taxon_labels=False,
                  **_kw) -> str:
    """Iterative Newick writer behind :meth:`Tree.as_string`."""
    buf: list[str] = []
    if not suppress_rooting and tree.is_rooted:
        buf.append("[&R] ")

    stack: list = [tree.seed_node]
    while stack:
        item = stack.pop()
        if isinstance(item, str):
            buf.append(item)
            continue
        node: Node = item
        if node.is_leaf():
            if not suppress_leaf_taxon_labels and node.taxon is not None:
                buf.append(quote_label(node.taxon.label))
            if node.edge.length is not None:
                buf.append(":")
                buf.append(str(node.edge.length))
        else:
            if node.edge.length is not None:
                stack.append(":" + str(node.edge.length))
            if not suppress_internal_node_labels:
                if node.taxon is not None:
                    stack.append(quote_label(node.taxon.label))
                elif node.label is not None:
                    stack.append(quote_label(str(node.label)))
            stack.append(")")
            children = node._children
            for i in range(len(children) - 1, -1, -1):
                stack.append(children[i])
                if i > 0:
                    stack.append(",")
            stack.append("(")

    buf.append(";\n")
    return "".join(buf)


# Native NJ (pure-Python fallback when RapidNJ is not available)

def native_nj(dists, labels, taxon_namespace=None):
    """Neighbour Joining from a distance matrix.  Pure Python, O(n^3) time.

    Returns an unrooted :class:`Tree` whose leaves are registered in
    *taxon_namespace* (a new namespace when ``None``).
    """
    tns = taxon_namespace if taxon_namespace is not None else TaxonNamespace()
    n = len(labels)
    if n == 0:
        return Tree(taxon_namespace=tns)
    if n == 1:
        tree = Tree(taxon_namespace=tns)
        tree.seed_node.taxon = tns.require_taxon(str(labels[0]))
        tree.is_rooted = False
        return tree
    if n == 2:
        tree = Tree(taxon_namespace=tns)
        tree.is_rooted = False
        a, b = Node(), Node()
        a.taxon = tns.require_taxon(str(labels[0]))
        b.taxon = tns.require_taxon(str(labels[1]))
        d = float(dists[0, 1])
        a.edge.length = d / 2.0
        b.edge.length = d / 2.0
        tree.seed_node.add_child(a)
        tree.seed_node.add_child(b)
        return tree

    D = np.array(dists, dtype=np.float64)
    active = list(range(n))
    nodes: list[Node] = []
    for lab in labels:
        nd = Node()
        nd.taxon = tns.require_taxon(str(lab))
        nodes.append(nd)

    while len(active) > 2:
        m = len(active)
        r = np.zeros(len(nodes) + 1, dtype=np.float64)
        for a_pos in range(m):
            s = 0.0
            ai = active[a_pos]
            for b_pos in range(m):
                if b_pos != a_pos:
                    s += D[ai, active[b_pos]]
            r[ai] = s

        best_q = np.inf
        best_a = best_b = -1
        for a_pos in range(m):
            ai = active[a_pos]
            for b_pos in range(a_pos + 1, m):
                bi = active[b_pos]
                q = (m - 2) * D[ai, bi] - r[ai] - r[bi]
                if q < best_q:
                    best_q = q
                    best_a, best_b = a_pos, b_pos

        i_idx = active[best_a]
        j_idx = active[best_b]

        d_ij = D[i_idx, j_idx]
        li = d_ij / 2.0 + (r[i_idx] - r[j_idx]) / (2.0 * (m - 2))
        lj = d_ij - li

        new_node = Node()
        nodes[i_idx].edge.length = li
        nodes[j_idx].edge.length = lj
        new_node.add_child(nodes[i_idx])
        new_node.add_child(nodes[j_idx])

        new_idx = len(nodes)
        nodes.append(new_node)

        old_n = D.shape[0]
        D_new = np.zeros((old_n + 1, old_n + 1), dtype=np.float64)
        D_new[:old_n, :old_n] = D
        for k_pos in range(m):
            k = active[k_pos]
            if k != i_idx and k != j_idx:
                d_new = (D[i_idx, k] + D[j_idx, k] - d_ij) / 2.0
                D_new[new_idx, k] = d_new
                D_new[k, new_idx] = d_new
        D = D_new

        active = [x for x in active if x != i_idx and x != j_idx]
        active.append(new_idx)

    i_f, j_f = active
    d_final = D[i_f, j_f]
    tree = Tree(taxon_namespace=tns)
    tree.is_rooted = False
    nodes[i_f].edge.length = d_final / 2.0
    nodes[j_f].edge.length = d_final / 2.0
    tree.seed_node.add_child(nodes[i_f])
    tree.seed_node.add_child(nodes[j_f])
    return tree


# Robinson-Foulds distance (bipartition-based)

def _bipartitions(tree: Tree) -> set[frozenset]:
    """Set of bipartitions (each a frozenset of two frozensets of leaf labels)."""
    leaf_sets: dict[int, frozenset[str]] = {}
    all_labels: set[str] = set()
    for node in tree.postorder_node_iter():
        if node.is_leaf():
            lab = node.taxon.label if node.taxon else ""
            leaf_sets[id(node)] = frozenset([lab])
            all_labels.add(lab)
        else:
            below: frozenset[str] = frozenset()
            for ch in node._children:
                below = below | leaf_sets[id(ch)]
            leaf_sets[id(node)] = below

    all_frozen = frozenset(all_labels)
    biparts: set[frozenset] = set()
    for node in tree.preorder_node_iter():
        if node.parent_node is None or node.is_leaf():
            continue
        below = leaf_sets[id(node)]
        complement = all_frozen - below
        if below and complement:
            biparts.add(frozenset((below, complement)))
    return biparts


def symmetric_difference(tree1: Tree, tree2: Tree) -> int:
    """Splits in one tree but not the other (unrooted Robinson-Foulds count, not halved)."""
    return len(_bipartitions(tree1) ^ _bipartitions(tree2))


