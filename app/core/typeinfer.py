"""Lightweight type inference for Python method references.

Name matching alone links every `.send(...)` in a repo to every class's `send`
method. This module infers, for each `receiver.name` access, what the receiver
is, using only what's written in the code:

- `self` / `cls` inside a class, and `super()` (the enclosing class's bases)
- parameter and variable annotations (`adapter: HTTPAdapter`, `x: Foo | None`)
- constructor calls (`s = Session()`), and `with Foo() as f`
- return annotations of repo functions and methods (`r = self.get_adapter(url)`)
- class attribute types (`self.adapter = HTTPAdapter()`, `conn: HTTPConnection`)

Each receiver resolves to a set of repo class names, to EXTERNAL (a type from
the standard library or a dependency, e.g. `IO[str]` or `str`), or to nothing
(unknown). Callers of `X.m` are then files with a reference whose receiver can
be X (or a class that can dispatch to X.m), plus unknown receivers as a name
fallback, but never receivers known to be something else.

It's flow-insensitive and deliberately modest: no data flow through
containers, no inference through untyped function returns. Everything
unresolved stays "unknown", falling back to name matching rather than
dropping a possible caller.
"""
import re
from dataclasses import dataclass, field
from typing import Iterable, Optional

EXTERNAL = ""  # receiver type known, but not a class defined in this repo
UNKNOWN = "?"  # receiver type couldn't be inferred: callers fall back to name matching

# Wrappers whose arguments name the actual type.
_UNWRAP = {"Optional", "Union", "Annotated", "Final", "ClassVar", "Type", "type"}
# Builtins that call a data-model method on their first argument.
_BUILTIN_PROTOCOLS = {"len": "__len__", "iter": "__iter__", "bool": "__bool__", "repr": "__repr__",
                      "str": "__str__", "hash": "__hash__", "next": "__next__", "copy": "__copy__"}

# Types that say nothing useful about the receiver.
_UNINFORMATIVE = {"Any", "object", "Self", "None", "TypeVar", "T", "F"}


@dataclass
class ClassFacts:
    name: str
    file_path: str
    bases: list[str] = field(default_factory=list)
    methods: set[str] = field(default_factory=set)
    attr_annotations: dict[str, list[str]] = field(default_factory=dict)  # attr → raw annotation / ctor texts


@dataclass
class FileFacts:
    classes: list[ClassFacts] = field(default_factory=list)
    # qualified function name ("Session.get_adapter", "request") → raw return annotation
    returns: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class AttrRef:
    name: str        # the attribute/method name accessed
    receiver: str    # a repo class name, EXTERNAL, or UNKNOWN
    line: int


@dataclass
class TypeIndex:
    """Repo-wide facts from pass 1, used to resolve receivers in pass 2."""
    classes: dict[str, list[ClassFacts]] = field(default_factory=dict)
    returns_by_qual: dict[str, str] = field(default_factory=dict)
    returns_by_name: dict[str, list[str]] = field(default_factory=dict)

    @classmethod
    def build(cls, facts: Iterable[FileFacts]) -> "TypeIndex":
        index = cls()
        for ff in facts:
            for c in ff.classes:
                index.classes.setdefault(c.name, []).append(c)
            for qual, ann in ff.returns.items():
                index.returns_by_qual[qual] = ann
                index.returns_by_name.setdefault(qual.rsplit(".", 1)[-1], []).append(ann)
        return index

    def is_class(self, name: str) -> bool:
        return name in self.classes

    def bases(self, name: str) -> list[str]:
        return [b for c in self.classes.get(name, []) for b in c.bases]

    def mro(self, name: str) -> list[str]:
        """The class and its repo ancestors, nearest first (approximate MRO)."""
        seen, order, queue = set(), [], [name]
        while queue:
            n = queue.pop(0)
            if n in seen or not self.is_class(n):
                continue
            seen.add(n)
            order.append(n)
            queue.extend(self.bases(n))
        return order

    def defines(self, cls_name: str, method: str) -> bool:
        return any(method in c.methods for c in self.classes.get(cls_name, []))

    def attr_types(self, cls_name: str, attr: str) -> Optional[set[str]]:
        for c in self.mro(cls_name):
            for facts in self.classes.get(c, []):
                if attr in facts.attr_annotations:
                    out: set[str] = set()
                    for text in facts.attr_annotations[attr]:
                        types = self.resolve_attr_text(text, c)
                        if types is None:
                            return None  # one unknown assignment makes the attribute unknown
                        out |= types
                    return out or None
        return None

    def resolve_attr_text(self, text: str, self_class: Optional[str] = None) -> Optional[set[str]]:
        """An attribute's recorded type: an annotation, or `__ctor__:Name` for
        `self.x = Name(...)` (a repo class, or a function's return annotation)."""
        if text.startswith("__ctor__:"):
            name = text[len("__ctor__:"):].rsplit(".", 1)[-1]
            if self.is_class(name):
                return {name}
            anns = self.returns_by_name.get(name)
            if anns and len(anns) == 1:
                return self.resolve_annotation(anns[0], self_class)
            return None
        return self.resolve_annotation(text, self_class)

    def method_return(self, cls_name: str, method: str) -> Optional[set[str]]:
        for c in self.mro(cls_name):
            ann = self.returns_by_qual.get(f"{c}.{method}")
            if ann is not None:
                return self.resolve_annotation(ann, cls_name)
        return None

    def resolve_annotation(self, text: str, self_class: Optional[str] = None) -> Optional[set[str]]:
        """Annotation text → repo class names, {EXTERNAL}, or None (unknown)."""
        text = text.strip().strip("'\"").strip()
        if not text:
            return None
        parts = _split_top(text, "|")
        if len(parts) > 1:
            return _union(self.resolve_annotation(p, self_class) for p in parts)
        if not parts:
            return None  # just None
        text = parts[0]  # "Foo | None" → "Foo"
        m = re.match(r"^([\w.]+)\s*\[(.*)\]$", text, re.S)
        if m:
            head = m.group(1).rsplit(".", 1)[-1]
            if head in _UNWRAP:
                return _union(self.resolve_annotation(a, self_class) for a in _split_top(m.group(2), ","))
            text = m.group(1)  # generic like CallbackDict[str, Any]: the receiver is the head
        name = text.rsplit(".", 1)[-1]
        if name in ("Self",) and self_class:
            return {self_class}
        if name in _UNINFORMATIVE:
            return None
        if self.is_class(name):
            return {name}
        return {EXTERNAL}


def _split_top(text: str, sep: str) -> list[str]:
    """Split on `sep` outside brackets."""
    parts, depth, cur = [], 0, []
    for ch in text:
        if ch in "[(":
            depth += 1
        elif ch in "])":
            depth -= 1
        if ch == sep and depth == 0:
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    parts.append("".join(cur))
    return [p.strip() for p in parts if p.strip() and p.strip() != "None"]


def _union(results: Iterable[Optional[set[str]]]) -> Optional[set[str]]:
    out: set[str] = set()
    for r in results:
        if r is None:
            return None  # any unknown part makes the whole unknown
        out |= r
    return out or None


def _text(node, src: bytes) -> str:
    return src[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


# ── Pass 1: facts ─────────────────────────────────────────────────────────────

def collect_facts(root, src: bytes, file_path: str) -> FileFacts:
    facts = FileFacts()

    def visit(node, scope: tuple, cls: Optional[ClassFacts]):
        for child in node.children:
            if child.type == "decorated_definition":
                visit(child, scope, cls)
            elif child.type == "class_definition":
                name_node = child.child_by_field_name("name")
                if name_node is None:
                    continue
                cf = ClassFacts(name=_text(name_node, src), file_path=file_path)
                supers = child.child_by_field_name("superclasses")
                if supers is not None:
                    for arg in supers.named_children:
                        if arg.type in ("identifier", "attribute"):
                            cf.bases.append(_text(arg, src).rsplit(".", 1)[-1])
                        elif arg.type == "subscript":  # Generic[T], Base[str]
                            value = arg.child_by_field_name("value")
                            if value is not None:
                                cf.bases.append(_text(value, src).rsplit(".", 1)[-1])
                facts.classes.append(cf)
                body = child.child_by_field_name("body")
                if body is not None:
                    for stmt in body.children:  # class-level `attr: T` / `attr = Ctor()`
                        _collect_assignment(stmt, src, cf, class_level=True)
                    visit(body, scope + (cf.name,), cf)
            elif child.type == "function_definition":
                name_node = child.child_by_field_name("name")
                if name_node is None:
                    continue
                fname = _text(name_node, src)
                qual = ".".join(scope + (fname,))
                ret = child.child_by_field_name("return_type")
                if ret is not None:
                    facts.returns[qual] = _text(ret, src)
                if cls is not None and len(scope) and scope[-1] == cls.name:
                    cls.methods.add(fname)
                    body = child.child_by_field_name("body")
                    if body is not None:
                        _collect_self_assignments(body, src, cls)
                # nested functions aren't attributes of anything; don't descend for classes
            else:
                visit(child, scope, cls)

    visit(root, (), None)
    return facts


def _collect_assignment(stmt, src: bytes, cf: ClassFacts, class_level: bool) -> None:
    if stmt.type != "expression_statement" or not stmt.named_children:
        return
    node = stmt.named_children[0]
    if node.type != "assignment":
        return
    left, right, ann = (node.child_by_field_name(f) for f in ("left", "right", "type"))
    if left is None:
        return
    if class_level and left.type == "identifier":
        attr = _text(left, src)
    elif not class_level and left.type == "attribute":
        obj = left.child_by_field_name("object")
        if obj is None or _text(obj, src) != "self":
            return
        attr = _text(left.child_by_field_name("attribute"), src)
    else:
        return
    if ann is not None:
        cf.attr_annotations.setdefault(attr, []).append(_text(ann, src))
    elif right is not None and right.type == "call":
        fn = right.child_by_field_name("function")
        if fn is not None and fn.type in ("identifier", "attribute"):
            cf.attr_annotations.setdefault(attr, []).append("__ctor__:" + _text(fn, src))


def _collect_self_assignments(body, src: bytes, cf: ClassFacts) -> None:
    stack = [body]
    while stack:
        n = stack.pop()
        if n.type in ("function_definition", "class_definition", "lambda"):
            continue
        if n.type == "expression_statement":
            _collect_assignment(n, src, cf, class_level=False)
        stack.extend(n.children)


# ── Pass 2: attribute references with inferred receivers ────────────────────

class _Scope:
    def __init__(self, parent: Optional["_Scope"], cls: Optional[str], bases: list[str]):
        self.parent = parent
        self.cls = cls
        self.bases = bases
        self.vars: dict[str, Optional[set[str]]] = {}

    def lookup(self, name: str) -> tuple[bool, Optional[set[str]]]:
        scope = self
        while scope is not None:
            if name in scope.vars:
                return True, scope.vars[name]
            scope = scope.parent
        return False, None


def attribute_refs(root, src: bytes, file_path: str, index: TypeIndex, method_names: set[str]) -> list[AttrRef]:
    """Every `receiver.name` (and `Class(...)` constructor call, recorded as
    `__init__`) where `name` is a method some repo class defines, with the
    receiver's inferred type(s): one AttrRef per possible class, EXTERNAL, or
    UNKNOWN."""
    refs: set[AttrRef] = set()

    def expr_type(node, scope: _Scope) -> Optional[set[str]]:
        t = node.type
        if t == "parenthesized_expression" and node.named_children:
            return expr_type(node.named_children[0], scope)
        if t == "identifier":
            name = _text(node, src)
            if name in ("self", "cls") and scope.cls:
                return {scope.cls}
            found, types = scope.lookup(name)
            if found:
                return types
            if index.is_class(name):
                return {name}
            return None
        if t == "call":
            fn = node.child_by_field_name("function")
            if fn is None:
                return None
            if fn.type == "identifier":
                name = _text(fn, src)
                if name == "super":
                    return set(scope.bases) or None
                if index.is_class(name):
                    return {name}
                anns = index.returns_by_name.get(name)
                if anns and len(anns) == 1:
                    return index.resolve_annotation(anns[0])
                return None
            if fn.type == "attribute":
                recv = fn.child_by_field_name("object")
                meth = _text(fn.child_by_field_name("attribute"), src)
                recv_types = expr_type(recv, scope) if recv is not None else None
                if not recv_types:
                    return None
                out: set[str] = set()
                for rt in recv_types:
                    if rt == EXTERNAL:
                        return None
                    if index.is_class(meth) and not index.defines(rt, meth):
                        out.add(meth)  # module-like access: pkg.Session()
                        continue
                    r = index.method_return(rt, meth)
                    if r is None:
                        return None
                    out |= r
                return out or None
            return None
        if t == "attribute":
            recv = node.child_by_field_name("object")
            attr = _text(node.child_by_field_name("attribute"), src)
            recv_types = expr_type(recv, scope) if recv is not None else None
            if not recv_types:
                return None
            if index.is_class(attr):
                return {attr}  # module.Class
            out: set[str] = set()
            for rt in recv_types:
                if rt == EXTERNAL:
                    return None
                types = index.attr_types(rt, attr)
                if types is None:
                    return None
                out |= types
            return out or None
        return None

    def bind(target, types: Optional[set[str]], scope: _Scope) -> None:
        if target is not None and target.type == "identifier":
            name = _text(target, src)
            if name in scope.vars and scope.vars[name] != types:
                prev = scope.vars[name]
                scope.vars[name] = (prev | types) if (prev and types) else None  # conflicting → merge/unknown
            else:
                scope.vars[name] = types

    def declare_params(fn_node, scope: _Scope) -> None:
        params = fn_node.child_by_field_name("parameters")
        if params is None:
            return
        for p in params.named_children:
            if p.type in ("typed_parameter", "typed_default_parameter"):
                name_node = p.child_by_field_name("name") or next(
                    (c for c in p.children if c.type == "identifier"), None)
                ann = p.child_by_field_name("type")
                if name_node is not None:
                    scope.vars[_text(name_node, src)] = (
                        index.resolve_annotation(_text(ann, src), scope.cls) if ann is not None else None)
            elif p.type in ("identifier", "default_parameter"):
                name_node = p if p.type == "identifier" else p.child_by_field_name("name")
                if name_node is not None:
                    name = _text(name_node, src)
                    if name not in ("self", "cls"):
                        scope.vars[name] = None

    def record(name_node_text: str, recv_types: Optional[set[str]], line: int) -> None:
        if name_node_text not in method_names:
            return
        if recv_types is None:
            refs.add(AttrRef(name_node_text, UNKNOWN, line))
        else:
            for rt in recv_types:
                refs.add(AttrRef(name_node_text, rt, line))

    def implicit(methods: tuple, expr, scope: _Scope, line: int, instances_only: bool = False) -> None:
        """Protocol methods Python calls for us (with, for, calling an instance).
        Only recorded when the operand's type is known; these never fall back
        to name matching, since no call site spells the method out."""
        if instances_only and expr.type == "identifier" and index.is_class(_text(expr, src)):
            return  # that's a constructor call, handled above
        types = expr_type(expr, scope)
        if not types:
            return
        for t in types:
            if t == EXTERNAL:
                continue
            for m in methods:
                if m in method_names:
                    refs.add(AttrRef(m, t, line))

    def walk(node, scope: _Scope) -> None:
        # Bind names first (flow-insensitive within a scope), then record refs.
        prebind(node, scope)
        stack = [node]
        while stack:
            n = stack.pop()
            if n is not node and n.type in ("function_definition", "class_definition"):
                enter(n, scope)
                continue
            if n.type == "attribute":
                recv = n.child_by_field_name("object")
                attr_node = n.child_by_field_name("attribute")
                if recv is not None and attr_node is not None:
                    record(_text(attr_node, src), expr_type(recv, scope), n.start_point[0] + 1)
            elif n.type == "call":
                fn = n.child_by_field_name("function")
                line = n.start_point[0] + 1
                if fn is not None and fn.type == "identifier" and index.is_class(_text(fn, src)):
                    refs.add(AttrRef("__init__", _text(fn, src), line))
                elif fn is not None and fn.type == "attribute" and index.is_class(
                        _text(fn.child_by_field_name("attribute"), src)):  # pkg.Session(...)
                    refs.add(AttrRef("__init__", _text(fn.child_by_field_name("attribute"), src), line))
                elif fn is not None and fn.type == "identifier" and _text(fn, src) in _BUILTIN_PROTOCOLS:
                    args = n.child_by_field_name("arguments")
                    if args is not None and args.named_children:  # len(x) → x.__len__
                        implicit((_BUILTIN_PROTOCOLS[_text(fn, src)],), args.named_children[0], scope, line)
                elif fn is not None and fn.type in ("identifier", "attribute"):
                    # calling an *instance* runs its __call__ (e.g. auth(r))
                    implicit(("__call__",), fn, scope, line, instances_only=True)
            elif n.type == "with_item":
                value = n.child_by_field_name("value")
                expr = value.named_children[0] if value is not None and value.type == "as_pattern" and value.named_children else value
                if expr is not None:
                    implicit(("__enter__", "__exit__"), expr, scope, n.start_point[0] + 1)
            elif n.type in ("for_statement", "for_in_clause"):
                right = n.child_by_field_name("right")
                if right is not None:
                    implicit(("__iter__",), right, scope, n.start_point[0] + 1)
            elif n.type == "subscript":
                value = n.child_by_field_name("value")
                parent = n.parent
                if value is not None:
                    if parent is not None and parent.type == "assignment" and parent.child_by_field_name("left") == n:
                        dunder = "__setitem__"
                    elif parent is not None and parent.type == "delete_statement":
                        dunder = "__delitem__"
                    else:
                        dunder = "__getitem__"
                    implicit((dunder,), value, scope, n.start_point[0] + 1)
            elif n.type == "comparison_operator":
                ops = [c.type for c in n.children if not c.is_named]
                operands = n.named_children
                if len(operands) == 2:
                    if "in" in ops or "not in" in ops:
                        implicit(("__contains__",), operands[1], scope, n.start_point[0] + 1)
                    elif "==" in ops or "!=" in ops:
                        implicit(("__eq__", "__ne__"), operands[0], scope, n.start_point[0] + 1)
            stack.extend(reversed(n.children))

    def prebind(node, scope: _Scope) -> None:
        stack = [node]
        while stack:
            n = stack.pop()
            if n is not node and n.type in ("function_definition", "class_definition", "lambda"):
                continue
            if n.type == "assignment":
                left, right, ann = (n.child_by_field_name(f) for f in ("left", "right", "type"))
                if ann is not None:
                    bind(left, index.resolve_annotation(_text(ann, src), scope.cls), scope)
                elif right is not None:
                    bind(left, expr_type(right, scope), scope)
            elif n.type == "with_item":
                value = n.child_by_field_name("value")
                if value is not None and value.type == "as_pattern":
                    expr = value.named_children[0] if value.named_children else None
                    target = value.child_by_field_name("alias") or (
                        value.named_children[-1] if len(value.named_children) > 1 else None)
                    if target is not None and target.type == "as_pattern_target" and target.named_children:
                        target = target.named_children[0]
                    if expr is not None:
                        bind(target, expr_type(expr, scope), scope)
            elif n.type in ("for_statement", "for_in_clause"):
                left = n.child_by_field_name("left")
                if left is not None and left.type == "identifier":
                    scope.vars[_text(left, src)] = None
            stack.extend(n.children)

    def enter(defn, scope: _Scope) -> None:
        if defn.type == "class_definition":
            name_node = defn.child_by_field_name("name")
            cname = _text(name_node, src) if name_node is not None else None
            inner = _Scope(scope, cname, index.bases(cname) if cname else [])
            body = defn.child_by_field_name("body")
            if body is not None:
                walk(body, inner)
        else:
            inner = _Scope(scope, scope.cls, scope.bases)
            declare_params(defn, inner)
            body = defn.child_by_field_name("body")
            if body is not None:
                walk(body, inner)
            # decorators, defaults and annotations belong to the enclosing scope
            for c in defn.children:
                if c.type not in ("block", "identifier", "parameters"):
                    walk(c, scope)

    walk(root, _Scope(None, None, []))
    return sorted(refs, key=lambda r: (r.line, r.name, r.receiver))

