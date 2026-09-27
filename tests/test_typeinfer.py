import warnings

import pytest

warnings.filterwarnings("ignore", category=FutureWarning)
tree_sitter_languages = pytest.importorskip("tree_sitter_languages")

from app.core.typeinfer import EXTERNAL, UNKNOWN, TypeIndex, attribute_refs, collect_facts

LIB = b'''
from typing import IO, Optional

class BaseAdapter:
    def send(self, request): ...
    def close(self): ...

class HTTPAdapter(BaseAdapter):
    def send(self, request): ...

class Mailer:
    def send(self, msg): ...
    def __call__(self, x): ...
    def __enter__(self): ...
    def __exit__(self, *a): ...
    def __iter__(self): ...
    def __getitem__(self, k): ...
    def __len__(self): ...

class Session:
    adapter: BaseAdapter
    def __init__(self):
        self.mailer = Mailer()
    def get_adapter(self, url) -> BaseAdapter: ...
    def maybe(self) -> Optional[HTTPAdapter]: ...
    def send(self, request):
        adapter = self.get_adapter(request)
        adapter.send(request)
        self.mailer.send("hi")
        self.close_all()
    def close_all(self): ...
'''

USER = b'''
def run(a: HTTPAdapter, fp: IO[str], thing):
    a.send(1)
    fp.read()
    thing.send(2)
    s = Session()
    s.send(3)
    s.maybe().send(4)
    m = Mailer()
    m(5)
    with Mailer() as ctx:
        pass
    for item in m:
        pass
    m["k"]
    len(m)

class Sub(HTTPAdapter):
    def send(self, request):
        super().send(request)
'''


def _refs(sources: dict[str, bytes]):
    parser = tree_sitter_languages.get_parser("python")
    trees = {f: parser.parse(src) for f, src in sources.items()}
    index = TypeIndex.build(collect_facts(t.root_node, sources[f], f) for f, t in trees.items())
    methods = {m for cs in index.classes.values() for c in cs for m in c.methods} | {"read"}
    return {
        f: {(r.name, r.receiver) for r in attribute_refs(t.root_node, sources[f], f, index, methods)}
        for f, t in trees.items()
    }


@pytest.fixture(scope="module")
def refs():
    return _refs({"lib.py": LIB, "user.py": USER})


def test_receivers_from_annotations_constructors_and_returns(refs):
    user = refs["user.py"]
    assert ("send", "HTTPAdapter") in user         # a: HTTPAdapter
    assert ("send", "Session") in user             # s = Session()
    assert ("send", "HTTPAdapter") in user         # s.maybe() -> Optional[HTTPAdapter]
    assert ("__init__", "Session") in user         # the constructor call itself
    assert ("send", UNKNOWN) in user               # thing: untyped parameter
    assert ("read", EXTERNAL) in user              # fp: IO[str] is not a repo class


def test_self_attributes_and_return_types_inside_a_class(refs):
    lib = refs["lib.py"]
    assert ("send", "BaseAdapter") in lib          # adapter = self.get_adapter(...) -> BaseAdapter
    assert ("send", "Mailer") in lib               # self.mailer = Mailer()
    assert ("close_all", "Session") in lib         # self.close_all()


def test_implicit_protocol_methods(refs):
    user = refs["user.py"]
    for dunder in ("__call__", "__enter__", "__exit__", "__iter__", "__getitem__", "__len__"):
        assert (dunder, "Mailer") in user, dunder


def test_super_resolves_to_base_class(refs):
    assert ("send", "HTTPAdapter") in refs["user.py"]


def test_annotation_resolution():
    index = TypeIndex.build([])
    index.classes["Foo"] = []
    assert index.resolve_annotation("Foo") == {"Foo"}
    assert index.resolve_annotation("'Foo | None'") == {"Foo"}
    assert index.resolve_annotation("t.Optional[Foo]") == {"Foo"}
    assert index.resolve_annotation("dict[str, Foo]") == {EXTERNAL}   # the receiver is the dict
    assert index.resolve_annotation("str") == {EXTERNAL}
    assert index.resolve_annotation("t.Any") is None                 # uninformative → unknown
    assert index.resolve_annotation("Foo | Any") is None
