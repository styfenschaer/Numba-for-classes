import ast
from functools import partial
from inspect import getfullargspec, getsourcelines
from operator import attrgetter

import numba as nb


class _AttrTransformer(ast.NodeTransformer):
    def __init__(self, inst):
        self.inst = inst
        self.self_ = getfullargspec(inst.__init__).args[0]
        self.ns = {}

    @staticmethod
    def gen_attr_name(chunks):
        return "__" + "_".join(chunks)

    def visit_Attribute(self, node):
        ids = []
        n = node
        while hasattr(n, "value"):
            ids.insert(0, n.attr)
            n = n.value
        ids.insert(0, n.id)

        if not self.self_ in ids:
            return node

        name = self.gen_attr_name(ids)
        path = ".".join(ids[1:])
        attr = attrgetter(path)(self.inst)

        if isinstance(attr, partial):
            attr, _ = transform(
                attr.args[0],
                attr.func.__self__.fn,
                attr.func.__self__.params,
            )
        self.ns.update({name: attr})

        node = ast.copy_location(ast.Name(id=name, ctx=node.ctx), node)
        return ast.fix_missing_locations(node)


class _FuncDefTransformer(ast.NodeTransformer):
    @staticmethod
    def gen_fn_name(base):
        return "__fn_" + base

    def visit_FunctionDef(self, node):
        node.args.args.pop(0)
        node.name = self.gen_fn_name(node.name)
        self.fn_name = node.name
        node.decorator_list = []
        return ast.fix_missing_locations(node)


def unindented_source(lns):
    wspaces = len(lns[0]) - len(lns[0].lstrip())
    lns = [ln[wspaces:] for ln in lns]
    return "".join(lns)


def transform(inst, fn, params):
    ft = _FuncDefTransformer()
    at = _AttrTransformer(inst)

    srclns = getsourcelines(fn)[0]
    src = unindented_source(srclns)
    tree = ast.parse(src)
    tree = ft.visit(tree)
    tree = at.visit(tree)

    ns = fn.__globals__
    ns.update(at.ns)
    src_new = ast.unparse(tree)
    exec(compile(src_new, "<ast>", "exec"), ns)
    fn_new = ns[ft.fn_name]
    fn_jit = nb.jit(**params)(fn_new)

    return fn_jit, src_new
