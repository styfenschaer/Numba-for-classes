from functools import partial, singledispatch, wraps
from weakref import WeakKeyDictionary

from .transformation import transform


class _Dispatcher:
    def __init__(self, fn, **params):
        self.fn = fn
        self.params = params

    def __call__(self, obj, *args, **kwargs):
        try:
            return self.disp(*args, **kwargs)
        except AttributeError:
            self.disp, self.src = transform(obj, self.fn, self.params)
            return self.disp(*args, **kwargs)


class _Proxy:
    def __init__(self, fn, **params):
        self.fn = fn
        self.params = params
        self.cache = WeakKeyDictionary()

    def __get__(self, obj, cls):
        try:
            return self.cache[obj]
        except (KeyError, TypeError):
            if obj is None:
                return self

            disp = _Dispatcher(self.fn, **self.params)
            self.cache[obj] = partial(disp.__call__, obj)
            return self.cache[obj]


def jit(fn=None, **params):
    if not fn is None:
        return _Proxy(fn)

    @wraps(fn)
    def wrapper(fn):
        return _Proxy(fn, **params)

    return wrapper


def _get_dispatcher(fn):
    disp = fn.func.__self__
    return disp


@singledispatch
def _get_proxy(arg: _Proxy):
    return arg


@_get_proxy.register
def _(arg: partial):
    inst = arg.args[0]
    cls_name = inst.__class__
    disp = _get_dispatcher(arg)
    fn_name = disp.fn.__name__
    return getattr(cls_name, fn_name)


@singledispatch
def delete(fn: partial):
    inst = fn.args[0]
    proxy = _get_proxy(fn)
    proxy.cache.pop(inst)


@singledispatch
def reset(fn: partial):
    disp = _get_dispatcher(fn)
    del disp.disp, disp.src


def _list_or_tuple(arg):
    if not isinstance(arg, (tuple, list)):
        arg = arg,
    return arg


def _do_before(names, todo):
    def decorator(fn):
        def wrapper(inst, *args, **kwargs):
            for name in _list_or_tuple(names):
                try:
                    fn_jit = getattr(inst, name)
                    todo(fn_jit)
                except (KeyError, AttributeError):
                    pass

                return fn(inst, *args, **kwargs)
        return wrapper
    return decorator


delete_deco = partial(_do_before, todo=delete)
reset_deco = partial(_do_before, todo=reset)

for ty in (str, list, tuple):
    @delete.register
    def _(fn_name: ty):
        return delete_deco(fn_name)

    @reset.register
    def _(fn_name: ty):
        return reset_deco(fn_name)


def get_proxy_params(fn):
    proxy = _get_proxy(fn)
    return proxy.params


def set_proxy_params(fn, **params):
    proxy = _get_proxy(fn)
    proxy.params.update(params)


def get_dispatcher_params(fn):
    disp = _get_dispatcher(fn)
    return disp.params


def set_dispatcher_params(fn, **params):
    disp = _get_dispatcher(fn)
    disp.params.update(params)


def get_source(fn):
    disp = _get_dispatcher(fn)
    return disp.src


def print_source(fn):
    print(get_source(fn))
