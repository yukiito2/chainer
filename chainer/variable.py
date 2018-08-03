import collections
import copy
import heapq
import traceback
import warnings
import weakref
import threading
import gc

import numpy

import chainer
from chainer import cuda
from chainer import initializers
from chainer.initializers import constant
from chainer.utils import argument

from chainer import configuration
from chainer.cuda import memory_pool

# control whether to swap or not
is_targets = list()
# control swap-in timing
swapin_counts = list()
ooc_optimize_setting = None

def _check_grad_type(func, x, gx):
    if x.data is None or gx is None:
        # ``x.data is None`` implies that the data array is not retained
        return
    if not isinstance(gx, type(x.data)):
        msg = ('Type of data and grad mismatch\n%s != %s' %
               (type(x.data), type(gx)))
        typ = TypeError
    elif gx.dtype != x.data.dtype:
        msg = ('Dtype of data and grad mismatch\n%s != %s' %
               (x.data.dtype, gx.dtype))
        typ = TypeError
    elif gx.shape != x.data.shape:
        msg = ('Shape of data and grad mismatch\n%s != %s' %
               (x.data.shape, gx.shape))
        typ = ValueError
    else:
        return

    detail = ''
    if func:
        detail = 'Function `{0}` ({1}) has a bug.\n'.format(
            type(func)._impl_name, func.label)
        stack = func.stack
        if stack:
            detail += 'Stacktrace of the function is below:\n'
            for line in traceback.format_list(func.stack):
                detail += line
        detail += '''
Please report this error to the issue tracker with the stack trace,
the information of your environment, and your script:
https://github.com/chainer/chainer/issues/new.
'''.format(type(func).__name__, func.label)

    raise typ(detail + msg)


def variable_repr(var):
    """Return the string representation of a variable.

    Args:
        var (~chainer.Variable): Input Variable.
    .. seealso:: numpy.array_repr
    """
    xp = cuda.get_array_module(var)
    if xp is numpy:
        arr = var.data
    else:
        arr = var.data.get()

    if var.name:
        prefix = 'variable ' + var.name
    else:
        prefix = 'variable'

    if arr is None:
        lst = 'None'
    elif arr.size > 0 or arr.shape == (0,):
        lst = numpy.array2string(arr, None, None, None, ', ', prefix + '(')
    else:  # show zero-length shape unless it is (0,)
        lst = '[], shape=%s' % (repr(arr.shape),)

    return '%s(%s)' % (prefix, lst)


def variable_str(var):
    """Return the string representation of a variable.

    Args:
        var (~chainer.Variable): Input Variable.
    .. seealso:: numpy.array_str
    """
    xp = cuda.get_array_module(var)
    if xp is numpy:
        arr = var.data
    else:
        arr = var.data.get()

    if var.name:
        prefix = 'variable ' + var.name
    else:
        prefix = 'variable'

    if arr is None:
        lst = 'None'
    else:
        lst = numpy.array2string(arr, None, None, None, ' ', prefix + '(')

    return '%s(%s)' % (prefix, lst)


def _add_instance(instances, seen_set, instance):
    """Add instance

    Copied from anaruse's repository
    Source: https://github.com/anaruse/
            chainer/blob/OOC_chainer_v202/chainer/variable.py
    """
    if instance is not None and instance not in seen_set:
        instances.append(instance)
        seen_set.add(instance)


def out_of_core_mode(async=True, fine_granularity=False, debug=False,
                     devices=None, optimize_setting=None):
    """Enable out of core training mode

    Originally from anaruse's repository
    Source: https://github.com/anaruse/
            chainer/blob/OOC_chainer_v202/chainer/variable.py
    """
    global ooc_optimize_setting

    events = []
    streams = []
    if devices is None:
        devices = [cuda.Device().id]
    if len(devices) == 1:
        with cuda.Device(devices[0]):
            if async:
                streams.append(cuda.Stream(non_blocking=True))
                streams.append(cuda.Stream(non_blocking=True))
            else:
                streams.append(cuda.Stream.null)
                streams.append(cuda.Stream.null)

    if optimize_setting in ['keep_all', 'swap_all_no_scheduling', 'swap_all', 'recompute_all', 'swap_opt', 'superneurons']:
        ooc_optimize_setting = optimize_setting
        
    return configuration.using_config('out_of_core_params',
                                      [True, async,
                                       fine_granularity, streams,
                                       events, debug])


class VariableNode(object):

    """Node in the backward computational graph representing a variable.

    This object represents a variable node in a computational graph. The node
    is used in error backpropagation (a.k.a. backprop) to determine which
    gradient to be passed to each function.

    A variable node is held by the corresponding :class:`Variable` object,
    which is managed by users. :class:`Function` objects that take the variable
    as an input also hold references to the variable node.

    Note that the node does not hold a reference to the corresponding data
    array in general. The data array is actually accessible by the node in the
    following cases.

    1. If there exists a :class:`Variable` object that holds a reference to the
       variable node, the variable node holds a weak reference to the variable
       object, and thus the data array is accessible via the weak reference.
    2. If :meth:`retain_data` is called, the node holds a reference to the data
       array. It is mainly called by a function that needs the input or output
       data array in its backprop procedure. See :meth:`Function.retain_inputs`
       and :meth:`Function.retain_outputs` for more details.

    Users usually do not need to touch this variable node object. The
    computational graph is automatically managed by Chainer, and any interface
    that is beneficial for users is also provided by :class:`Variable`.

    Args:
        variable (Variable): The corresponding variable object.
        name (str): Name of the variable node.

    Attributes:
        ~VariableNode.dtype: Data type of the data array.
        ~VariableNode.shape: Shape of the data array.
        ~VariableNode.name (str): Name of the variable node.

    """

    _creator_node = None
    _data = None
    _rank = 0
    # Name of the Function is assigned if this variable is a gradient generated
    # by an old-style Function
    _old_style_grad_generator = None

    def __init__(self, variable, name, **kwargs):
        argument.check_unexpected_kwargs(
            kwargs,
            grad='unexpected keyword argument "grad": '
                 'pass the gradient to Variable instead'
        )
        self._variable = weakref.ref(variable)
        self.name = name
        self._requires_grad = variable.requires_grad

        vdata = variable.data
        self._set_data_type(vdata)

        # [OOC/LWR]
        self._creator_node_g = None  # original creator node
        self._break_point = None  # True if this node is a break point
        # Store the gradient variable of break-point nodes that have
        # no corresponding variable object.
        # It is used to do backpropagation again from this node
        self._grad_var = None
        self._is_data_swapout = False
        self.is_target = None
        self.already_swapin = False

    @property
    def creator(self):
        """Function object that created this variable node.

        When the function is implemented with the old-style API (i.e., it uses
        :class:`Function` class), this property returns the :class:`Function`
        object. The object is extracted from the :class:`FunctionAdapter`
        object, so the returned object is not the function node, but instead
        the actual implementation of forward and backward procedures.

        When the function is implemented with the new-style API (i.e., it uses
        :class:`FunctionNode` class), this property returns the function node
        object. In this case, the returned object is same as
        :attr:`creator_node`.

        .. warning::

           As of v3.0.0, when the creator is an old-style function, the
           following code is invalid:

           .. code-block:: python

              creator = v.creator
              v.creator = None
              ...
              v.creator = creator

           The point is that :class:`FunctionNode` objects are used as nodes
           in the computational graph instead of :class:`Function`, and each
           :class:`Function` object only holds a *weak reference* to the
           corresponding :class:`FunctionNode`. Since ``creator`` returns the
           :class:`Function` object, the :class:`FunctionNode` object is not
           kept by preserving ``creator``.

           The above code should be fixed as follows.

           .. code-block:: python

              creator_node = v.creator_node
              v.creator_node = None
              ...
              v.creator_node = creator_node

        """
        node = self._creator_node
        if node is None:
            return None

        if isinstance(node, chainer.function.FunctionAdapter):
            return node.function
        return node

    @creator.setter
    def creator(self, func):
        self.creator_node = func

    @property
    def creator_node(self):
        """Function node that has this variable as an output.

        See :class:`FunctionNode` for the definition of a function node.

        """
        return self._creator_node

    @creator_node.setter
    def creator_node(self, func):
        if isinstance(func, chainer.Function):
            func = func.node
        self._creator_node = func
        # [OOC/LWR]
        self._creator_node_g = func
        if func is not None:
            self._rank = func.rank + 1

    @property
    def data(self):
        """Data array of the corresponding variable.

        If the data is not available, it returns ``None``.

        """
        return self._data

    @data.setter
    def data(self, d):
        self._data = d
        self._set_data_type(d)

    @property
    def grad(self):
        """Gradient array of the corresponding variable.

        If the variable is not available, it returns ``None``.

        """
        var = self.get_variable()
        return None if var is None else var.grad

    @property
    def grad_var(self):
        """Gradient variable of the corresponding variable.

        If the corresponding variable is not available, it return ``None``.

        """
        var = self.get_variable()
        return None if var is None else var._grad_var

    @property
    def label(self):
        """Short text that represents the variable node."""
        if self.shape == ():
            return str(self.dtype)
        return '(%s), %s' % (', '.join(map(str, self.shape)),
                             str(self.dtype))

    @property
    def rank(self):
        return self._rank

    @property
    def requires_grad(self):
        """It indicates that ``grad`` will be set in backward calculation."""
        return self._requires_grad

    def get_variable(self):
        """Returns the corresponding :class:`Variable` object.

        VariableNode object holds a weak reference of the variable object. If
        the reference is alive, it is returned by this property. Otherwise,
        this property creates a new :class:`Variable` object from this node
        object and returns it.

        Returns:
            Variable: The variable object that refers this node.

        """
        var = self._variable()
        if var is not None:
            return var

        var = Variable(self.data, name=self.name,
                       requires_grad=self._requires_grad)
        var._node = self
        return var

    def set_creator(self, creator):
        """Sets a :class:`Function` object that created this node.

        This method is equivalent to ``self.creator = creator``. A
        :class:`FunctionNode` object can also be passed.

        Args:
            creator (Function or FunctionNode): Function that has created this
                variable.

        """
        self.creator = creator

    def set_creator_node(self, creator_node):
        """Sets a :class:`FunctionNode` object that created this node.

        This method is equivalent to ``self.creator_node = creator_node``. A
        :class:`Function` object can also be passed, in which case the
        :attr:`~Function.node` object is extracted.

        Args:
            creator_node (FunctionNode or Function): Function node that has
                this variable as an output.

        """
        self.creator_node = creator_node

    def unchain(self):
        """Deletes the reference to the creator of this variable node.

        This method is equivalent to ``self.creator_node = None``.

        """
        self.creator_node = None
        # [OOC/LWR]
        self._creator_node_g = None

    def retain_data(self):
        """Lets the node hold a reference to the underlying data array.

        This method gets the data array of the corresponding variable and keeps
        it. If the weak reference to the corresponding variable is dead, it
        raises an error.

        """
        variable = self._variable()
        if variable is not None:
            self.data = variable.data
        else:
            raise RuntimeError('cannot retain variable data: the variable has '
                               'been already released')

    def _set_data_type(self, d):
        if d is None:
            self.dtype = None
            self.shape = None
        else:
            self.dtype = d.dtype
            self.shape = d.shape

    def _check_old_style_gradient(self):
        if self._old_style_grad_generator is not None:
            raise RuntimeError(
                'cannot twice-differentiate an old style Function "%s"' %
                self._old_style_grad_generator)
            
    def ancestors_free(self):
        ancestor_vnodes = self.ancestors()
        ancestor_vnodes.append(self)
        
        for vnode in ancestor_vnodes:
            if vnode.creator is None:
                continue
            variable = vnode._variable()
            if variable is None and vnode.data is not None:
                vnode.data.free_data()
                vnode.data = None
                
    def ancestors_swapout(self, stream=None, inclusive=False,
                          early_stop=False, events=None, debug=False):
        """...

        Copied from anaruse's repository
        Source: https://github.com/anaruse/
                chainer/blob/OOC_chainer_v202/chainer/variable.py
        """
        if early_stop:
            ancestor_vnodes = self.ancestors_whose_data_on_gpu()
        else:
            ancestor_vnodes = self.ancestors()
        if inclusive:
            ancestor_vnodes.append(self)
        if debug:
            print('# variablep.py:319, *_swapout(), ancestors: {}'
                  .format(ancestor_vnodes))

        for vnode in ancestor_vnodes:
            if vnode.creator is None:
                continue
            vnode.to_swap(stream=stream, events=events, debug=debug)

    def ancestors_swapin(self, stream=None, inclusive=False,
                         debug=False):
        """...

        Copied from anaruse's repository
        Source: https://github.com/anaruse/
                chainer/blob/OOC_chainer_v202/chainer/variable.py
        """
        ancestor_vnodes = self.ancestors_for_recompute()
        if inclusive:
            ancestor_vnodes.append(self)
        if debug:
            print('# variablep.py:333, *_swapin(), ancestors: {}'
                  .format(ancestor_vnodes))

        for vnode in ancestor_vnodes:
            vnode.to_gpu(stream=stream, debug=debug)
            
    def ancestors_swapin_bytes(self, stream=None, inclusive=False,
                         debug=False):
        ancestor_vnodes = self.ancestors_for_recompute(True)
        if inclusive:
            ancestor_vnodes.append(self)
        swapin_bytes = 0
        for vnode in ancestor_vnodes:
            if (vnode._is_data_swapout is True) and (vnode.already_swapin is False):
                swapin_bytes += vnode._data.nbytes 
                vnode.already_swapin = True
        return swapin_bytes

    def ancestors(self):
        """Gets a list of my ancestor variable nodes.

        Originally from anaruse's repository
        Source: https://github.com/anaruse/
                chainer/blob/OOC_chainer_v202/chainer/variable.py
        """
        ancestor_funcs = []
        ancestor_vnodes = []
        seen_funcs = set()
        seen_vnodes = set()

        _add_instance(ancestor_funcs, seen_funcs, self._creator_node_g)
        while ancestor_funcs:
            func = ancestor_funcs.pop()
            for vnode in func.inputs:
                _add_instance(ancestor_vnodes, seen_vnodes, vnode)
                _add_instance(ancestor_funcs, seen_funcs, vnode.creator_node)

        return ancestor_vnodes

    def ancestors_whose_data_on_gpu(self):
        """Gets a list of my ancestor variable nodes.

        Originally from anaruse's repository
        Source: https://github.com/anaruse/
                chainer/blob/OOC_chainer_v202/chainer/variable.py

        """
        ancestor_funcs = []
        ancestor_vnodes = []
        seen_funcs = set()
        seen_vnodes = set()

        _add_instance(ancestor_funcs, seen_funcs, self._creator_node_g)
        while ancestor_funcs:
            func = ancestor_funcs.pop()
            for vnode in func.inputs:
                #if vnode._is_data_swapout is False and vnode.is_target is not "keep":
                if vnode.is_target is not "keep":
                    _add_instance(ancestor_vnodes, seen_vnodes, vnode)
                    _add_instance(ancestor_funcs, seen_funcs, vnode.creator_node)

        return ancestor_vnodes

    def ancestors_for_recompute(self, swapin_bytes_mode=False):
        ancestor_vnodes = []

        def trace_break_points(vnode):
            seen_vnodes = set()
            funcs = []
            seen_funcs = set()

            seen_vnodes.add(vnode)
            func = vnode._creator_node_g
            if func is not None:
                funcs.append(func)
                seen_funcs.add(func)
            
            while funcs:
                func = funcs.pop()
                for new_vnode in func.inputs:
                    if (new_vnode.data is not None) or (new_vnode.is_target == "recompute"):
                        return seen_funcs

                    if new_vnode in seen_vnodes:
                        continue
                    seen_vnodes.add(new_vnode)
                    
                    new_func = new_vnode._creator_node_g
                    if new_func not in seen_funcs and new_func is not None:
                        funcs.append(new_func)
                        seen_funcs.add(new_func)
            return seen_funcs

        def trace_recompute(vnode):
            new_vnodes = [vnode]

            if vnode.data is None:
                # for controle swapin vars
                if swapin_bytes_mode and vnode.already_swapin:
                    return new_vnodes

                tmp_func = vnode._creator_node_g
                for x in tmp_func.inputs:
                    new_vnodes += trace_recompute(x)
            return new_vnodes

        for func in trace_break_points(self):
            if func._input_indexes_to_retain is not None:
                for index in func._input_indexes_to_retain:
                    ancestor_vnodes += trace_recompute(func.inputs[index])

            if func._output_indexes_to_retain is not None:
                for index in func._output_indexes_to_retain:
                    ancestor_vnodes += trace_recompute(func.outputs[index]())

        return ancestor_vnodes


    def to_swap(self, stream=None, events=None,
                debug=False, force=False):
        """Copies the data and gradient arrays to pinned memory.

        Originally from anaruse's repository
        Source: https://github.com/anaruse/
                chainer/blob/OOC_chainer_v202/chainer/variable.py

        """
        variable = self._variable()
        if force is False:
            if variable is not None:
                # Does not swap-out the array
                # when it is linked from the variable.
                return

        if self.data is not None:
            if self._is_data_swapout is False:
                if debug:
                    print('# variable.py:377, to_swap(), {} {}'.format(
                        self, self._creator_node))
                
                # check whether to swap or not
                if self.is_target is None:
                    if len(is_targets) == 0:
                        self.is_target = "swap"
                    elif self.data.nbytes == is_targets[0][1]:
                        self.is_target = is_targets.pop(0)[0]
                    else:
                        self.is_target = "keep"

                if self.is_target == "recompute":
                    self.data.free_data()
                    self.data = None
                    return
                elif self.is_target != "swap":
                    return

                if memory_pool.get_profile_mode() and stream is not None:
                    memory_pool.memory_log_add(("swapout", self.data.nbytes, str(self), str(self.data.data.ptr)))
                self._data = cuda.to_swap(self.data, stream=stream)
                self._is_data_swapout = True

    def to_gpu(self, stream=None, events=None,
               debug=False):
        """Copies the data and gradient arrays to GPU memory.

        Copied from anaruse's repository
        Source: https://github.com/anaruse/
                chainer/blob/OOC_chainer_v202/chainer/variable.py

        """
        if self.data is not None:
            if self._is_data_swapout is True:
                if debug:
                    print('# variable.py:389, to_gpu(), {}'.format(self))

                if memory_pool.get_profile_mode() and stream is not None:
                    memory_pool.memory_log_add(("swapin", self.data.nbytes, str(self)))

                self._data = cuda.to_gpu(self.data, stream=stream)
                self._is_data_swapout = False

    def interrupt_backward(self):
        """Cuts a link to my creator function temporarily.

        Originally from anaruse's repository
        Source: https://github.com/anaruse/
                chainer/blob/OOC_chainer_v202/chainer/variable.py

        """
        self._creator_node_g = self._creator_node
        self._creator_node = None

    def resume_backward(self):
        """Recovers a link to my creator function.

        Copied from anaruse's repository
        Source: https://github.com/anaruse/
                chainer/blob/OOC_chainer_v202/chainer/variable.py

        """
        self._creator_node = self._creator_node_g

    def _show_memory_usage(self):
        """Show memory usage.

        Copied from anaruse's repository
        Source: https://github.com/anaruse/
                chainer/blob/OOC_chainer_v202/chainer/variable.py

        """
        tmp = []
        seen_tmp = set()

        funcs = []
        seen_funcs = set()

        _add_instance(tmp, seen_tmp, self._creator_g)
        while tmp:
            func = tmp.pop()
            heapq.heappush(funcs, (~func.rank, len(seen_funcs), func))
            seen_funcs.add(func)
            for vnode in func.inputs:
                if vnode._creator_g is not None:
                    _add_instance(tmp, seen_tmp, vnode._creator_g)

        print('# _show_memory_usage()')

        total_data_size = 0
        total_grad_size = 0
        total_param_size = 0
        total_unkn_size = 0
        while funcs:
            rank, _, func = heapq.heappop(funcs)
            outputs = [y() for y in func.outputs]
            for y in outputs:
                if y.data is not None:
                    if y._is_data_swapout is False:
                        size = y.data.data.mem.size
                        ptr = y.data.data.mem.ptr
                        print('#     {} data {} {} ({})'
                              .format(rank, func, size, ptr))
                        total_data_size += size
                if y.grad is not None:
                    size = y.grad.data.mem.size
                    ptr = y.grad.data.mem.ptr
                    print('#     {} grad {} {} ({})'
                          .format(rank, func, size, ptr))
                    total_grad_size += size

            for varn in func.inputs:
                var = varn._variable()
                if var is not None and var.__class__.__name__ == 'Parameter':
                    size = var.data.data.mem.size
                    ptr = var.data.data.mem.ptr
                    # print('#     {} param {} {} ({})'
                    #        .format(rank, func, size, ptr))
                    total_param_size += size
                # else:
                #     if varn._creator_g is not None:
                #         continue
                #     size = varn.data.data.mem.size
                #     ptr = varn.data.data.mem.ptr
                #     print('#     {} unkn {} {} ({})'
                #           .format(rank, func, size, ptr))
                #     total_unkn_size += size

        print('#     total_data_size: {}'.format(total_data_size))
        print('#     total_grad_size: {}'.format(total_grad_size))
        print('#     total_para_size: {}'.format(total_param_size))
        print('#     total_unkn_size: {}'.format(total_unkn_size))

    def set_break_point(self):
        """Set break point

        Originally from anaruse's repository
        Source: https://github.ibm.com/IMAIHAL/
                chainer_v2_ooc/blob/OOC_chainer_v202/chainer/cuda.py
        """
        self._break_point = True

    def get_break_points(self, fine_granularity=False):
        """Get break points

        Originally from anaruse's repository
        Source: https://github.com/anaruse/
                chainer/blob/OOC_chainer_v202/chainer/variable.py
        """
        funcs = []
        seen_funcs = set()
        break_points = []
        seen_break_points = set()
        seen_vnodes = set()

        def add_break_point(cand):
            if cand not in seen_break_points and cand.creator_node is not None:
                cand.interrupt_backward()
                cand._break_point = True
                heapq.heappush(break_points,
                               (~cand.rank, len(seen_break_points), cand))
                seen_break_points.add(cand)

        add_break_point(self)
        _add_instance(funcs, seen_funcs, self._creator_node_g)
        while funcs:
            func = funcs.pop()
            for vnode in func.inputs:
                if vnode in seen_vnodes:
                    add_break_point(vnode)
                if getattr(vnode, '_break_point', False):
                    add_break_point(vnode)
                    # debug
                    # print('# variable.py:445, user set break point: {}'
                    #       .format(vnode))

                if fine_granularity and ((vnode.data is not None) or (vnode.is_target == "recompute")):
                    add_break_point(vnode)

                if vnode not in seen_vnodes:
                    seen_vnodes.add(vnode)

                _add_instance(funcs, seen_funcs, vnode._creator_node_g)

        return break_points


def _create_variable(data, name, grad, requires_grad):
    return Variable(
        data, name=name, grad=grad, requires_grad=requires_grad)


class Variable(object):

    """__init__(data=None, *, name=None, grad=None, requires_grad=True)

    Array with a structure to keep track of computation.

    Every variable holds a data array of type either :class:`numpy.ndarray` or
    :class:`cupy.ndarray`.

    A variable object holds a data array and a :class:`VariableNode` object of
    a computational graph. If the variable is constructed by the user, the node
    is *root* and does not hold any parent. If the variable is constructed by a
    :class:`FunctionNode` object, the node holds a reference to its parent
    called :attr:`creator_node`. This reference is used in backpropagation to
    backtrack the graph.

    Users can disable (resp. enable) this chaining behavior by calling
    :func:`~chainer.no_backprop_mode` (resp.
    :func:`~chainer.force_backprop_mode`).
    In the former context, a variable never creates a computational graph,
    whereas in the latter context, it is forced to create.

    .. warning::

       ``volatile`` argument is not supported anymore since v2.
       Instead, use :func:`chainer.no_backprop_mode`.

    Args:
        data (numpy.ndarray or cupy.ndarray): Initial data array.
        name (str): Name of the variable.
        grad (numpy.ndarray or cupy.ndarray): Initial gradient array.
        requires_grad (bool): Boolean indicating whether ``grad`` will be set
            in backward calculation.

    """  # NOQA

    def __init__(self, data=None, **kwargs):
        argument.check_unexpected_kwargs(
            kwargs, volatile='volatile argument is not supported anymore. '
            'Use chainer.using_config')
        name, grad, requires_grad \
            = argument.parse_kwargs(
                kwargs, ('name', None), ('grad', None),
                ('requires_grad', True))

        if (data is not None and
                not isinstance(data, (numpy.ndarray, cuda.ndarray))):
            msg = '''numpy.ndarray or cuda.ndarray are expected.
Actual: {0}'''.format(type(data))
            raise TypeError(msg)

        # Use a list as a data structure to hold the data array indirectly to
        # abstract its initialized/uninitialized state.
        self._data = [data]
        self._requires_grad = requires_grad
        self._node = VariableNode(self, name)
        self._grad_var = None if grad is None else Variable(grad)
        self.data_size = 0
        if self.data is not None:
            self.data_size = self.data.size

    def __copy__(self):
        return self._copy_to(Variable())

    def _copy_to(self, target):
        target.__dict__ = copy.copy(self.__dict__)
        target._node = VariableNode(target, self.name)
        return target

    def __reduce__(self):
        return _create_variable, (self.data, self.name, self.grad,
                                  self._requires_grad)

    def __repr__(self):
        return variable_repr(self)

    def __str__(self):
        return variable_str(self)

    @property
    def name(self):
        return self._node.name

    @name.setter
    def name(self, n):
        self._node.name = n

    def summary(self):
        if self.name:
            return '<variable %s>' % self.name
        else:
            return '<variable at 0x%x>' % id(self)

    def debug_print(self):
        """Display a summary of the stored data and location of the Variable"""

        msg = """{summary}
- device: {device}
- backend: {background}
- shape: {shape}
- dtype: {dtype}
- statistics: {stats}
- grad: {grad}"""

        stats_msg = 'mean={0:.8f}, std={1:.8f}'

        try:
            device = self.data.device
        except AttributeError:
            device = 'CPU'

        with cuda.get_device_from_array(self.data) as dev:
            xp = numpy if int(dev) == -1 else cuda.cupy

            if self.grad is None:
                grad = None
            elif xp.all(self.grad == 0):
                grad = 0
            else:
                grad = stats_msg.format(float(xp.mean(self.grad)),
                                        float(xp.std(self.grad)))

            stats = stats_msg.format(float(xp.mean(self.data)),
                                     float(xp.std(self.data)))

        return msg.format(summary=self.summary(),
                          grad=grad, shape=self.data.shape,
                          background=type(self.data),
                          dtype=self.data.dtype, device=device,
                          stats=stats)

    def __pos__(self):
        return self

    def __len__(self):
        """Returns the first dimension of the data array.

        Returns:
            int: Number of the first dimension of the data array.

        """
        return len(self.data)

    @property
    def label(self):
        """Short text that represents the variable."""
        return self._node.label

    @property
    def creator(self):
        """Function implementation that created this variable.

        When this variable has been created by an old-style function (i.e., it
        is implemented as a subclass of :class:`Function`), this property
        returns that :class:`Function` object.

        When this variable has been created by a new-style function (i.e., it
        is implemented as a subclass of :class:`FunctionNode` class), this
        property returns that node object.

        """
        return self._node.creator

    @creator.setter
    def creator(self, func):
        self._node.creator = func

    @property
    def creator_node(self):
        """:class:`FunctionNode` object that created this variable.

        This property has a setter to which ``None`` can be set. Setting
        ``None`` to this property is equivalent to call :meth:`unchain`;
        it purges the variable from the function that created this variable.

        The setter also accepts the original :class:`FunctionNode` object that
        created this variable. For example, you can once set ``None`` to this
        property and then set the original value again.

        .. note::
           Setting an irrelevant :meth:`FunctionNode` object does not emit any
           error immediately, whereas the behavior is undefined. Do not set
           a :meth:`FunctionNode` object that did not create this variable
           object.

        """
        return self._node._creator_node

    @creator_node.setter
    def creator_node(self, func):
        self._node.creator_node = func

    @property
    def array(self):
        """The underlying data array.

        It is either :class:`numpy.ndarray` or :class:`cupy.ndarray` object,
        or ``None`` if the variable in in an uninitialized state.

        """
        return self._data[0]

    @array.setter
    def array(self, d):
        self._data[0] = d
        self._node._set_data_type(d)

    @property
    def data(self):
        """The underlying data array (equivalent to :attr:`array`).

        Note that using this attribute directly is discouraged; use
        :attr:`array` instead. Using :attr:`array`, you can find an error
        earlier when your code mixes up Variable and ndarray because
        ndarray does not have an attribute ``.array`` while it has
        ``.data``.

        """
        return self._data[0]

    @data.setter
    def data(self, d):
        self._data[0] = d
        self._node._set_data_type(d)

    @property
    def grad(self):
        """Gradient array of this variable.

        Note that this property returns the underlying array of the gradient
        variable instead of the gradient variable itself; to get/set
        gradient variable, use :attr:`grad_var` instead.

        """
        gv = self._grad_var
        return None if gv is None else gv.data

    @grad.setter
    def grad(self, g):
        self.grad_var = None if g is None else Variable(g)

    @property
    def grad_var(self):
        """Gradient variable."""
        return self._grad_var

    @grad_var.setter
    def grad_var(self, g):
        if g is not None:
            _check_grad_type(None, self, g.data)
        self._grad_var = g

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def size(self):
        return self.data.size

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def rank(self):
        return self._node.rank

    @property
    def node(self):
        return self._node

    @property
    def requires_grad(self):
        """It indicates that ``grad`` will be set in backward calculation."""
        return self._requires_grad

    @property
    def T(self):
        """Transposition of this variable."""
        return chainer.functions.transpose(self)

    def to_cpu(self):
        """Copies the data and gradient arrays to CPU."""
        if self.data is None:
            return

        self._data = [cuda.to_cpu(self.data)]
        if self._grad_var is not None:
            self._grad_var.to_cpu()
        # ensure that the node tracks the device migration
        node = self._node
        if node._data is not None:
            node.retain_data()

    def to_gpu(self, device=None):
        """Copies the data and gradient arrays to specified GPU.

        Args:
            device: Target device specifier. If omitted, the current device is
                used.

        """
        if self.data is None:
            self._initial_device = (cuda.Device().id
                                    if device is None else device)
        else:
            self._data = [cuda.to_gpu(self.data, device)]
            if self._grad_var is not None:
                self._grad_var.to_gpu(device)
            # ensure that the node tracks the device migration
            node = self._node
            if node._data is not None:
                node.retain_data()

    def cleargrad(self):
        """Clears the gradient array."""
        self._grad_var = None

    def zerograd(self):
        """Initializes the gradient array by zeros.

        Note that the gradient variable is unchained from the computational
        graph by this method because this operation breaks the backprop
        validity.

        .. deprecated:: v1.15
           Use :meth:`cleargrad` instead.

        """
        warnings.warn(
            'Variable.zerograd is deprecated. Use Variable.cleargrad instead.',
            DeprecationWarning)

        if self.data is None:
            return

        with cuda.get_device_from_array(self.data) as dev:
            gv = self._grad_var
            if gv is None:
                xp = numpy if dev.id == -1 else cuda.cupy
                self.grad = xp.zeros_like(self.data)
            else:
                gv.unchain()
                gv.data.fill(0)

    def copydata(self, var):
        """Copies the data array from given source variable.

        This method copies the data array from given variable to this variable.
        The copy is done even if the arrays reside on different devices,
        including across the host and a GPU device. If this variable has an
        uninitialized data array, this method initializes it by the data array
        of the given variable. Similarly, if the given variable has an
        uninitialized data array, this method initializes it by the data array
        of this variable (``self``). If both are uninitialized, this method
        does nothing.

        Args:
            var (Variable): Source variable.

        """
        src = var.data
        dst = self.data
        if src is None:
            if dst is None:
                return
            var.initialize(self.shape)
            src = var.data
        elif dst is None:
            self.initialize(src.shape)
            dst = self.data
        src_xp = cuda.get_array_module(src)
        dst_xp = cuda.get_array_module(dst)
        if dst_xp is src_xp:
            dst_xp.copyto(dst, src)
        elif dst_xp is numpy:
            dst_xp.copyto(dst, src.get())
        else:
            dst.set(src)

    def addgrad(self, var):
        """Accumulates the gradient array from given source variable.

        This method adds the gradient of a given variable to the gradient of
        this variable. The accumulation is even done across the host and
        different devices. If this variable has uninitialized data/grad arrays,
        this method initializes it with the shape of the given variable and
        then accumulates the gradient.

        Args:
            var (Variable): Source variable.

        """
        src = var._grad_var
        if src is None:
            return

        if self.data is None:
            self.initialize(var.shape)
        dst = self._grad_var

        src_dev = cuda.get_device_from_array(src.data)
        dst_dev = cuda.get_device_from_array(self.data)

        if src_dev.id != dst_dev.id:
            src = chainer.functions.copy(src, dst_dev.id)
        self._grad_var = src if dst is None else src + dst

    def set_creator(self, gen_func):
        """Notifies the variable that the given function is its creator.

        Args:
            gen_func (Function): Function object that creates this variable as
                one of its outputs.

        """
        self._node.set_creator(gen_func)

    def set_creator_node(self, fnode):
        """Notifies the variable that the given node is its creator.

        Args:
            fnode (FunctionNode): Function node that has this variable as an
                output.

        """
        self._node.set_creator_node(fnode)

    def interrupt_backward(self):
        """Cuts a link to my creator function temporarily.

        Originally from anaruse's repository
        Source: https://github.com/anaruse/
                chainer/blob/OOC_chainer_v202/chainer/variable.py
        """
        self._node.interrupt_backward()

    def resume_backward(self):
        """Recovers a link to my creator function.

        Originally from anaruse's repository
        Source: https://github.com/anaruse/
                chainer/blob/OOC_chainer_v202/chainer/variable.py
        """
        self._node.resume_backward()

    def backward(self, retain_grad=False, enable_double_backprop=False):
        """Wrapper backward function for OOC/LWR

        Original backward function is renamed as _backward

        Originally from anaruse's repository
        Source: https://github.com/anaruse/
                chainer/blob/OOC_chainer_v202/chainer/variable.py
        """
        root_node = self.node

        ooc_enabled, ooc_async, fine_granularity, streams, events, ooc_debug = getattr(
            configuration.config, 'out_of_core_params',
            [False, True, False, [None, None], [], False])

        if ooc_enabled:
            streams[0].synchronize()
            while events:
                events.pop(0).synchronize()

        #if memory_pool.get_profile_mode():
        memory_pool.memory_log_add(("forward_to_backward", ))

        # [OOC/LWR]
        break_points = self.node.get_break_points(fine_granularity)
        if ooc_debug:
            print('# break_points: {}'.format(break_points))

        # Create queue for swapped bp
        bp_swapin_heap = copy.copy(break_points)
        events_swapin = []
        
        # Create events for threading
        thread_events = len(break_points)*[None]
        for i in range(len(break_points)):
            thread_events[i] = threading.Event()
            thread_events[i].clear()
        
        # Thread to schedule swap-in task
        def schedule_swapin(var, id):
            if id > 0:
                thread_events[id-1].wait()
            var.ancestors_swapin(stream=streams[0], inclusive=True, debug=ooc_debug)
            events_swapin.append(streams[0].record())
            streams[0].synchronize()
            thread_events[id].set()
        
        _, _, bp = heapq.heappop(break_points)
        backward_task_count = 0
        swapin_task_count = 0
        while bp is not None:
            if ooc_debug is True:
                """
                print('#    total_bytes: {}'.format(memory_pool.total_bytes()))
                print('#     free_bytes: {}'.format(memory_pool.free_bytes()))
                print('#     used_bytes: {}'.format(memory_pool.used_bytes()))
                root_node._show_memory_usage()
                """
            # swapin
            if ooc_enabled:
                if memory_pool.get_profile_mode():
                    memory_pool.memory_log_add(("swapin_timing", ))
                
                # for profiling step
                if len(swapin_counts) == 0:
                    swapin_counts.append(1)
                
                swapin_count = swapin_counts.pop(0)         
                for count in range(swapin_count):
                    if len(bp_swapin_heap) > 0:
                        _, _, bp_swapin = heapq.heappop(bp_swapin_heap)
                             
                        swapin_bytes = bp_swapin.ancestors_swapin_bytes(stream=streams[0], inclusive=True, debug=ooc_debug)
                        if swapin_bytes > 0:
                            #print(len(bp_swapin_heap), swapin_bytes)
                            swapin_thread = threading.Thread(target=schedule_swapin, args=(bp_swapin, swapin_task_count))
                            swapin_thread.start()
                            #bp_swapin.ancestors_swapin(stream=streams[0], inclusive=True, debug=ooc_debug)
                            #events_swapin.append(streams[0].record())
                            #thread_events[swapin_task_count].set()
                        else:
                            events_swapin.append(None)
                            thread_events[swapin_task_count].set()

                        swapin_task_count += 1

            if ooc_async is False:
                cuda.Stream.null.synchronize()
                  
            thread_events[backward_task_count].wait()
            backward_task_count += 1
            if ooc_enabled and (len(events_swapin) > 0):
                event_swapin = events_swapin.pop(0)
                if event_swapin is not None:
                    # events_swapin.pop(0).synchronize()
                    cuda.Stream.null.wait_event(event_swapin)

            bp.resume_backward()
            bp_var = bp.get_variable()
            if bp_var._grad_var is None:
                bp_var._grad_var = bp._grad_var
            bp_var._backward(retain_grad, enable_double_backprop, root_node)

            if ooc_enabled:
                cuda.Stream.null.synchronize()
                # streams[1].wait_event(cuda.Stream.null.record())
                bp.ancestors_free()
               
            if ooc_async is False:
                cuda.Stream.null.synchronize()
    
            bp = None
            if break_points:
                _, _, bp = heapq.heappop(break_points)
            
        if ooc_enabled:
            streams[0].synchronize()
            streams[1].synchronize()


    def _backward(self, retain_grad, enable_double_backprop, root_node):
        """Runs error backpropagation (a.k.a.\\  backprop) from this variable.

        On backprop, :meth:`FunctionNode.backward` is called on each
        :class:`FunctionNode` object appearing in the backward graph starting
        from this variable. The backward graph is represented by backward
        references from variable nodes to their creators, and from function
        nodes to their input variable nodes. The backprop stops at all root
        nodes. Some function nodes set ``None`` as gradients of some inputs,
        where further backprop does not take place at such inputs.

        This method uses :data:`grad` as the initial error array. User can
        manually set a gradient array before calling this method. If
        :data:`data` contains only one element (i.e., it is scalar) and
        :data:`grad` is ``None``, then this method automatically complements
        1.0 as the initial error. This is useful on starting backprop from
        some scalar loss value.

        Note that this method does not support *differentiable backprop*. Use
        :func:`grad` to compute the gradient of gradients.

        Args:
            retain_grad (bool): If ``True``, the gradient arrays of all
                intermediate variables are kept. Otherwise, :data:`grad` of the
                intermediate variables are set to ``None`` on appropriate
                timing, which may reduce the maximum memory consumption.

                In most cases of training some models, the purpose of backprop
                is to compute gradients of parameters, not of all variables,
                and therefore it is recommended to set this flag ``False``.
            enable_double_backprop (bool): *(Added in v3.0)* If ``True``,
                computational trace of the whole backpropagation procedure is
                recorded to the computational graph so that one can further do
                backpropagation from the resulting gradients. Note that
                enabling it results in larger memory consumption needed to
                store the gradients w.r.t intermediate variables that are
                required for the second gradient computation.

        """
        with chainer.using_config('enable_backprop', enable_double_backprop):
            self._backward_main(retain_grad, root_node)

    def _backward_main(self, retain_grad, root_node):
        self._node._check_old_style_gradient()
        if self.creator_node is None:
            return
        initial_device = None
        if cuda.available and isinstance(self.data, cuda.cupy.ndarray):
            try:
                initial_device = cuda.Device()
            except cuda.cupy.cuda.runtime.CUDARuntimeError as e:
                if e.status != 38:  # cudaErrorNoDevice
                    raise

        is_debug = chainer.is_debug()

        cand_funcs = []
        seen_set = set()
        grads = {}

        # Initialize error by 1, if this is a loss variable
        if self.data_size == 1 and self._grad_var is None:
            with cuda.get_device_from_array(self.data) as device:
                if device is cuda.DummyDevice:
                    self.grad = numpy.ones_like(self.data)
                else:
                    self.grad = cuda.cupy.ones_like(self.data)
        grads[self._node] = self._grad_var

        def add_cand(cand):
            if cand not in seen_set:
                # Negate since heapq is min-heap
                heapq.heappush(cand_funcs, (-cand.rank, len(seen_set), cand))
                seen_set.add(cand)

        add_cand(self.creator_node)

        def get_grad(node):
            if node is None:
                return None
            if node in grads:
                return grads[node]
            return node.grad_var

        # depthはあとで削除(debug用に)
        def recompute_data(vnode, depth=0):
            if vnode.data is not None:
                return vnode.data
            
            tmp_func = vnode._creator_node_g
            #print(depth, str(vnode.data.__class__), vnode.is_target)
            #print(str(tmp_func))
            tmp_inputs = tmp_func.inputs
            tmp_in_data = tuple([recompute_data(x, depth+1) for x in tmp_inputs])

            #print("recompute: ", str(tmp_func))
            tmp_outputs = [y() for y in tmp_func.outputs]
            output_index = tmp_outputs.index(vnode)
            tmp_out_data = tmp_func.forward(tmp_in_data)[output_index]

            gc.collect()
            return tmp_out_data

        while cand_funcs:
            _, _, func = heapq.heappop(cand_funcs)
            inputs = func.inputs
            target_input_indexes = [
                i for i, x in enumerate(inputs) if x.requires_grad
            ]
            if not target_input_indexes:
                continue
            outputs = [y() for y in func.outputs]  # access via weak ref

            # recompute for inputs and outputs
            #print("recompute inputs: ", func._input_indexes_to_retain)
            if func._input_indexes_to_retain is not None:
                for index in func._input_indexes_to_retain:
                    inputs[index].data = recompute_data(inputs[index])
            #print("recompute outputs: ", func._output_indexes_to_retain)
            if func._output_indexes_to_retain is not None:
                retained_data = []
                for index in func._output_indexes_to_retain:
                    outputs[index].data = recompute_data(outputs[index])
                    retained_data.append(outputs[index].data)
                func._retained_output_data = tuple(retained_data)

            in_data = tuple([x.data for x in inputs])
            out_grad = tuple([get_grad(y) for y in outputs])
            out_grad_data = tuple(
                [None if g is None else g.data for g in out_grad])
            hooks = chainer.get_function_hooks()
            if func._n_local_function_hooks != 0:
                hooks = collections.OrderedDict(hooks)
                hooks.update(func.local_function_hooks)
            hooks = hooks.values()  # avoid six for performance

            cuda.get_device_from_array(*in_data).use()
            for hook in hooks:
                hook.backward_preprocess(func, in_data, out_grad_data)

            # Collect the current input gradients.
            #
            # Note (Tokui): When the same variable is passed to multiple input
            # slots (e.g. an expression like ``f(x, x)``), it makes the
            # gradient accumulation complicated since the back-propagated
            # gradients w.r.t. the first and second argument should be
            # accumulated to the current gradient w.r.t. the same variable.
            # In this case, the current implementation passes the current
            # gradient only to the first occurrence of the variable in the
            # input tuple and passes ``None`` to the rest of the occurrences.
            # For example, when the input variables are ``(x, x)``, the
            # input gradient passed to the ``backward_accumulate`` method is
            # ``(gx, None)`` where ``gx`` is the current gradient of ``x``.
            # See also the docstring of ``FunctionNode.backward_accumulate``.
            target_inputs = [inputs[i] for i in target_input_indexes]
            in_grad = []
            for i, index_i in enumerate(target_input_indexes):
                x = inputs[index_i]
                if x in target_inputs[:i]:
                    # Pass ``None`` for duplicated input variables except for
                    # the first occurrence (see the comment above).
                    gx = None
                elif x in grads:
                    gx = grads[x]
                elif x.creator_node is None:
                    x._check_old_style_gradient()
                    # accumulate the gradient if the node is a leaf
                    gx = x.grad_var
                    # or the node is a break point [OOC/LWR]
                    if gx is None and x._break_point:
                        gx = x._grad_var
                else:
                    gx = None
                in_grad.append(gx)
            
            #print("backward: ", str(func), [x.data.__class__.__name__ for x in func.inputs], [y().data.__class__.__name__ for y in func.outputs])
            gxs = func.backward_accumulate(
                target_input_indexes, out_grad, in_grad)
            

            assert len(gxs) == len(in_grad)
            for hook in hooks:
                hook.backward_postprocess(func, in_data, out_grad_data)

            if is_debug:
                for gx in gxs:
                    if gx is None:
                        continue
                    gx_data = gx.data
                    if gx_data.dtype.kind == 'f':
                        cuda.get_device_from_array(gx_data).use()
                        if cuda.get_array_module(gx_data).isnan(gx_data).any():
                            raise RuntimeError(
                                'NaN is detected on backward computation of '
                                '{}'.format(func.label))

            if not retain_grad:
                for y in outputs:
                    if y is not None and y is not root_node:
                        grads[y] = None
                        y_var = y.get_variable()
                        if y_var is not None:
                            y_var._grad_var = None
                        if y._break_point:
                            # [OOC/LWR] delete grad
                            y._grad_var = None

            for i, gx in enumerate(gxs):
                if gx is None:
                    continue

                x = target_inputs[i]
                if not x.requires_grad:
                    continue

                _check_grad_type(func, x, gx.data)

                if x in target_inputs[:i]:
                    # Accumulate the duplicated gradients here. See the comment
                    # above the code that builds ``in_grad``.
                    cur_gx = grads[x]
                    grads[x] = gx if cur_gx is None else gx + cur_gx
                else:
                    grads[x] = gx

                x_var = x.get_variable()
                if x_var is not None:
                    if x._break_point:
                        # [OOC/LWR] to reconstruct its variable
                        x._grad_var = grads[x]
                    x_var._grad_var = grads[x]

                if x.creator_node is not None:
                    add_cand(x.creator_node)

            del gxs  # to reduce memory usage
            if initial_device is not None:
                initial_device.use()

    def reshape(self, *shape):
        """Returns a variable of a different shape and the same content.

        .. seealso::
           :func:`chainer.functions.reshape` for full documentation,

        """
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        return chainer.functions.reshape(self, shape)

    def transpose(self, *axes):
        """Permute the dimensions of an input variable without copy.

        .. seealso::
           :func:`chainer.functions.transpose` for full documentation.

        """
        if len(axes) == 0:
            axes = None
        elif len(axes) == 1 and (isinstance(axes[0], (tuple, list)) or
                                 axes[0] is None):
            axes = axes[0]
        return chainer.functions.transpose(self, axes)

    def unchain(self):
        """Deletes the reference to the creator of this variable.

        This method deletes the reference to the creator from the corresponding
        variable node. Unlike :meth:`unchain_backward`, it does not backtrack
        the graph.

        This method is equivalent to ``self.creator_node = None``.

        """
        self.creator_node = None

    def unchain_backward(self):
        """Deletes references between variable nodes and functions backward.

        After this method completes, intermediate variable nodes and functions
        that are not referenced from anywhere are deallocated by reference
        count GC. Also this variable itself deletes the reference to its
        creator function from the node, i.e. the node becomes root in the
        computation graph. It indicates that backprop after unchaining stops at
        this variable. This behavior is useful to implement truncated BPTT.

        """
        cand_funcs = []
        seen_set = set()

        def add_cand(cand):
            if cand is not None and cand not in seen_set:
                cand_funcs.append(cand)
                seen_set.add(cand)

        add_cand(self.creator_node)

        while cand_funcs:
            func = cand_funcs.pop()
            for var in func.inputs:
                add_cand(var.creator_node)
            func.unchain()

    def retain_data(self):
        """Lets the corresponding variable node keep the underlying array."""
        self._node.data = self._data[0]

    def __lt__(self, other):
        raise NotImplementedError()

    def __le__(self, other):
        raise NotImplementedError()

    def __eq__(self, other):
        raise NotImplementedError()

    def __ne__(self, other):
        raise NotImplementedError()

    def __gt__(self, other):
        raise NotImplementedError()

    def __ge__(self, other):
        raise NotImplementedError()

    def __nonzero__(self):
        raise NotImplementedError()

    def __bool__(self):
        raise NotImplementedError()

    __array_priority__ = 200
    __hash__ = None


class Parameter(Variable):

    """Parameter variable that can be registered to a link.

    Parameter is a subclass of :class:`Variable`. It almost behaves as same
    as a usual variable except that a parameter can be registered to a
    :class:`~chainer.Link` object just by assigning it to an attribute of
    the link within an :meth:`~chainer.Link.init_scope` context.

    Parameter also supports an initialization by an initializer. It can have
    two initializers: one for the data array, and the other for the gradient
    array. The initializer only specifies the way of filling the elements of
    these arrays, and the shape information is specified at the initialization
    point.

    When a link that the parameter has been registered to is passed to an
    :class:`~chainer.GradientMethod`, an update rule is set to the parameter.
    This update rule specifies how to update the data array of the parameter
    using its gradient array.

    Args:
        initializer (~chainer.Initializer or numpy.ndarray or cupy.ndarray):
            Initializer of the data array. If ``shape`` is given, this
            initializer is immediately used to initialize the data array.
            Otherwise, if it is an array, it is immediately used as the data
            array, and otherwise the data array is left uninitialized and will
            be initialized by this initializer in :meth:`initialize`. It can
            also be a scalar, in which case the data array will be filled by
            this scalar. Note that float32 is used in this case.
        shape (int or tuple of int or None): Shape of the parameter. If it is
            ``None``, the initialization is deferred to the call of
            :meth:`initialize`.
        name (str): Name of the parameter.

    Attributes:
        initializer: Initializer of the data array. It is used for
            initializing the data array of an uninitialized variable.
        update_rule: :class:`~chainer.optimizer.UpdateRule` instance that
            updates this variable as a parameter. This argument is set to
            :attr:`update_rule`.

    """

    initializer = None
    _grad_initializer = None
    _initial_device = None

    def __init__(self, initializer=None, shape=None, name=None):
        if initializer is None:
            initializer = constant.NaN()
        elif numpy.isscalar(initializer):
            initializer = constant.Constant(initializer)
        if shape is None:
            if isinstance(initializer, (numpy.ndarray, cuda.ndarray)):
                # parameter initialized by the initial array
                super(Parameter, self).__init__(initializer, name=name)
            else:
                # uninitialized parameter
                super(Parameter, self).__init__(name=name)
                self.initializer = initializer
                dtype = getattr(initializer, 'dtype', numpy.float32)
                self._grad_initializer = constant.NaN(dtype)
        else:
            # parameter initialized with a given shape
            if isinstance(initializer, (numpy.ndarray, cuda.ndarray)):
                xp = cuda.get_array_module(initializer)
                initializer = constant.Constant(initializer)
            else:
                xp = numpy
            data = initializers.generate_array(initializer, shape, xp)
            grad = xp.full_like(data, numpy.nan)
            super(Parameter, self).__init__(data, name=name, grad=grad)

        self.update_rule = None

    def __copy__(self):
        return self._copy_to(Parameter())

    def __reduce__(self):
        return _recover_parameter, (self.data, self.name, self.grad,
                                    self.initializer, self.update_rule)

    def to_cpu(self):
        super(Parameter, self).to_cpu()
        if self.data is None:
            self._initial_device = None

    def to_gpu(self, device=None):
        super(Parameter, self).to_gpu(device)
        if self.data is None:
            if device is None:
                device = cuda.Device().id
            self._initial_device = device

    def cleargrad(self):
        super(Parameter, self).cleargrad()
        if self.data is None:
            self._grad_initializer = None

    def zerograd(self):
        super(Parameter, self).zerograd()
        if self.data is None:
            dtype = getattr(self.initializer, 'dtype', None)
            self._grad_initializer = initializers.Zero(dtype)

    def initialize(self, shape):
        """Initializes the uninitialized variable.

        Uninitialized variable is a variable created with the data array set to
        None. This method creates and initializes the data array. The shape of
        the variable can be left unknown until this method is called.

        Args:
            shape (tuple of int): Shape of the data array.

        """
        xp = numpy if self._initial_device is None else cuda.cupy
        with cuda.get_device_from_id(self._initial_device):
            data = initializers.generate_array(self.initializer, shape, xp)

            ginit = self._grad_initializer
            grad = None if ginit is None else initializers.generate_array(
                ginit, shape, xp)

        self.data = data
        self.grad = grad

    def update(self):
        """Updates the data array using the gradient and the update rule.

        This method updates the parameter using the attached update rule.

        """
        if self.update_rule is not None:
            self.update_rule.update(self)


def as_variable(obj):
    """Converts an array or a variable into :class:`~chainer.Variable`.

    This is a convenient function to get a :class:`~chainer.Variable` object
    transparently from a raw array or a variable.

    Note that this function should only be used for type consistency (i.e., to
    enforce the return value of an API having type :class:`~chainer.Varialbe`).
    The :class:`~chainer.Variable.requires_grad` flag is kept as is; if ``obj``
    is a raw array, the newly created variable has ``requires_grad = False``.
    In order to make a variable w.r.t. which you want to compute the gradient,
    you should use :class:`~chainer.Variable` directly.

    Args:
        obj (numpy.ndarray or cupy.ndarray or ~chainer.Variable): An array or
            a variable that you want to convert to :class:`~chainer.Variable`.

    Returns:
        ~chainer.Variable:
        A variable converted from ``obj``. If ``obj`` is a raw array, this is a
        new :class:`~chainer.Variable` object that wraps the array. If ``obj``
        is already a :class:`~chainer.Variable` object, this function returns
        ``obj`` as is.

    """
    if isinstance(obj, Variable):
        return obj
    return Variable(obj, requires_grad=False)


def _recover_parameter(data, name, grad, initializer, update_rule):
    p = Parameter(initializer=initializer, name=name)
    p.data = data
    p.grad = grad
    p.update_rule = update_rule
    return p
