from __future__ import print_function

import collections
import os
import sys
import time
import traceback
import itertools
import copy
import gc

import six

from chainer import reporter as reporter_module
from chainer import serializer as serializer_module
from chainer.training import extension as extension_module
from chainer.training import trigger as trigger_module
from chainer.utils import argument
from chainer import cuda
from chainer import variable, function_node
from chainer.cuda import memory_pool
from chainer import configuration
# Select the best-resolution timer function
try:
    _get_time = time.perf_counter
except AttributeError:
    if os.name == 'nt':
        _get_time = time.clock
    else:
        _get_time = time.time

class _ExtensionEntry(object):

    def __init__(self, extension, priority, trigger):
        self.extension = extension
        self.trigger = trigger
        self.priority = priority


class Trainer(object):

    """The standard training loop in Chainer.

    Trainer is an implementation of a training loop. Users can invoke the
    training by calling the :meth:`run` method.

    Each iteration of the training loop proceeds as follows.

    - Update of the parameters. It includes the mini-batch loading, forward
      and backward computations, and an execution of the update formula.
      These are all done by the update object held by the trainer.
    - Invocation of trainer extensions in the descending order of their
      priorities. A trigger object is attached to each extension, and it
      decides at each iteration whether the extension should be executed.
      Trigger objects are callable objects that take the trainer object as the
      argument and return a boolean value indicating whether the extension
      should be called or not.

    Extensions are callable objects that take the trainer object as the
    argument. There are three ways to define custom extensions: inheriting the
    :class:`Extension` class, decorating functions by :func:`make_extension`,
    and defining any callable including lambda functions. See
    :class:`Extension` for more details on custom extensions and how to
    configure them.

    Users can register extensions to the trainer by calling the :meth:`extend`
    method, where some configurations can be added.

    - Trigger object, which is also explained above. In most cases,
      :class:`IntervalTrigger` is used, in which case users can simply specify
      a tuple of the interval length and its unit, like
      ``(1000, 'iteration')`` or ``(1, 'epoch')``.
    - The order of execution of extensions is determined by their priorities.
      Extensions of higher priorities are invoked earlier. There are three
      standard values for the priorities:

      - ``PRIORITY_WRITER``. This is the priority for extensions that write
        some records to the :attr:`observation` dictionary. It includes cases
        that the extension directly adds values to the observation dictionary,
        or the extension uses the :func:`chainer.report` function to report
        values to the observation dictionary.
      - ``PRIORITY_EDITOR``. This is the priority for extensions that edit the
        :attr:`observation` dictionary based on already reported values.
      - ``PRIORITY_READER``. This is the priority for extensions that only read
        records from the :attr:`observation` dictionary. This is also suitable
        for extensions that do not use the :attr:`observation` dictionary at
        all.

    The current state of the trainer object and objects handled by the trainer
    can be serialized through the standard serialization protocol of Chainer.
    It enables us to easily suspend and resume the training loop.

    .. note::
       The serialization does not recover everything of the training loop. It
       only recovers the states which change over the training (e.g.
       parameters, optimizer states, the batch iterator state, extension
       states, etc.). You must initialize the objects correctly before
       deserializing the states.

       On the other hand, it means that users can change the settings on
       deserialization. For example, the exit condition can be changed on the
       deserialization, so users can train the model for some iterations,
       suspend it, and then resume it with larger number of total iterations.

    During the training, it also creates a :class:`~chainer.Reporter` object to
    store observed values on each update. For each iteration, it creates a
    fresh observation dictionary and stores it in the :attr:`observation`
    attribute.

    Links of the target model of each optimizer are registered to the reporter
    object as observers, where the name of each observer is constructed as the
    format ``<optimizer name><link name>``. The link name is given by the
    :meth:`chainer.Link.namedlink` method, which represents the path to each
    link in the hierarchy. Other observers can be registered by accessing the
    reporter object via the :attr:`reporter` attribute.

    The default trainer is `plain`, i.e., it does not contain any extensions.

    Args:
        updater (~chainer.training.Updater): Updater object. It defines how to
            update the models.
        stop_trigger: Trigger that determines when to stop the training loop.
            If it is not callable, it is passed to :class:`IntervalTrigger`.

    Attributes:
        updater: The updater object for this trainer.
        stop_trigger: Trigger that determines when to stop the training loop.
            The training loop stops at the iteration on which this trigger
            returns ``True``.
        observation: Observation of values made at the last update. See the
            :class:`Reporter` class for details.
        out: Output directory.
        reporter: Reporter object to report observed values.

    """

    def __init__(self, updater, stop_trigger=None, out='result'):
        self.updater = updater
        self.stop_trigger = trigger_module.get_trigger(stop_trigger)
        self.observation = {}
        self.out = out

        reporter = reporter_module.Reporter()
        for name, optimizer in six.iteritems(updater.get_all_optimizers()):
            reporter.add_observer(name, optimizer.target)
            reporter.add_observers(
                name, optimizer.target.namedlinks(skipself=True))
        self.reporter = reporter

        self._done = False
        self._extensions = collections.OrderedDict()

        self._start_at = None
        self._snapshot_elapsed_time = 0.0
        self._final_elapsed_time = None
        
        self.is_targets = list()
        self.swapin_counts = list()

        updater.connect_trainer(self)

    @property
    def elapsed_time(self):
        """Total time used for the training.

        The time is in seconds. If the training is resumed from snapshot, it
        includes the time of all the previous training to get the current
        state of the trainer.

        """
        if self._done:
            return self._final_elapsed_time
        if self._start_at is None:
            raise RuntimeError('training has not been started yet')
        return _get_time() - self._start_at + self._snapshot_elapsed_time

    def extend(self, extension, name=None, trigger=None, priority=None,
               **kwargs):
        """Registers an extension to the trainer.

        :class:`Extension` is a callable object which is called after each
        update unless the corresponding trigger object decides to skip the
        iteration. The order of execution is determined by priorities:
        extensions with higher priorities are called earlier in each iteration.
        Extensions with the same priority are invoked in the order of
        registrations.

        If two or more extensions with the same name are registered, suffixes
        are added to the names of the second to last extensions. The suffix is
        ``_N`` where N is the ordinal of the extensions.

        See :class:`Extension` for the interface of extensions.

        Args:
            extension: Extension to register.
            name (str): Name of the extension. If it is omitted, the
                ``default_name`` attribute of the extension is used instead.
                Note that the name would be suffixed by an ordinal in case of
                duplicated names as explained above.
            trigger (tuple or Trigger): Trigger object that determines when to
                invoke the extension. If it is ``None``, ``extension.trigger``
                is used instead. If it is ``None`` and the extension does not
                have the trigger attribute, the extension is triggered at every
                iteration by default. If the trigger is not callable, it is
                passed to :class:`IntervalTrigger` to build an interval
                trigger.
            priority (int): Invocation priority of the extension. Extensions
                are invoked in the descending order of priorities in each
                iteration. If this is ``None``, ``extension.priority`` is used
                instead.

        """
        argument.check_unexpected_kwargs(
            kwargs,
            invoke_before_training='invoke_before_training has been removed '
            'since Chainer v2.0.0. Use initializer= instead.')
        argument.assert_kwargs_empty(kwargs)

        if name is None:
            name = getattr(extension, 'name', None)
            if name is None:
                name = getattr(extension, 'default_name', None)
                if name is None:
                    name = getattr(extension, '__name__', None)
                    if name is None:
                        raise TypeError('name is not given for the extension')
        if name == 'training':
            raise ValueError(
                'the name "training" is prohibited as an extension name')

        if trigger is None:
            trigger = getattr(extension, 'trigger', (1, 'iteration'))
        trigger = trigger_module.get_trigger(trigger)

        if priority is None:
            priority = getattr(
                extension, 'priority', extension_module.PRIORITY_READER)

        modified_name = name
        ordinal = 0
        while modified_name in self._extensions:
            ordinal += 1
            modified_name = '%s_%d' % (name, ordinal)

        extension.name = modified_name
        self._extensions[modified_name] = _ExtensionEntry(
            extension, priority, trigger)

    def get_extension(self, name):
        """Returns the extension of a given name.

        Args:
            name (str): Name of the extension.

        Returns:
            Extension.

        """
        extensions = self._extensions
        if name in extensions:
            return extensions[name].extension
        else:
            raise ValueError('extension %s not found' % name)
    
    def make_log_list(self):

        tmp_list, swap_events = memory_pool.memory_log_get()
        
        exsit_swapin_items = list()
        exsit_swapout_items = collections.defaultdict(lambda: None)
        
        target_vars = list()
        
        # collect compute time information
        compute_times = {}
        while len(function_node.compute_events) > 0:
            item = function_node.compute_events.pop()
            item[2].synchronize()
            item[3].synchronize()
            compute_times[(item[0], item[1])] = cuda.cupy.cuda.get_elapsed_time(item[2], item[3])
        
        # collect swap time information
        swap_times = list()
        while len(swap_events) > 0:
            item = swap_events.pop(0)
            item[0].synchronize()
            item[1].synchronize()
            swap_times.append(cuda.cupy.cuda.get_elapsed_time(item[0], item[1]))
        
        # make log_list
        log_list = list()
        new_item = tmp_list.pop(0)

        memory_event_count = 0
        memory_events = {}
        
        while len(tmp_list) > 0:
            current_item = tmp_list.pop(0)

            if current_item[0] == "malloc":
                if len(exsit_swapin_items) > 0:
                    exsit_swapin_item = exsit_swapin_items.pop(0)
                    new_item = ("swapin", exsit_swapin_item[1], exsit_swapin_item[2], swap_times.pop(0), current_item[3])
                else:
                    log_list.append(new_item)
                    new_item = ("malloc", current_item[1], current_item[3])
            elif current_item[0] == "free":
                if exsit_swapout_items[current_item[2]] is not None:
                    log_list.append(new_item)
                    new_item = ("swapout_free", current_item[1], exsit_swapout_items[current_item[2]], current_item[3])
                    exsit_swapout_items[current_item[2]] = None
                else:
                    log_list.append(new_item)
                    new_item = ("free", current_item[1], current_item[3])
            elif current_item[0] == "swapin":
                log_list.append(new_item)
                exsit_swapin_items.append(current_item)
            elif current_item[0] == "swapout":
                log_list.append(new_item)
                new_item = ("swapout", current_item[1], current_item[2], swap_times.pop(0))
                exsit_swapout_items[current_item[3]] = current_item[2]
                target_vars.append(current_item[2])
            elif current_item[0] == "forward" or current_item[0] == "backward":
                if ("forward", current_item[1]) in compute_times.keys() and ("backward", current_item[1]) in compute_times.keys(): 
                    log_list.append(new_item)
                    new_item = (current_item[0], current_item[1], compute_times[(current_item[0], current_item[1])])
            else:
                log_list.append(new_item)
                new_item = current_item
        log_list.append(new_item)
        
        return log_list, target_vars
        
    def optimize_targets(self):
        
        if memory_pool.get_profile_mode():
            log_list, target_vars = self.make_log_list()
            total_mem_size = memory_pool.total_bytes()
            
            #for i in log_list:
            #    print(i)

            compute_graph = function_node.compute_graph
            var_sizes = function_node.var_sizes
            func_backward_use_vars = function_node.func_backward_use_vars

            #print("log_list = ", log_list)
            #print("target_vars = ", target_vars)
            #print("compute_graph = ", compute_graph)
            #print("var_sizes = ", var_sizes)
            #print("func_backward_use_vars = ", func_backward_use_vars)
            
            # filter memory_items
            forward_memory_items = list(filter(lambda item:item[0] in {"used_bytes", "malloc", "free", "swapout", "forward", "forward_to_backward"}, log_list))
            backward_memory_items = list(filter(lambda item:item[0] in {"malloc", "free", "swapin_timing", "forward_to_backward"}, log_list))
            while backward_memory_items[0][0] != "forward_to_backward":
                backward_memory_items.pop(0)

            # make forward_times (for recompute)
            forward_times = collections.defaultdict(lambda: 0.0)
            for item in log_list:
                if item[0] == "forward":
                    forward_times[item[1]] = item[2]
                    
            # make forward_workspace_sizes (for recompute)
            forward_workspace_sizes = collections.defaultdict(lambda: 0)
            current_layer = None

            for item in log_list:
                if item[0] == "malloc":
                    if current_layer is not None:
                        forward_workspace_sizes[current_layer] += item[1]
                elif item[0] == "forward":
                    current_layer = item[1]
                    forward_workspace_sizes[current_layer] = 0    
                elif item[0] == "forward_to_backward":
                    break

            # filter backward_items
            backward_items = list(filter(lambda item:item[0] in {"backward", "swapin_timing"}, log_list))
            
            # filter swapout_items, swapin_items
            tmp_list = list(filter(lambda item:item[0] in {"swapout"}, log_list))
            swapout_items = {}
            for item in tmp_list:
                swapout_items[item[2]] = (item[1], item[3])

            tmp_list = list(filter(lambda item:item[0] in {"swapin"}, log_list))
            swapin_items = {}
            for item in tmp_list:
                swapin_items[item[2]] = (item[1], item[3])

            num_swapin_timing = len(list(filter(lambda item:item[0] in {"swapin_timing"}, backward_items)))

            # make var_is_retained
            var_is_retained = dict(zip(var_sizes.keys(), len(var_sizes.keys())*[False]))
            for use_vars in func_backward_use_vars.values():
                for var in use_vars:
                    var_is_retained[var] = True


            def sync_swapout(targets):            
                compute_time = 0.0
                swap_time = 0.0
                swap_tasks = list()
                swapout_sync_vars = list()

                current_memory_usage = 0
                max_memory_usage = 0

                # simulate forward compute time and memory management
                for memory_item in forward_memory_items:
                    if memory_item[0] == "malloc":
                        current_memory_usage += memory_item[1]
                        # swapout_free when over memory capacity and synchronize
                        while swap_tasks and (current_memory_usage > total_mem_size):
                            var = swap_tasks[0][0]
                            if compute_time < swap_tasks[0][1]:
                                compute_time = swap_tasks[0][1]
                            current_memory_usage -= var_sizes[var]
                            swapout_sync_vars.append(var)
                            swap_tasks.pop(0)
                    elif memory_item[0] == "free":
                        current_memory_usage -= memory_item[1]
                    elif memory_item[0] == "swapout":
                        var = memory_item[2]
                        if targets[var] == "swap":
                            swap_time = max(compute_time, swap_time) + swapout_items[var][1]
                            swap_tasks.append((var, swap_time))
                        elif targets[var] == "recompute":
                            current_memory_usage -= var_sizes[var]
                    elif memory_item[0] == "forward":
                        # swapout_free when swapout has been finished
                        while swap_tasks and (swap_tasks[0][1] < compute_time):
                            var = swap_tasks[0][0]
                            current_memory_usage -= var_sizes[var]
                            swap_tasks.pop(0)
                        compute_time += memory_item[2]
                    elif memory_item[0] == "used_bytes":
                        current_memory_usage = memory_item[1]
                    elif memory_item[0] == "forward_to_backward":
                        # swapout_free before backward
                        while swap_tasks:
                            var = swap_tasks[0][0]
                            current_memory_usage -= var_sizes[var]
                            swapout_sync_vars.append(var)
                            swap_tasks.pop(0)
                        break

                    if current_memory_usage > max_memory_usage:
                        max_memory_usage = current_memory_usage
                    if max_memory_usage > total_mem_size:
                        return float('inf'), float('inf'), None
                    
                return current_memory_usage, max(compute_time, swap_time), swapout_sync_vars


            def compact_swapin(current_memory_usage, swapin_counts, swapin_sizes, recompute_sizes):
            
                max_memory_usage = 0
                tmp_val = 0 # record max_memory_usage between swapin_timing
                tmp_memory_usage = list()

                swapin_timing_count = -1
                swapin_task_count = 0

                # simulate memory management
                for memory_item in backward_memory_items:
                    if memory_item[0] == "malloc":
                        current_memory_usage += memory_item[1]
                    elif memory_item[0] == "free":
                        current_memory_usage -= memory_item[1]
                    elif memory_item[0] == "swapin_timing":
                        tmp_memory_usage.append(tmp_val)
                        tmp_val = current_memory_usage
                        swapin_timing_count += 1
                        for j in range(swapin_counts[swapin_timing_count]):
                            current_memory_usage += swapin_sizes[swapin_task_count]
                            swapin_task_count += 1

                        for recompute_size in recompute_sizes[swapin_timing_count]:
                            current_memory_usage += recompute_size
                            if current_memory_usage > tmp_val:
                                tmp_val = current_memory_usage
                            if current_memory_usage > max_memory_usage:
                                max_memory_usage = current_memory_usage
                    elif memory_item[0] == "forward_to_backward":
                        tmp_val = current_memory_usage

                    if current_memory_usage > tmp_val:
                        tmp_val = current_memory_usage
                    if current_memory_usage > max_memory_usage:
                        max_memory_usage = current_memory_usage
                    if max_memory_usage > total_mem_size:
                        return None
                tmp_memory_usage.pop(0)
                tmp_memory_usage.append(tmp_val)

                # compact swapin_items
                new_swapin_counts = num_swapin_timing*[1]
                fast_swapin = False
                for i in range(num_swapin_timing):
                    for j in range(i+1, num_swapin_timing):
                        if new_swapin_counts[j] == 0:
                            break
                        fast_swapin = True
                        for k in range(i, j):
                            if tmp_memory_usage[k]+swapin_sizes[j] > total_mem_size:
                                fast_swapin = False
                                break
                        if fast_swapin:
                            new_swapin_counts[i] += 1
                            new_swapin_counts[j] -= 1
                            for k in range(i, j):
                                tmp_memory_usage[k] += swapin_sizes[j]
                        else:
                            break

                return new_swapin_counts


            def split_backward_timeline(swapin_counts, backward_times, swapin_times):

                swapin_task_count = 0
                sync_times = num_swapin_timing*[0]
                swapin_sync_vars = list()

                # simulate backward timeline
                compute_time = backward_times[0]
                swap_time = 0
                for j in range(swapin_counts[0]):
                    swap_time += swapin_times[swapin_task_count]
                    sync_times[swapin_task_count] = swap_time
                    swapin_task_count += 1

                for i in range(1, num_swapin_timing):

                    if swap_time < compute_time:
                        swap_time = compute_time
                
                    for j in range(swapin_counts[i]):
                        swap_time += swapin_times[swapin_task_count]
                        sync_times[swapin_task_count] = swap_time
                        swapin_task_count += 1
                            
                    if compute_time < sync_times[i]:
                        compute_time = sync_times[i]
                        #swapin_sync_vars.append(i)

                    compute_time += backward_times[i]

                return compute_time, swapin_sync_vars


            def rebuild_compute_graph(targets):
                # calculate backward_times and swapin_times, swapin_sizes, recompute_sizes
                backward_times = num_swapin_timing*[0.0]
                swapin_times = num_swapin_timing*[0.0]
                swapin_sizes = num_swapin_timing*[0]
                recompute_sizes = num_swapin_timing*[None]
                
                timing_count = -1
                current_var_is_retained = dict(var_is_retained)
                for var in target_vars:
                    if targets[var] == "keep":
                        current_var_is_retained[var] = True
                    else:
                        current_var_is_retained[var] = False
                
                for backward_item in backward_items:
                    if backward_item[0] == "swapin_timing":
                        timing_count += 1
                        recompute_sizes[timing_count] = list()
                        continue
                    
                    backward_times[timing_count] += backward_item[2]
                    use_vars = list(func_backward_use_vars[backward_item[1]])
                    
                    while use_vars:
                        var = use_vars.pop(0)

                        swap_flag = (var in target_vars) and (targets[var] == "swap") and not(current_var_is_retained[var])
                        if swap_flag:
                            swapin_times[timing_count] += swapin_items[var][1]
                            swapin_sizes[timing_count] += swapin_items[var][0]
                        elif not(current_var_is_retained[var]):
                            # recompute
                            tmp_func = compute_graph[var]
                            backward_times[timing_count] += forward_times[tmp_func]
                            # memory management for recompute
                            input_var_size = 0
                            for input_var in compute_graph[tmp_func]:
                                if not(current_var_is_retained[input_var]):
                                    if (input_var not in target_vars) or (targets[input_var] != "swap"):
                                        input_var_size += var_sizes[input_var]
                            # malloc "output var and workspace" + free "input var and workspace"
                            tmp_recompute_sizes = [forward_workspace_sizes[tmp_func], var_sizes[var]-forward_workspace_sizes[tmp_func]-input_var_size]
                            recompute_sizes[timing_count] = tmp_recompute_sizes + recompute_sizes[timing_count]

                            use_vars = list(compute_graph[tmp_func]) + use_vars

                        # update current_var_is_retained
                        if swap_flag or (var in func_backward_use_vars[backward_item[1]]):
                            current_var_is_retained[var] = True
                
                return backward_times, swapin_times, swapin_sizes, recompute_sizes


            def simulate_forward_and_backward(targets):
                # simulate forward
                current_memory_usage, forward_total_time, swapout_sync_vars = sync_swapout(targets)
                # simulate backward
                backward_times, swapin_times, swapin_sizes, recompute_sizes = rebuild_compute_graph(targets)
                swapin_counts = compact_swapin(current_memory_usage, num_swapin_timing*[1], swapin_sizes, recompute_sizes)
                if swapin_counts is None:
                    return float("inf"), None, None, None
                backward_total_time, swapin_sync_vars = split_backward_timeline(swapin_counts, backward_times, swapin_times)

                total_time = forward_total_time + backward_total_time
                return total_time, swapin_counts, swapout_sync_vars, swapin_sync_vars


            def optimize_swap_targets_heuristics(selected_vars, targets):

                new_selected_vars = list(selected_vars)
                current_targets = dict(targets)

                total_time, swapin_counts, swapout_sync_vars, swapin_sync_vars = simulate_forward_and_backward(targets)
                if swapin_counts is None:
                    return float('inf'), None, None

                swapin_vars = list()
                # "keep" vs "swap" in backward
                for swapin_var in swapin_vars:
                    if swapin_var in new_selected_vars:
                        continue
                    new_selected_vars.append(swapin_var)
                    
                    if swapin_var in swapin_sync_vars:
                        targets[swapin_var] = "keep"
                        tmp_total_time, tmp_current_targets, tmp_swapin_counts = optimize_swap_targets_heuristics(new_selected_vars, dict(targets))
                        if total_time > tmp_total_time:
                            total_time = tmp_total_time 
                            current_targets = tmp_current_targets
                            swapin_counts = tmp_swapin_counts
                    targets[swapin_var] = "swap"

                # "keep" vs "swap" in forward
                for swapout_var in swapout_sync_vars[::-1]:
                    current_targets[swapout_var] = "keep"
                    _, swapin_counts, _, _ = simulate_forward_and_backward(current_targets)
                    if swapin_counts is None:
                        current_targets[swapout_var] = "swap"
                        break

                swapout_items = list(filter(lambda item:item[0] in {"swapout"}, log_list))
                for item in swapout_items[::-1]:
                    current_targets[item[2]] = "keep"
                    _, swapin_counts, _, _ = simulate_forward_and_backward(current_targets)
                    if swapin_counts is None:
                        current_targets[item[2]] = "swap"
                        
                total_time, swapin_counts, _, _ = simulate_forward_and_backward(current_targets)
                return total_time, current_targets, swapin_counts


            def optimize_recompute_targets_heuristics(targets):

                not_selected_vars = list()
                current_targets = dict(targets)

                for var in target_vars:
                    if targets[var] != "keep":
                        not_selected_vars.append(var)

                # "swap" vs "recompute"
                while not_selected_vars:
                    min_ratio = float('inf')
                    best_var = None
                    remove_vars = list()
                    
                    for var in not_selected_vars:
                        current_targets[var] = "swap"
                        swap_case_time, _, _, _ = simulate_forward_and_backward(current_targets)
                        tmp1 = swapout_items[var]
                        tmp2 = swapin_items[var]
                        swapout_items[var] = (var_sizes[var], 0.0)
                        swapin_items[var] = (var_sizes[var], 0.0)
                        retain_case_time, _, _, _ = simulate_forward_and_backward(current_targets)
                        swap_overhead = swap_case_time - retain_case_time
                        swapout_items[var] = tmp1
                        swapin_items[var] = tmp2
                        
                        current_targets[var] = "keep"
                        retain_case_time = sum(rebuild_compute_graph(current_targets)[0])
                        current_targets[var] = "recompute"
                        _, swapin_counts, _, _ = simulate_forward_and_backward(current_targets)
                        if swapin_counts is not None:
                            recompute_case_time = sum(rebuild_compute_graph(current_targets)[0])
                        else:
                            recompute_case_time = float('inf')
                        recompute_overhead = recompute_case_time - retain_case_time

                        
                        if swap_overhead == 0:
                            remove_vars.append(var)
                        else:
                            tmp_ratio = recompute_overhead / swap_overhead
                            #print(swap_overhead, recompute_overhead, tmp_ratio)
                            if tmp_ratio >= 1.0:
                                remove_vars.append(var)
                            elif min_ratio > tmp_ratio:
                                min_ratio = tmp_ratio
                                best_var = var
                            
                        current_targets[var] = "swap"

                    if best_var is not None:
                        current_targets[best_var] = "recompute"
                        not_selected_vars.remove(best_var)
                    else:
                        break
                    #print(len(not_selected_vars), len(remove_vars))
                    while remove_vars:
                        not_selected_vars.remove(remove_vars.pop())
                
                total_time, swapin_counts, _, _ = simulate_forward_and_backward(current_targets)
                return total_time, current_targets, swapin_counts


            def delay_swapin(targets, current_swapin_counts):
                backward_times, swapin_times, _, _ = rebuild_compute_graph(targets)
            
                swapin_blocks = list()
                swapin_block_start = 0
                swapin_block_end = 0
                for i in range(num_swapin_timing):
                    swapin_count = current_swapin_counts[i]
                    if swapin_count != 0:
                        swapin_block_end = swapin_block_start + swapin_count - 1
                        swapin_blocks.append((swapin_block_start, swapin_block_end, i))
                        swapin_block_start += swapin_count
                
                swapin_counts = num_swapin_timing*[0]
                current_backward = num_swapin_timing - 1
                for swapin_block in swapin_blocks[::-1]:
                    swapin_counts[swapin_block[2]] = swapin_block[1]-swapin_block[0]+1
                    
                    swapin_block_time = 0
                    for j in range(swapin_block[0], swapin_block[1]):
                        swapin_block_time += swapin_times[j]
                    j = swapin_block[1]
                    tmp1 = swapin_times[j]

                    backward_block_time = 0
                    for k in range(swapin_block[2], current_backward):
                        backward_block_time += backward_times[k]
                    k = current_backward
                    tmp2 = backward_times[k]

                    while j >= swapin_block[0] and k >= swapin_block[2]:
                        while k >= j:
                            tmp1 = swapin_times[j]
                            k -= 1
                            tmp2 = backward_times[k]
                            backward_block_time -= backward_times[k]
                        
                        if tmp1 <= tmp2:
                            if swapin_block_time <= backward_block_time:
                                swapin_counts[k] += 1
                                swapin_counts[swapin_block[2]] -= 1
                            j -= 1
                            tmp1 += swapin_times[j]
                            swapin_block_time -= swapin_times[j]
                        elif swapin_block_time >= backward_block_time:
                            swapin_counts[k] += 1
                            swapin_counts[swapin_block[2]] -= 1
                            j -= 1
                            tmp1 += swapin_times[j]
                            swapin_block_time -= swapin_times[j]
                        else:
                            k -= 1
                            tmp2 += backward_times[k]
                            backward_block_time -= backward_times[k]
                            
                    current_backward = k-1
                            
                return swapin_counts

            def optimize_targets_superneurons():
                targets = dict(zip(target_vars, len(target_vars)*["swap"]))
                swapout_items = list(filter(lambda item:item[0] in {"swapout"}, log_list))
                for item in swapout_items[::-1]:
                    targets[item[2]] = "keep"
                    _, swapin_counts, _, _ = simulate_forward_and_backward(targets)
                    if swapin_counts is None:
                        targets[item[2]] = "swap"
                        break

                for var in target_vars:
                    if targets[var] == "swap":
                        if compute_graph[var].count("convolution") == 0 and compute_graph[var].count("FunctionAdapter") == 0:
                            targets[var] = "recompute"
                
                backward_items = list(filter(lambda item:item[0] in {"backward", "swapin_timing"}, log_list))
                conv_layers = list()
                timing_count = -1
                for item in backward_items:
                    if item[0] == "swapin_timing":
                        timing_count += 1
                    elif item[0] == "backward":
                        if item[1].count("convolution") > 0 or item[1].count("FunctionAdapter") > 0:
                            if len(conv_layers) > 0 and conv_layers[-1] == timing_count:
                                continue
                            conv_layers.append(timing_count)
                conv_layers.append(num_swapin_timing)

                swapin_counts = num_swapin_timing*[0]
                current_layer = 0
                tmp_layer = 0
                for conv_layer in conv_layers:
                    while current_layer < conv_layer:
                        swapin_counts[tmp_layer] += 1
                        current_layer += 1
                    tmp_layer = conv_layer
                return targets, swapin_counts

            # optimize targets and swapin_counts
            start_time = time.time()
            optimize_setting = variable.ooc_optimize_setting
            #print(optimize_setting)
            
            initial_vars = list()
            best_targets = dict(zip(target_vars, len(target_vars)*["swap"]))
            best_swapin_counts = num_swapin_timing*[1]
            best_time = float("inf")

            if optimize_setting is None:
                best_time, best_targets, best_swapin_counts = optimize_swap_targets_heuristics(initial_vars, best_targets)
                best_time, best_targets, best_swapin_counts = optimize_recompute_targets_heuristics(best_targets)
                best_swapin_counts = delay_swapin(best_targets, best_swapin_counts)
            elif optimize_setting == 'superneurons':
                best_targets, best_swapin_counts = optimize_targets_superneurons()
            elif optimize_setting == 'keep_all':
                best_targets = dict(zip(target_vars, len(target_vars)*["keep"]))
            elif optimize_setting == 'swap_all_no_scheduling':
                best_targets = dict(zip(target_vars, len(target_vars)*["swap"]))
                best_swapin_counts = num_swapin_timing*[1]
                for i in range(num_swapin_timing-1):
                    best_swapin_counts[i] += 1
                    best_swapin_counts[i+1] -= 1
                    cm, _, _ = sync_swapout(best_targets)
                    _, _, ss, rs = rebuild_compute_graph(best_targets)
                    if compact_swapin(cm, best_swapin_counts, ss, rs) is None:
                        best_swapin_counts[i] -= 1
                        best_swapin_counts[i+1] += 1
            elif optimize_setting == 'swap_all':
                best_targets = dict(zip(target_vars, len(target_vars)*["swap"]))
                _, best_swapin_counts, _, _ = simulate_forward_and_backward(best_targets)
                best_swapin_counts = delay_swapin(best_targets, best_swapin_counts)
            elif optimize_setting == 'swap_opt':
                best_time, best_targets, best_swapin_counts = optimize_swap_targets_heuristics(initial_vars, best_targets)
                best_swapin_counts = delay_swapin(best_targets, best_swapin_counts)
            elif optimize_setting == 'recompute_all':
                best_targets = dict(zip(target_vars, len(target_vars)*["recompute"]))

            #print()
            #print(rebuild_compute_graph(best_targets)[2])
            #print(rebuild_compute_graph(best_targets)[3])
            #print()

            #print("optimize_time: ", time.time() - start_time)

            # partition compute_graph per sequence (by "input layer")
            num_inputs = 0
            func_seq_dict = dict()
            for node in compute_graph:
                if "chainer.variable.VariableNode" in node:
                    if node not in compute_graph:
                        continue
                    func = compute_graph[node]
                    func_inputs = compute_graph[func]
                    if all((var not in compute_graph) for var in func_inputs):
                        num_inputs += 1
                        #print("input")
                    func_seq_dict[func] = num_inputs
            #print("num_inputs: ", num_inputs)

            
            self.is_targets = list()
            if best_targets is not None:
                self.is_targets = [(best_targets[var], var_sizes[var], num_inputs-func_seq_dict[compute_graph[var]]) for var in target_vars]
            self.swapin_counts = best_swapin_counts

            
            current_problem_size = self.is_targets[0][1]
            variable.ooc_optimize_dict[(num_inputs, current_problem_size)] = (list(self.is_targets), list(self.swapin_counts))
            #if all(x[0] == "keep" for x in self.is_targets):
            #    variable.keep_all_problem_size[num_inputs] = max(current_problem_size, variable.keep_all_problem_size[num_inputs])

            

            # use same "num_inputs" for next iteration by defalut
            variable.advise_num_inputs(num_inputs)

            #print("targets = ", self.is_targets)
            #print("swapin_counts = ", self.swapin_counts)
            #print()
            #for item in compute_graph:
            #    if "chainer.variable.VariableNode" in item:
            #        print(item, compute_graph[item], item in target_vars)
            #        if item in var_sizes:
            #            print(var_sizes[item])
            #print()

        memory_pool.set_profile_mode(self.updater.iteration == 2)
        memory_pool.memory_log_reset()
        function_node.compute_graph = {}
        function_node.var_size = {}
        function_node.func_backward_use_vars = {}

    def run(self, show_loop_exception_msg=True):
        """Executes the training loop.

        This method is the core of ``Trainer``. It executes the whole loop of
        training the models.

        Note that this method cannot run multiple times for one trainer object.

        """
        if self._done:
            raise RuntimeError('cannot run training loop multiple times')

        try:
            os.makedirs(self.out)
        except OSError:
            pass

        # sort extensions by priorities
        extension_order = sorted(
            self._extensions.keys(),
            key=lambda name: self._extensions[name].priority, reverse=True)
        extensions = [(name, self._extensions[name])
                      for name in extension_order]

        self._start_at = _get_time()

        # invoke initializer of each extension
        for _, entry in extensions:
            initializer = getattr(entry.extension, 'initialize', None)
            if initializer:
                initializer(self)

        update = self.updater.update
        reporter = self.reporter
        stop_trigger = self.stop_trigger

        # main training loop
        try:
            while not stop_trigger(self):
                self.observation = {}
                with reporter.scope(self.observation):
                    self.optimize_targets()
                    update()
                    for name, entry in extensions:
                        if entry.trigger(self):
                            entry.extension(self)
        except Exception as e:
            if show_loop_exception_msg:
                # Show the exception here, as it will appear as if chainer
                # hanged in case any finalize method below deadlocks.
                print('Exception in main training loop: {}'.format(e),
                      file=sys.stderr)
                print('Traceback (most recent call last):', file=sys.stderr)
                traceback.print_tb(sys.exc_info()[2])
                print('Will finalize trainer extensions and updater before '
                      'reraising the exception.', file=sys.stderr)
            six.reraise(*sys.exc_info())
        finally:
            for _, entry in extensions:
                finalize = getattr(entry.extension, 'finalize', None)
                if finalize:
                    finalize()
            self.updater.finalize()

        self._final_elapsed_time = self.elapsed_time
        self._done = True

    def serialize(self, serializer):
        self.updater.serialize(serializer['updater'])
        if hasattr(self.stop_trigger, 'serialize'):
            self.stop_trigger.serialize(serializer['stop_trigger'])

        s = serializer['extensions']
        t = serializer['extension_triggers']
        for name, entry in six.iteritems(self._extensions):
            if hasattr(entry.extension, 'serialize'):
                entry.extension.serialize(s[name])
            if hasattr(entry.trigger, 'serialize'):
                entry.trigger.serialize(t[name])

        if isinstance(serializer, serializer_module.Serializer):
            serializer('_snapshot_elapsed_time', self.elapsed_time)
        else:
            self._snapshot_elapsed_time = serializer(
                '_snapshot_elapsed_time', 0.0)
