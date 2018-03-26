from __future__ import print_function

import collections
import os
import sys
import time
import traceback
import itertools
import copy

import six

from chainer import reporter as reporter_module
from chainer import serializer as serializer_module
from chainer.training import extension as extension_module
from chainer.training import trigger as trigger_module
from chainer.utils import argument
from chainer import cuda
from chainer import variable, function_node
from chainer.cuda import memory_pool

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
        
        self.is_swap_targets = list()
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
    
    def make_log_list(self, tmp_list):
        
        exsit_swapin_item = False
        exsit_swapout_items = collections.defaultdict(lambda: None)
        
        swapout_vars = list()
        swapout_var_sizes = {}
        
        # collect compute time information
        compute_times = {}
        while len(function_node.compute_events) > 0:
            item = function_node.compute_events.pop()
            item[3].synchronize()
            compute_times[(item[0], item[1])] = cuda.cupy.cuda.get_elapsed_time(item[2], item[3])
        
        # collect swap time information
        swap_times = {}
        while len(variable.swap_events) > 0:
            item = variable.swap_events.pop()
            item[3].synchronize()
            swap_times[(item[0], item[1])] = cuda.cupy.cuda.get_elapsed_time(item[2], item[3])
        
        # make log_list
        log_list = list()
        new_item = tmp_list.pop(0)
        
        while len(tmp_list) > 0:
            current_item = tmp_list.pop(0)

            if current_item[0] == "malloc":
                if exsit_swapin_item:
                    exsit_swapin_item = False
                elif new_item[0] == "malloc":
                    new_item = ("malloc", new_item[1]+current_item[1])
                else:
                    log_list.append(new_item)
                    new_item = current_item
            elif current_item[0] == "free":
                if exsit_swapout_items[current_item[2]] is not None:
                    log_list.append(new_item)
                    new_item = ("swapout_free", current_item[1], exsit_swapout_items[current_item[2]])
                    exsit_swapout_items[current_item[2]] = None
                elif new_item[0] == "free":
                    new_item = ("free", new_item[1]+current_item[1])
                else:
                    log_list.append(new_item)
                    new_item = ("free", current_item[1])
            elif current_item[0] == "swapin":
                log_list.append(new_item)
                new_item = ("swapin", current_item[1], current_item[2], swap_times[("swapin", current_item[2])])
                exsit_swapin_item = True
            elif current_item[0] == "swapout":
                log_list.append(new_item)
                new_item = ("swapout", current_item[1], current_item[2], swap_times[("swapout", current_item[2])])
                exsit_swapout_items[current_item[3]] = current_item[2]
                swapout_vars.append(current_item[2])
                swapout_var_sizes[current_item[2]] = current_item[1]
            elif current_item[0] == "forward" or current_item[0] == "backward":
                if ("forward", current_item[1]) in compute_times.keys() and ("backward", current_item[1]) in compute_times.keys(): 
                    log_list.append(new_item)
                    new_item = (current_item[0], current_item[1], compute_times[(current_item[0], current_item[1])])
            else:
                log_list.append(new_item)
                new_item = current_item
        log_list.append(new_item)
        
        return log_list, swapout_vars, swapout_var_sizes
        
    def optimize_swap_targets(self):
        
        if memory_pool.get_profile_mode():
            log_list, swapout_vars, swapout_var_sizes = self.make_log_list(memory_pool.memory_log_get())
            total_mem_size = max(15*1024*1024*1024, memory_pool.total_bytes())
            
            #print(log_list)
            #print(swapout_vars)
            
            # filter memory_items
            memory_items = list(filter(lambda item:item[0] in {"used_bytes", "malloc", "free", "swapin_timing", "swapout_free", "forward_to_backward"}, log_list))

            # filter forward_times, swapout_items, backward_times and swapin_items
            forward_times = list()
            tmp_forward_time = 0.0
            swapout_items = list()
            backward_times = list()
            tmp_backward_time = 0.0
            swapin_items = list()
            tmp_swapin_items = list()
            num_swapin_timing = 0
            for item in log_list:
                if item[0] == "forward":
                    tmp_forward_time += item[2]
                elif item[0] == "swapout":
                    forward_times.append(tmp_forward_time)
                    tmp_forward_time = 0.0
                    swapout_items.append(item)
                elif item[0] == "backward":
                    tmp_backward_time += item[2]
                elif item[0] == "swapin":
                    tmp_swapin_items.append(item)
                elif item[0] == "swapin_timing":
                    backward_times.append(tmp_backward_time)
                    tmp_backward_time = 0.0
                    swapin_items.append(tmp_swapin_items)
                    tmp_swapin_items = list()
                    num_swapin_timing += 1
            backward_times.pop(0)
            backward_times.append(tmp_backward_time)
            swapin_items.pop(0)
            swapin_items.append(tmp_swapin_items)

            # make variables set
            seen_vars = list()
            for var in swapout_vars:
                if var not in seen_vars:
                    seen_vars.append(var)
                    
            def compact_swapin(targets, swapin_counts):
            
                current_memory_usage = 0
                max_memory_usage = 0
                tmp_val = 0 # record max_memory_usage between swapin_timing
                tmp_memory_usage = list()

                # calculate swapin_sizes
                swapin_sizes = num_swapin_timing*[0]
                for i in range(num_swapin_timing):
                    items = swapin_items[i]
                    for item in items:
                        if targets[item[2]]:
                            swapin_sizes[i] += item[1]

                swapin_timing_count = 0
                swapin_times_count = 0

                # simulate memory management
                for i in range(len(memory_items)):
                    memory_item = memory_items[i]
                    if memory_item[0] == "malloc":
                        current_memory_usage += memory_item[1]
                    elif memory_item[0] == "free":
                        current_memory_usage -= memory_item[1]
                    elif memory_item[0] == "swapin_timing":
                        tmp_memory_usage.append(tmp_val)
                        tmp_val = current_memory_usage
                        for j in range(swapin_counts[swapin_timing_count]):
                            current_memory_usage += swapin_sizes[swapin_times_count]
                            swapin_times_count += 1
                        swapin_timing_count += 1
                    elif memory_item[0] == "swapout_free":
                        if targets[memory_item[2]]:
                            current_memory_usage -= memory_item[1]
                    elif memory_item[0] == "used_bytes":
                        current_memory_usage = memory_item[1]
                    elif memory_item[0] == "forward_to_backward":
                        tmp_val = current_memory_usage

                    if current_memory_usage > tmp_val:
                        tmp_val = current_memory_usage
                    if current_memory_usage > max_memory_usage:
                        max_memory_usage = current_memory_usage
                        if max_memory_usage > total_mem_size:
                            return None, max_memory_usage
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

                return new_swapin_counts, max_memory_usage


            def split_forward_timeline(targets):
            
                compute_time = 0.0
                swap_time = 0.0

                swapout_var_block = list()
                total_forward_compute_time = sum(forward_times)

                # simulate forward timeline
                for i in range(len(forward_times)):
                    compute_time += forward_times[i]

                    if targets[swapout_items[i][2]]:
                        swap_time = max(compute_time, swap_time) + swapout_items[i][3]
                        if total_forward_compute_time < swap_time:
                            swapout_var_block.append(swapout_items[i][2])

                return max(compute_time, swap_time), swapout_var_block

            def split_backward_timeline(targets, swapin_counts):

                # calculate swapin_times
                swapin_times = num_swapin_timing*[0]
                for i in range(num_swapin_timing):
                    items = swapin_items[i]
                    for item in items:
                        if targets[item[2]]:
                            swapin_times[i] += item[3]

                swapin_item_blocks = list()
                swapin_item_block = list()
                block_status = None

                swapin_times_count = 0
                tmp_times = num_swapin_timing*[0]

                # simulate backward timeline
                compute_time = backward_times[0]
                swap_time = 0
                tmp_backward_time = 0.0
                for j in range(swapin_counts[0]):
                    swap_time += swapin_times[swapin_times_count]
                    tmp_times[swapin_times_count] = swap_time
                    swapin_times_count += 1

                for i in range(1, num_swapin_timing):

                    if swapin_times_count == i:
                        if swap_time < compute_time:
                            swap_time = compute_time
                            block_status = True
                
                    for j in range(swapin_counts[i]):
                        swap_time += swapin_times[swapin_times_count]
                        tmp_times[swapin_times_count] = swap_time
                        for item in swapin_items[swapin_times_count]:
                            swapin_item_block.append(item)
                        swapin_times_count += 1
                            
                    if compute_time < tmp_times[i]:
                        compute_time = tmp_times[i]
                        block_status = False

                    compute_time += backward_times[i]
                    tmp_backward_time += backward_times[i]

                    if block_status is not None:
                        swapin_item_blocks.append((swapin_item_block, block_status, tmp_backward_time))
                        swapin_item_block = list()
                        block_status = None
                        tmp_backward_time = 0.0

                if len(swapin_item_block) > 0:
                    swapin_item_blocks.append((swapin_item_block, True, tmp_backward_time))

                return compute_time, swapin_item_blocks

            
            def optimize_swap_targets_main(selected_vars, targets):

                new_selected_vars = list(selected_vars)
                current_targets = dict(targets)

                swapin_counts, _ = compact_swapin(targets, num_swapin_timing*[1])
                if swapin_counts is None:
                    return float("inf"), None, None

                forward_compute_time, swapout_var_block = split_forward_timeline(targets)
                backward_compute_time, swapin_item_blocks = split_backward_timeline(targets, swapin_counts)
                compute_time = forward_compute_time + backward_compute_time
                
                for element in swapin_item_blocks[::-1]:
                    swapin_item_block = element[0]
                    block_status = element[1]
                    tmp_backward_time = element[2]
                    
                    tmp_swapin_time = 0.0
                    search_items = list()
                    for swapin_item in swapin_item_block:
                        swapin_var = swapin_item[2]
                        if swapin_var not in new_selected_vars:
                            new_selected_vars.append(swapin_var)
                            search_items.append(swapin_item)
                        if targets[swapin_var]:
                            tmp_swapin_time += swapin_item[3]

                    if block_status is False:
                        search_items = list(sorted(search_items, key=lambda x: x[1]))
                        while tmp_swapin_time > tmp_backward_time:
                            search_item = search_items.pop(0)
                            search_var = search_item[2]
                            tmp_swapin_time -= search_item[3]
                            
                            targets[search_var] = False
                            tmp_compute_time, tmp_current_targets, tmp_swapin_counts = optimize_swap_targets_main(new_selected_vars, dict(targets))
                            if compute_time > tmp_compute_time:
                                compute_time = tmp_compute_time 
                                current_targets = tmp_current_targets
                                swapin_counts = tmp_swapin_counts
                                
                for var in swapout_var_block:
                    targets[var] = False
                    tmp_swapin_counts, _ = compact_swapin(targets, num_swapin_timing*[1])
                    if tmp_swapin_counts is None:
                        targets[var] = True
                        tmp_swapin_counts, _ = compact_swapin(targets, num_swapin_timing*[1])
                        break
                        
                forward_compute_time, _ = split_forward_timeline(targets)
                backward_compute_time, _ = split_backward_timeline(targets, swapin_counts)
                tmp_compute_time = forward_compute_time + backward_compute_time
                if compute_time > tmp_compute_time:
                    compute_time = tmp_compute_time 
                    current_targets = targets
                    swapin_counts = tmp_swapin_counts
                
                return compute_time, current_targets, swapin_counts
            
            def delay_swapin(targets, current_swapin_counts):

                # calculate swapin_times
                swapin_times = num_swapin_timing*[0]
                for i in range(num_swapin_timing):
                    items = swapin_items[i]
                    for item in items:
                        if targets[item[2]]:
                            swapin_times[i] += item[3]
                            
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

                    while j >= swapin_block[0]:
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
                        else:
                            k -= 1
                            tmp2 += backward_times[k]
                            backward_block_time -= backward_times[k]
                    current_backward = k-1
                            
                return swapin_counts
            
            
            execution_time = time.time()

            initial_vars = list()
            initial_targets = dict(zip(seen_vars, len(seen_vars)*[True]))
            best_time, best_targets, best_swapin_counts = optimize_swap_targets_main(initial_vars, initial_targets)
            
            best_swapin_counts = delay_swapin(best_targets, best_swapin_counts)
            
            # for comparison with original chainer or chainer_ooc
            #for var in seen_vars:
            #    best_targets[var] = True
            #    best_targets[var] = False
            #best_swapin_counts = num_swapin_timing*[1]
            #best_swapin_counts[0] = 2
            #best_swapin_counts[num_swapin_timing-1] = 0
            
            self.is_swap_targets = list()
            if best_targets is not None:
                while len(seen_vars):
                    var = seen_vars.pop(0)
                    self.is_swap_targets.append((best_targets[var], swapout_var_sizes[var]))
            self.swapin_counts = best_swapin_counts
            #print(self.is_swap_targets.append)
            #print(self.swapin_counts)

        memory_pool.set_profile_mode(self.updater.iteration == 2)
        memory_pool.memory_log_reset()
        variable.is_swap_targets = list(self.is_swap_targets)
        variable.swapin_counts = list(self.swapin_counts)

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
                    self.optimize_swap_targets()
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
