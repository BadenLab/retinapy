
(function(l, r) { if (!l || l.getElementById('livereloadscript')) return; r = l.createElement('script'); r.async = 1; r.src = '//' + (self.location.host || 'localhost').split(':')[0] + ':35729/livereload.js?snipver=1'; r.id = 'livereloadscript'; l.getElementsByTagName('head')[0].appendChild(r) })(self.document);
var app = (function () {
    'use strict';

    function noop() { }
    function assign(tar, src) {
        // @ts-ignore
        for (const k in src)
            tar[k] = src[k];
        return tar;
    }
    function is_promise(value) {
        return value && typeof value === 'object' && typeof value.then === 'function';
    }
    function add_location(element, file, line, column, char) {
        element.__svelte_meta = {
            loc: { file, line, column, char }
        };
    }
    function run(fn) {
        return fn();
    }
    function blank_object() {
        return Object.create(null);
    }
    function run_all(fns) {
        fns.forEach(run);
    }
    function is_function(thing) {
        return typeof thing === 'function';
    }
    function safe_not_equal(a, b) {
        return a != a ? b == b : a !== b || ((a && typeof a === 'object') || typeof a === 'function');
    }
    function is_empty(obj) {
        return Object.keys(obj).length === 0;
    }
    function validate_store(store, name) {
        if (store != null && typeof store.subscribe !== 'function') {
            throw new Error(`'${name}' is not a store with a 'subscribe' method`);
        }
    }
    function subscribe(store, ...callbacks) {
        if (store == null) {
            return noop;
        }
        const unsub = store.subscribe(...callbacks);
        return unsub.unsubscribe ? () => unsub.unsubscribe() : unsub;
    }
    function get_store_value(store) {
        let value;
        subscribe(store, _ => value = _)();
        return value;
    }
    function component_subscribe(component, store, callback) {
        component.$$.on_destroy.push(subscribe(store, callback));
    }
    function create_slot(definition, ctx, $$scope, fn) {
        if (definition) {
            const slot_ctx = get_slot_context(definition, ctx, $$scope, fn);
            return definition[0](slot_ctx);
        }
    }
    function get_slot_context(definition, ctx, $$scope, fn) {
        return definition[1] && fn
            ? assign($$scope.ctx.slice(), definition[1](fn(ctx)))
            : $$scope.ctx;
    }
    function get_slot_changes(definition, $$scope, dirty, fn) {
        if (definition[2] && fn) {
            const lets = definition[2](fn(dirty));
            if ($$scope.dirty === undefined) {
                return lets;
            }
            if (typeof lets === 'object') {
                const merged = [];
                const len = Math.max($$scope.dirty.length, lets.length);
                for (let i = 0; i < len; i += 1) {
                    merged[i] = $$scope.dirty[i] | lets[i];
                }
                return merged;
            }
            return $$scope.dirty | lets;
        }
        return $$scope.dirty;
    }
    function update_slot_base(slot, slot_definition, ctx, $$scope, slot_changes, get_slot_context_fn) {
        if (slot_changes) {
            const slot_context = get_slot_context(slot_definition, ctx, $$scope, get_slot_context_fn);
            slot.p(slot_context, slot_changes);
        }
    }
    function get_all_dirty_from_scope($$scope) {
        if ($$scope.ctx.length > 32) {
            const dirty = [];
            const length = $$scope.ctx.length / 32;
            for (let i = 0; i < length; i++) {
                dirty[i] = -1;
            }
            return dirty;
        }
        return -1;
    }
    function null_to_empty(value) {
        return value == null ? '' : value;
    }
    function append(target, node) {
        target.appendChild(node);
    }
    function insert(target, node, anchor) {
        target.insertBefore(node, anchor || null);
    }
    function detach(node) {
        node.parentNode.removeChild(node);
    }
    function destroy_each(iterations, detaching) {
        for (let i = 0; i < iterations.length; i += 1) {
            if (iterations[i])
                iterations[i].d(detaching);
        }
    }
    function element(name) {
        return document.createElement(name);
    }
    function text(data) {
        return document.createTextNode(data);
    }
    function space() {
        return text(' ');
    }
    function listen(node, event, handler, options) {
        node.addEventListener(event, handler, options);
        return () => node.removeEventListener(event, handler, options);
    }
    function attr(node, attribute, value) {
        if (value == null)
            node.removeAttribute(attribute);
        else if (node.getAttribute(attribute) !== value)
            node.setAttribute(attribute, value);
    }
    function to_number(value) {
        return value === '' ? null : +value;
    }
    function children(element) {
        return Array.from(element.childNodes);
    }
    function set_input_value(input, value) {
        input.value = value == null ? '' : value;
    }
    function set_style(node, key, value, important) {
        if (value === null) {
            node.style.removeProperty(key);
        }
        else {
            node.style.setProperty(key, value, important ? 'important' : '');
        }
    }
    function select_option(select, value) {
        for (let i = 0; i < select.options.length; i += 1) {
            const option = select.options[i];
            if (option.__value === value) {
                option.selected = true;
                return;
            }
        }
        select.selectedIndex = -1; // no option should be selected
    }
    function select_value(select) {
        const selected_option = select.querySelector(':checked') || select.options[0];
        return selected_option && selected_option.__value;
    }
    function custom_event(type, detail, { bubbles = false, cancelable = false } = {}) {
        const e = document.createEvent('CustomEvent');
        e.initCustomEvent(type, bubbles, cancelable, detail);
        return e;
    }

    let current_component;
    function set_current_component(component) {
        current_component = component;
    }
    function get_current_component() {
        if (!current_component)
            throw new Error('Function called outside component initialization');
        return current_component;
    }
    function onMount(fn) {
        get_current_component().$$.on_mount.push(fn);
    }
    function onDestroy(fn) {
        get_current_component().$$.on_destroy.push(fn);
    }
    function setContext(key, context) {
        get_current_component().$$.context.set(key, context);
        return context;
    }
    function getContext(key) {
        return get_current_component().$$.context.get(key);
    }

    const dirty_components = [];
    const binding_callbacks = [];
    const render_callbacks = [];
    const flush_callbacks = [];
    const resolved_promise = Promise.resolve();
    let update_scheduled = false;
    function schedule_update() {
        if (!update_scheduled) {
            update_scheduled = true;
            resolved_promise.then(flush);
        }
    }
    function add_render_callback(fn) {
        render_callbacks.push(fn);
    }
    // flush() calls callbacks in this order:
    // 1. All beforeUpdate callbacks, in order: parents before children
    // 2. All bind:this callbacks, in reverse order: children before parents.
    // 3. All afterUpdate callbacks, in order: parents before children. EXCEPT
    //    for afterUpdates called during the initial onMount, which are called in
    //    reverse order: children before parents.
    // Since callbacks might update component values, which could trigger another
    // call to flush(), the following steps guard against this:
    // 1. During beforeUpdate, any updated components will be added to the
    //    dirty_components array and will cause a reentrant call to flush(). Because
    //    the flush index is kept outside the function, the reentrant call will pick
    //    up where the earlier call left off and go through all dirty components. The
    //    current_component value is saved and restored so that the reentrant call will
    //    not interfere with the "parent" flush() call.
    // 2. bind:this callbacks cannot trigger new flush() calls.
    // 3. During afterUpdate, any updated components will NOT have their afterUpdate
    //    callback called a second time; the seen_callbacks set, outside the flush()
    //    function, guarantees this behavior.
    const seen_callbacks = new Set();
    let flushidx = 0; // Do *not* move this inside the flush() function
    function flush() {
        const saved_component = current_component;
        do {
            // first, call beforeUpdate functions
            // and update components
            while (flushidx < dirty_components.length) {
                const component = dirty_components[flushidx];
                flushidx++;
                set_current_component(component);
                update(component.$$);
            }
            set_current_component(null);
            dirty_components.length = 0;
            flushidx = 0;
            while (binding_callbacks.length)
                binding_callbacks.pop()();
            // then, once components are updated, call
            // afterUpdate functions. This may cause
            // subsequent updates...
            for (let i = 0; i < render_callbacks.length; i += 1) {
                const callback = render_callbacks[i];
                if (!seen_callbacks.has(callback)) {
                    // ...so guard against infinite loops
                    seen_callbacks.add(callback);
                    callback();
                }
            }
            render_callbacks.length = 0;
        } while (dirty_components.length);
        while (flush_callbacks.length) {
            flush_callbacks.pop()();
        }
        update_scheduled = false;
        seen_callbacks.clear();
        set_current_component(saved_component);
    }
    function update($$) {
        if ($$.fragment !== null) {
            $$.update();
            run_all($$.before_update);
            const dirty = $$.dirty;
            $$.dirty = [-1];
            $$.fragment && $$.fragment.p($$.ctx, dirty);
            $$.after_update.forEach(add_render_callback);
        }
    }
    const outroing = new Set();
    let outros;
    function group_outros() {
        outros = {
            r: 0,
            c: [],
            p: outros // parent group
        };
    }
    function check_outros() {
        if (!outros.r) {
            run_all(outros.c);
        }
        outros = outros.p;
    }
    function transition_in(block, local) {
        if (block && block.i) {
            outroing.delete(block);
            block.i(local);
        }
    }
    function transition_out(block, local, detach, callback) {
        if (block && block.o) {
            if (outroing.has(block))
                return;
            outroing.add(block);
            outros.c.push(() => {
                outroing.delete(block);
                if (callback) {
                    if (detach)
                        block.d(1);
                    callback();
                }
            });
            block.o(local);
        }
        else if (callback) {
            callback();
        }
    }

    function handle_promise(promise, info) {
        const token = info.token = {};
        function update(type, index, key, value) {
            if (info.token !== token)
                return;
            info.resolved = value;
            let child_ctx = info.ctx;
            if (key !== undefined) {
                child_ctx = child_ctx.slice();
                child_ctx[key] = value;
            }
            const block = type && (info.current = type)(child_ctx);
            let needs_flush = false;
            if (info.block) {
                if (info.blocks) {
                    info.blocks.forEach((block, i) => {
                        if (i !== index && block) {
                            group_outros();
                            transition_out(block, 1, 1, () => {
                                if (info.blocks[i] === block) {
                                    info.blocks[i] = null;
                                }
                            });
                            check_outros();
                        }
                    });
                }
                else {
                    info.block.d(1);
                }
                block.c();
                transition_in(block, 1);
                block.m(info.mount(), info.anchor);
                needs_flush = true;
            }
            info.block = block;
            if (info.blocks)
                info.blocks[index] = block;
            if (needs_flush) {
                flush();
            }
        }
        if (is_promise(promise)) {
            const current_component = get_current_component();
            promise.then(value => {
                set_current_component(current_component);
                update(info.then, 1, info.value, value);
                set_current_component(null);
            }, error => {
                set_current_component(current_component);
                update(info.catch, 2, info.error, error);
                set_current_component(null);
                if (!info.hasCatch) {
                    throw error;
                }
            });
            // if we previously had a then/catch block, destroy it
            if (info.current !== info.pending) {
                update(info.pending, 0);
                return true;
            }
        }
        else {
            if (info.current !== info.then) {
                update(info.then, 1, info.value, promise);
                return true;
            }
            info.resolved = promise;
        }
    }
    function update_await_block_branch(info, ctx, dirty) {
        const child_ctx = ctx.slice();
        const { resolved } = info;
        if (info.current === info.then) {
            child_ctx[info.value] = resolved;
        }
        if (info.current === info.catch) {
            child_ctx[info.error] = resolved;
        }
        info.block.p(child_ctx, dirty);
    }

    const globals = (typeof window !== 'undefined'
        ? window
        : typeof globalThis !== 'undefined'
            ? globalThis
            : global);
    function create_component(block) {
        block && block.c();
    }
    function mount_component(component, target, anchor, customElement) {
        const { fragment, on_mount, on_destroy, after_update } = component.$$;
        fragment && fragment.m(target, anchor);
        if (!customElement) {
            // onMount happens before the initial afterUpdate
            add_render_callback(() => {
                const new_on_destroy = on_mount.map(run).filter(is_function);
                if (on_destroy) {
                    on_destroy.push(...new_on_destroy);
                }
                else {
                    // Edge case - component was destroyed immediately,
                    // most likely as a result of a binding initialising
                    run_all(new_on_destroy);
                }
                component.$$.on_mount = [];
            });
        }
        after_update.forEach(add_render_callback);
    }
    function destroy_component(component, detaching) {
        const $$ = component.$$;
        if ($$.fragment !== null) {
            run_all($$.on_destroy);
            $$.fragment && $$.fragment.d(detaching);
            // TODO null out other refs, including component.$$ (but need to
            // preserve final state?)
            $$.on_destroy = $$.fragment = null;
            $$.ctx = [];
        }
    }
    function make_dirty(component, i) {
        if (component.$$.dirty[0] === -1) {
            dirty_components.push(component);
            schedule_update();
            component.$$.dirty.fill(0);
        }
        component.$$.dirty[(i / 31) | 0] |= (1 << (i % 31));
    }
    function init(component, options, instance, create_fragment, not_equal, props, append_styles, dirty = [-1]) {
        const parent_component = current_component;
        set_current_component(component);
        const $$ = component.$$ = {
            fragment: null,
            ctx: null,
            // state
            props,
            update: noop,
            not_equal,
            bound: blank_object(),
            // lifecycle
            on_mount: [],
            on_destroy: [],
            on_disconnect: [],
            before_update: [],
            after_update: [],
            context: new Map(options.context || (parent_component ? parent_component.$$.context : [])),
            // everything else
            callbacks: blank_object(),
            dirty,
            skip_bound: false,
            root: options.target || parent_component.$$.root
        };
        append_styles && append_styles($$.root);
        let ready = false;
        $$.ctx = instance
            ? instance(component, options.props || {}, (i, ret, ...rest) => {
                const value = rest.length ? rest[0] : ret;
                if ($$.ctx && not_equal($$.ctx[i], $$.ctx[i] = value)) {
                    if (!$$.skip_bound && $$.bound[i])
                        $$.bound[i](value);
                    if (ready)
                        make_dirty(component, i);
                }
                return ret;
            })
            : [];
        $$.update();
        ready = true;
        run_all($$.before_update);
        // `false` as a special case of no DOM component
        $$.fragment = create_fragment ? create_fragment($$.ctx) : false;
        if (options.target) {
            if (options.hydrate) {
                const nodes = children(options.target);
                // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
                $$.fragment && $$.fragment.l(nodes);
                nodes.forEach(detach);
            }
            else {
                // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
                $$.fragment && $$.fragment.c();
            }
            if (options.intro)
                transition_in(component.$$.fragment);
            mount_component(component, options.target, options.anchor, options.customElement);
            flush();
        }
        set_current_component(parent_component);
    }
    /**
     * Base class for Svelte components. Used when dev=false.
     */
    class SvelteComponent {
        $destroy() {
            destroy_component(this, 1);
            this.$destroy = noop;
        }
        $on(type, callback) {
            const callbacks = (this.$$.callbacks[type] || (this.$$.callbacks[type] = []));
            callbacks.push(callback);
            return () => {
                const index = callbacks.indexOf(callback);
                if (index !== -1)
                    callbacks.splice(index, 1);
            };
        }
        $set($$props) {
            if (this.$$set && !is_empty($$props)) {
                this.$$.skip_bound = true;
                this.$$set($$props);
                this.$$.skip_bound = false;
            }
        }
    }

    function dispatch_dev(type, detail) {
        document.dispatchEvent(custom_event(type, Object.assign({ version: '3.49.0' }, detail), { bubbles: true }));
    }
    function append_dev(target, node) {
        dispatch_dev('SvelteDOMInsert', { target, node });
        append(target, node);
    }
    function insert_dev(target, node, anchor) {
        dispatch_dev('SvelteDOMInsert', { target, node, anchor });
        insert(target, node, anchor);
    }
    function detach_dev(node) {
        dispatch_dev('SvelteDOMRemove', { node });
        detach(node);
    }
    function listen_dev(node, event, handler, options, has_prevent_default, has_stop_propagation) {
        const modifiers = options === true ? ['capture'] : options ? Array.from(Object.keys(options)) : [];
        if (has_prevent_default)
            modifiers.push('preventDefault');
        if (has_stop_propagation)
            modifiers.push('stopPropagation');
        dispatch_dev('SvelteDOMAddEventListener', { node, event, handler, modifiers });
        const dispose = listen(node, event, handler, options);
        return () => {
            dispatch_dev('SvelteDOMRemoveEventListener', { node, event, handler, modifiers });
            dispose();
        };
    }
    function attr_dev(node, attribute, value) {
        attr(node, attribute, value);
        if (value == null)
            dispatch_dev('SvelteDOMRemoveAttribute', { node, attribute });
        else
            dispatch_dev('SvelteDOMSetAttribute', { node, attribute, value });
    }
    function set_data_dev(text, data) {
        data = '' + data;
        if (text.wholeText === data)
            return;
        dispatch_dev('SvelteDOMSetData', { node: text, data });
        text.data = data;
    }
    function validate_each_argument(arg) {
        if (typeof arg !== 'string' && !(arg && typeof arg === 'object' && 'length' in arg)) {
            let msg = '{#each} only iterates over array-like objects.';
            if (typeof Symbol === 'function' && arg && Symbol.iterator in arg) {
                msg += ' You can use a spread to convert this iterable into an array.';
            }
            throw new Error(msg);
        }
    }
    function validate_slots(name, slot, keys) {
        for (const slot_key of Object.keys(slot)) {
            if (!~keys.indexOf(slot_key)) {
                console.warn(`<${name}> received an unexpected slot "${slot_key}".`);
            }
        }
    }
    /**
     * Base class for Svelte components with some minor dev-enhancements. Used when dev=true.
     */
    class SvelteComponentDev extends SvelteComponent {
        constructor(options) {
            if (!options || (!options.target && !options.$$inline)) {
                throw new Error("'target' is a required option");
            }
            super();
        }
        $destroy() {
            super.$destroy();
            this.$destroy = () => {
                console.warn('Component was already destroyed'); // eslint-disable-line no-console
            };
        }
        $capture_state() { }
        $inject_state() { }
    }

    const subscriber_queue = [];
    /**
     * Creates a `Readable` store that allows reading by subscription.
     * @param value initial value
     * @param {StartStopNotifier}start start and stop notifications for subscriptions
     */
    function readable(value, start) {
        return {
            subscribe: writable(value, start).subscribe
        };
    }
    /**
     * Create a `Writable` store that allows both updating and reading by subscription.
     * @param {*=}value initial value
     * @param {StartStopNotifier=}start start and stop notifications for subscriptions
     */
    function writable(value, start = noop) {
        let stop;
        const subscribers = new Set();
        function set(new_value) {
            if (safe_not_equal(value, new_value)) {
                value = new_value;
                if (stop) { // store is ready
                    const run_queue = !subscriber_queue.length;
                    for (const subscriber of subscribers) {
                        subscriber[1]();
                        subscriber_queue.push(subscriber, value);
                    }
                    if (run_queue) {
                        for (let i = 0; i < subscriber_queue.length; i += 2) {
                            subscriber_queue[i][0](subscriber_queue[i + 1]);
                        }
                        subscriber_queue.length = 0;
                    }
                }
            }
        }
        function update(fn) {
            set(fn(value));
        }
        function subscribe(run, invalidate = noop) {
            const subscriber = [run, invalidate];
            subscribers.add(subscriber);
            if (subscribers.size === 1) {
                stop = start(set) || noop;
            }
            run(value);
            return () => {
                subscribers.delete(subscriber);
                if (subscribers.size === 0) {
                    stop();
                    stop = null;
                }
            };
        }
        return { set, update, subscribe };
    }
    function derived(stores, fn, initial_value) {
        const single = !Array.isArray(stores);
        const stores_array = single
            ? [stores]
            : stores;
        const auto = fn.length < 2;
        return readable(initial_value, (set) => {
            let inited = false;
            const values = [];
            let pending = 0;
            let cleanup = noop;
            const sync = () => {
                if (pending) {
                    return;
                }
                cleanup();
                const result = fn(single ? values[0] : values, set);
                if (auto) {
                    set(result);
                }
                else {
                    cleanup = is_function(result) ? result : noop;
                }
            };
            const unsubscribers = stores_array.map((store, i) => subscribe(store, (value) => {
                values[i] = value;
                pending &= ~(1 << i);
                if (inited) {
                    sync();
                }
            }, () => {
                pending |= (1 << i);
            }));
            inited = true;
            sync();
            return function stop() {
                run_all(unsubscribers);
                cleanup();
            };
        });
    }

    // Some props for the app
    const width = writable(window.innerWidth);
    const height = writable(window.innerHeight);
    const pixelRatio = writable(window.devicePixelRatio);
    const context = writable();
    const canvas = writable();
    const time = writable(0);
    const pause_ctrl = (() => {
    	const {subscribe, set, update } = writable(false);
    	const pause = () => {
    		console.log('paused');
    		set(true);
    	};
    	const resume = () => {
    		console.log('resumed');
    		set(false);
    	};
    	const toggle = () => {
    		update(v => !v);
    	};

    	const current = () => {
    		//return get(subscribe);
    		// Hacky
    		return get_store_value(pause_ctrl);
    	};
    	return {
    		subscribe,
    		pause,
    		resume, 
    		toggle,
    		current
    	}
    })();

    //export const snippets = writable([]);
    //export const playback_speed = writable(0.05);
    //export const sample_rate = readable(99.182);
    //export const snippet_len = readable(120);
    //export const snippet_pad = readable(20);
    //export const is_paused = writable(false);
    //export const playback_time = writable(0);
    //export const sample_idx = derived(
    //	[playback_time, sample_rate, snippet_len], 
    //	([p, sr, sl]) => cur_sample(p, sr, sl));

    // A more convenient store for grabbing all game props
    const props = deriveObject({
    	context,
    	canvas,
    	width,
    	height,
    	pixelRatio,
    	time
    });

    let _playback_time = 0;
    let _playback_speed = 0.05;
    let _sample_rate = 99.182;
    let _snippet_len = 90;
    let _snippet_pad = 10;
    let _snippets = [];

    function update_clock(cur_time, dt) {
    	if(!get_store_value(pause_ctrl)) {
    		_playback_time = (_playback_time + dt * _playback_speed) % playback_duration();
    	}
    }


    function playback_duration() {
    	return _snippet_len / _sample_rate;
    }

    function snippet_len() {
    	return _snippet_len;
    }

    function snippet_pad() {
    	return _snippet_pad;
    }

    function is_paused() {
    	return get_store_value(pause_ctrl);
    }

    function playback_time() {
    	return _playback_time;
    }

    function snippet_time() {
    	const spike_at = (_snippet_len - _snippet_pad) / _sample_rate;
    	return playback_time() - spike_at;
    }

    function set_playback_time(t) {
    	_playback_time = t;
    }


    function set_playback_time_rel(r) {
    	// Either here or somewhere else, handle the the fact that modulo will
    	// send the end to the begininning.
    	r = Math.min(Math.max(0, r), 0.9999999);
    	_playback_time = r * _snippet_len / _sample_rate;
    }

    function sample_idx() {
    	const res = Math.floor((_playback_time * _sample_rate) % _snippet_len);
    	return res;
    }

    function snippets() {
    	return _snippets;
    }

    function set_snippets(snippets) {
    	_snippets = snippets;
    }

    // Javascript built-in function that returns a unique symbol primitive.
    const key = Symbol();

    function getState() {
    	const api = getContext(key);
    	return api.getState();
    }
    function renderable(render) {
    	const api = getContext(key);
    	const element = {
    		ready: false,
    		mounted: false
    	};
    	if (typeof render === 'function') element.render = render;
    	else if (render) {
    		if (render.render) element.render = render.render;
    		if (render.setup) element.setup = render.setup;
    	}
    	api.add(element);
    	onMount(() => {
    		element.mounted = true;
    		return () => {
    			api.remove(element);
    			element.mounted = false;
    		};
    	});
    }

    function deriveObject (obj) {
    	const keys = Object.keys(obj);
    	const list = keys.map(key => {
    		return obj[key];
    	});
    	return derived(list, (array) => {
    		return array.reduce((dict, value, i) => {
    			dict[keys[i]] = value;
    			return dict;
    		}, {});
    	});
    }

    function canvas_pos(global_pos) {
      var rect = get_store_value(canvas).getBoundingClientRect();
      var x = (global_pos[0] - rect.left); 
      var y = (global_pos[1] - rect.top); 
      return [x, y]
    }

    /*export function canvas_pos_from_rel(rel_pos) {
      var rect = canvas.getBoundingClientRect();
      var w = rect.right - rect.left
      var h = rect.bottom - rect.top
      return [rel_pos[0] * h + rel_pos[1] * w]
    }*/

    function mouse_pos(event) {
      return canvas_pos([event.clientX, event.clientY])
    }

    function mouse_pos_rel(event) {
      return canvas_pos_rel([event.clientX, event.clientY])
    }

    var engine = /*#__PURE__*/Object.freeze({
        __proto__: null,
        width: width,
        height: height,
        pixelRatio: pixelRatio,
        context: context,
        canvas: canvas,
        time: time,
        pause_ctrl: pause_ctrl,
        props: props,
        update_clock: update_clock,
        playback_duration: playback_duration,
        snippet_len: snippet_len,
        snippet_pad: snippet_pad,
        is_paused: is_paused,
        playback_time: playback_time,
        snippet_time: snippet_time,
        set_playback_time: set_playback_time,
        set_playback_time_rel: set_playback_time_rel,
        sample_idx: sample_idx,
        snippets: snippets,
        set_snippets: set_snippets,
        key: key,
        getState: getState,
        renderable: renderable,
        mouse_pos: mouse_pos,
        mouse_pos_rel: mouse_pos_rel
    });

    /* src/Canvas.svelte generated by Svelte v3.49.0 */

    const { console: console_1$1, window: window_1 } = globals;

    const file$1 = "src/Canvas.svelte";

    function create_fragment$4(ctx) {
    	let canvas_1;
    	let canvas_1_width_value;
    	let canvas_1_height_value;
    	let t;
    	let current;
    	let mounted;
    	let dispose;
    	const default_slot_template = /*#slots*/ ctx[8].default;
    	const default_slot = create_slot(default_slot_template, ctx, /*$$scope*/ ctx[7], null);

    	const block = {
    		c: function create() {
    			canvas_1 = element("canvas");
    			t = space();
    			if (default_slot) default_slot.c();
    			attr_dev(canvas_1, "width", canvas_1_width_value = /*$width*/ ctx[2] * /*$pixelRatio*/ ctx[1]);
    			attr_dev(canvas_1, "height", canvas_1_height_value = /*$height*/ ctx[3] * /*$pixelRatio*/ ctx[1]);
    			set_style(canvas_1, "width", /*$width*/ ctx[2] + "px");
    			set_style(canvas_1, "height", /*$height*/ ctx[3] + "px");
    			add_location(canvas_1, file$1, 107, 0, 2089);
    		},
    		l: function claim(nodes) {
    			throw new Error("options.hydrate only works if the component was compiled with the `hydratable: true` option");
    		},
    		m: function mount(target, anchor) {
    			insert_dev(target, canvas_1, anchor);
    			/*canvas_1_binding*/ ctx[9](canvas_1);
    			insert_dev(target, t, anchor);

    			if (default_slot) {
    				default_slot.m(target, anchor);
    			}

    			current = true;

    			if (!mounted) {
    				dispose = listen_dev(window_1, "resize", /*handleResize*/ ctx[4], { passive: true }, false, false);
    				mounted = true;
    			}
    		},
    		p: function update(ctx, [dirty]) {
    			if (!current || dirty & /*$width, $pixelRatio*/ 6 && canvas_1_width_value !== (canvas_1_width_value = /*$width*/ ctx[2] * /*$pixelRatio*/ ctx[1])) {
    				attr_dev(canvas_1, "width", canvas_1_width_value);
    			}

    			if (!current || dirty & /*$height, $pixelRatio*/ 10 && canvas_1_height_value !== (canvas_1_height_value = /*$height*/ ctx[3] * /*$pixelRatio*/ ctx[1])) {
    				attr_dev(canvas_1, "height", canvas_1_height_value);
    			}

    			if (!current || dirty & /*$width*/ 4) {
    				set_style(canvas_1, "width", /*$width*/ ctx[2] + "px");
    			}

    			if (!current || dirty & /*$height*/ 8) {
    				set_style(canvas_1, "height", /*$height*/ ctx[3] + "px");
    			}

    			if (default_slot) {
    				if (default_slot.p && (!current || dirty & /*$$scope*/ 128)) {
    					update_slot_base(
    						default_slot,
    						default_slot_template,
    						ctx,
    						/*$$scope*/ ctx[7],
    						!current
    						? get_all_dirty_from_scope(/*$$scope*/ ctx[7])
    						: get_slot_changes(default_slot_template, /*$$scope*/ ctx[7], dirty, null),
    						null
    					);
    				}
    			}
    		},
    		i: function intro(local) {
    			if (current) return;
    			transition_in(default_slot, local);
    			current = true;
    		},
    		o: function outro(local) {
    			transition_out(default_slot, local);
    			current = false;
    		},
    		d: function destroy(detaching) {
    			if (detaching) detach_dev(canvas_1);
    			/*canvas_1_binding*/ ctx[9](null);
    			if (detaching) detach_dev(t);
    			if (default_slot) default_slot.d(detaching);
    			mounted = false;
    			dispose();
    		}
    	};

    	dispatch_dev("SvelteRegisterBlock", {
    		block,
    		id: create_fragment$4.name,
    		type: "component",
    		source: "",
    		ctx
    	});

    	return block;
    }

    function instance$4($$self, $$props, $$invalidate) {
    	let $props;
    	let $pixelRatio;
    	let $width;
    	let $height;
    	validate_store(props, 'props');
    	component_subscribe($$self, props, $$value => $$invalidate(12, $props = $$value));
    	validate_store(pixelRatio, 'pixelRatio');
    	component_subscribe($$self, pixelRatio, $$value => $$invalidate(1, $pixelRatio = $$value));
    	validate_store(width, 'width');
    	component_subscribe($$self, width, $$value => $$invalidate(2, $width = $$value));
    	validate_store(height, 'height');
    	component_subscribe($$self, height, $$value => $$invalidate(3, $height = $$value));
    	let { $$slots: slots = {}, $$scope } = $$props;
    	validate_slots('Canvas', slots, ['default']);
    	let { killLoopOnError = true } = $$props;
    	let { attributes = {} } = $$props;
    	let listeners = [];
    	let canvas$1;
    	let context$1;
    	let frame;

    	onMount(() => {
    		// prepare canvas stores
    		context$1 = canvas$1.getContext('2d', attributes);

    		canvas.set(canvas$1);
    		context.set(context$1);

    		// setup entities
    		listeners.forEach(async entity => {
    			if (entity.setup) {
    				let p = entity.setup($props);
    				if (p && p.then) await p;
    			}

    			entity.ready = true;
    		});

    		// start game loop
    		return createLoop(render);
    	});

    	setContext(key, {
    		add(fn) {
    			this.remove(fn);
    			listeners.push(fn);
    		},
    		remove(fn) {
    			const idx = listeners.indexOf(fn);
    			if (idx >= 0) listeners.splice(idx, 1);
    		}
    	});

    	function handleResize() {
    		width.set(window.innerWidth);
    		height.set(window.innerHeight);
    		pixelRatio.set(window.devicePixelRatio);
    	}

    	/**
     * Called by `loop()`, which is run repeatedly in `createLoop()`.
     */
    	function render(elapsed, dt) {
    		update_clock(elapsed, dt);
    		context$1.save();
    		context$1.scale($pixelRatio, $pixelRatio);

    		listeners.forEach(entity => {
    			try {
    				if (entity.mounted && entity.ready && entity.render) {
    					entity.render($props, dt);
    				}
    			} catch(err) {
    				console.error(err);

    				if (killLoopOnError) {
    					cancelAnimationFrame(frame);
    					console.warn('Animation loop stopped due to an error');
    				}
    			}
    		});

    		context$1.restore();
    	}

    	/**
     * Typically called as `createLoop(render)`. 
     *
     * The equivalent two.js function: 
     *
     * 		https://github.com/jonobr1/two.js/blob/ea7491d0b2741dde4f62f5fedf035910368ac433/src/two.js#L1141
     */
    	function createLoop(fn) {
    		let elapsed = 0;
    		let lastTime = performance.now();

    		(function loop() {
    			frame = requestAnimationFrame(loop);
    			const beginTime = performance.now();
    			const dt = (beginTime - lastTime) / 1000;
    			lastTime = beginTime;
    			elapsed += dt;
    			fn(elapsed, dt);
    		})();

    		return () => {
    			cancelAnimationFrame(frame);
    		};
    	}

    	const writable_props = ['killLoopOnError', 'attributes'];

    	Object.keys($$props).forEach(key => {
    		if (!~writable_props.indexOf(key) && key.slice(0, 2) !== '$$' && key !== 'slot') console_1$1.warn(`<Canvas> was created with unknown prop '${key}'`);
    	});

    	function canvas_1_binding($$value) {
    		binding_callbacks[$$value ? 'unshift' : 'push'](() => {
    			canvas$1 = $$value;
    			$$invalidate(0, canvas$1);
    		});
    	}

    	$$self.$$set = $$props => {
    		if ('killLoopOnError' in $$props) $$invalidate(5, killLoopOnError = $$props.killLoopOnError);
    		if ('attributes' in $$props) $$invalidate(6, attributes = $$props.attributes);
    		if ('$$scope' in $$props) $$invalidate(7, $$scope = $$props.$$scope);
    	};

    	$$self.$capture_state = () => ({
    		onMount,
    		onDestroy,
    		setContext,
    		key,
    		width,
    		height,
    		canvasStore: canvas,
    		contextStore: context,
    		pixelRatio,
    		props,
    		update_clock,
    		killLoopOnError,
    		attributes,
    		listeners,
    		canvas: canvas$1,
    		context: context$1,
    		frame,
    		handleResize,
    		render,
    		createLoop,
    		$props,
    		$pixelRatio,
    		$width,
    		$height
    	});

    	$$self.$inject_state = $$props => {
    		if ('killLoopOnError' in $$props) $$invalidate(5, killLoopOnError = $$props.killLoopOnError);
    		if ('attributes' in $$props) $$invalidate(6, attributes = $$props.attributes);
    		if ('listeners' in $$props) listeners = $$props.listeners;
    		if ('canvas' in $$props) $$invalidate(0, canvas$1 = $$props.canvas);
    		if ('context' in $$props) context$1 = $$props.context;
    		if ('frame' in $$props) frame = $$props.frame;
    	};

    	if ($$props && "$$inject" in $$props) {
    		$$self.$inject_state($$props.$$inject);
    	}

    	return [
    		canvas$1,
    		$pixelRatio,
    		$width,
    		$height,
    		handleResize,
    		killLoopOnError,
    		attributes,
    		$$scope,
    		slots,
    		canvas_1_binding
    	];
    }

    class Canvas extends SvelteComponentDev {
    	constructor(options) {
    		super(options);
    		init(this, options, instance$4, create_fragment$4, safe_not_equal, { killLoopOnError: 5, attributes: 6 });

    		dispatch_dev("SvelteRegisterComponent", {
    			component: this,
    			tagName: "Canvas",
    			options,
    			id: create_fragment$4.name
    		});
    	}

    	get killLoopOnError() {
    		throw new Error("<Canvas>: Props cannot be read directly from the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	set killLoopOnError(value) {
    		throw new Error("<Canvas>: Props cannot be set directly on the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	get attributes() {
    		throw new Error("<Canvas>: Props cannot be read directly from the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	set attributes(value) {
    		throw new Error("<Canvas>: Props cannot be set directly on the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}
    }

    /* src/Background.svelte generated by Svelte v3.49.0 */

    function create_fragment$3(ctx) {
    	let current;
    	const default_slot_template = /*#slots*/ ctx[2].default;
    	const default_slot = create_slot(default_slot_template, ctx, /*$$scope*/ ctx[1], null);

    	const block = {
    		c: function create() {
    			if (default_slot) default_slot.c();
    		},
    		l: function claim(nodes) {
    			throw new Error("options.hydrate only works if the component was compiled with the `hydratable: true` option");
    		},
    		m: function mount(target, anchor) {
    			if (default_slot) {
    				default_slot.m(target, anchor);
    			}

    			current = true;
    		},
    		p: function update(ctx, [dirty]) {
    			if (default_slot) {
    				if (default_slot.p && (!current || dirty & /*$$scope*/ 2)) {
    					update_slot_base(
    						default_slot,
    						default_slot_template,
    						ctx,
    						/*$$scope*/ ctx[1],
    						!current
    						? get_all_dirty_from_scope(/*$$scope*/ ctx[1])
    						: get_slot_changes(default_slot_template, /*$$scope*/ ctx[1], dirty, null),
    						null
    					);
    				}
    			}
    		},
    		i: function intro(local) {
    			if (current) return;
    			transition_in(default_slot, local);
    			current = true;
    		},
    		o: function outro(local) {
    			transition_out(default_slot, local);
    			current = false;
    		},
    		d: function destroy(detaching) {
    			if (default_slot) default_slot.d(detaching);
    		}
    	};

    	dispatch_dev("SvelteRegisterBlock", {
    		block,
    		id: create_fragment$3.name,
    		type: "component",
    		source: "",
    		ctx
    	});

    	return block;
    }

    function instance$3($$self, $$props, $$invalidate) {
    	let { $$slots: slots = {}, $$scope } = $$props;
    	validate_slots('Background', slots, ['default']);
    	let { color = null } = $$props;

    	renderable(props => {
    		const { context, width, height } = props;
    		context.clearRect(0, 0, width, height);

    		if (color) {
    			context.fillStyle = color;
    			context.fillRect(0, 0, width, height);
    		}
    	});

    	const writable_props = ['color'];

    	Object.keys($$props).forEach(key => {
    		if (!~writable_props.indexOf(key) && key.slice(0, 2) !== '$$' && key !== 'slot') console.warn(`<Background> was created with unknown prop '${key}'`);
    	});

    	$$self.$$set = $$props => {
    		if ('color' in $$props) $$invalidate(0, color = $$props.color);
    		if ('$$scope' in $$props) $$invalidate(1, $$scope = $$props.$$scope);
    	};

    	$$self.$capture_state = () => ({ renderable, color });

    	$$self.$inject_state = $$props => {
    		if ('color' in $$props) $$invalidate(0, color = $$props.color);
    	};

    	if ($$props && "$$inject" in $$props) {
    		$$self.$inject_state($$props.$$inject);
    	}

    	return [color, $$scope, slots];
    }

    class Background extends SvelteComponentDev {
    	constructor(options) {
    		super(options);
    		init(this, options, instance$3, create_fragment$3, safe_not_equal, { color: 0 });

    		dispatch_dev("SvelteRegisterComponent", {
    			component: this,
    			tagName: "Background",
    			options,
    			id: create_fragment$3.name
    		});
    	}

    	get color() {
    		throw new Error("<Background>: Props cannot be read directly from the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	set color(value) {
    		throw new Error("<Background>: Props cannot be set directly on the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}
    }

    /* src/Snippets.svelte generated by Svelte v3.49.0 */

    function create_fragment$2(ctx) {
    	const block = {
    		c: noop,
    		l: function claim(nodes) {
    			throw new Error("options.hydrate only works if the component was compiled with the `hydratable: true` option");
    		},
    		m: noop,
    		p: noop,
    		i: noop,
    		o: noop,
    		d: noop
    	};

    	dispatch_dev("SvelteRegisterBlock", {
    		block,
    		id: create_fragment$2.name,
    		type: "component",
    		source: "",
    		ctx
    	});

    	return block;
    }

    const OUTER_PADDING = 10;
    const BOX_PADDING_FRACTION = 0.1;
    const BOX_Y_START = 70;

    function instance$2($$self, $$props, $$invalidate) {
    	let { $$slots: slots = {}, $$scope } = $$props;
    	validate_slots('Snippets', slots, []);
    	let { num_x_boxes = 55 } = $$props;

    	function kth_box(k, c_width) {
    		const available_width = c_width - 2 * OUTER_PADDING;
    		const box_space = available_width / num_x_boxes;
    		const x_idx = k % num_x_boxes;
    		const y_idx = Math.floor(k / num_x_boxes);

    		// (x,y)
    		const box_space_T_left_x = OUTER_PADDING + x_idx * box_space;

    		const box_space_T_left_y = OUTER_PADDING + y_idx * box_space;
    		const padding = box_space * BOX_PADDING_FRACTION;
    		const box_TL_y = BOX_Y_START + box_space_T_left_y + padding / 2;
    		const box_TL_x = box_space_T_left_x + padding / 2;
    		const size = box_space - padding;
    		return [box_TL_x, box_TL_y, size, size];
    	}

    	function draw_sample(sample, idx, width, ctx) {
    		const box = kth_box(idx, width);
    		ctx.lineWidth = 2;
    		ctx.strokeStyle = 'rgb(0, 0, 0)';
    		ctx.fillStyle = 'rgb(0, 0, 0)';
    		ctx.fillStyle = `rgb(${sample[3] * 255}, 0, ${sample[3] * 255})`;
    		ctx.fillRect(...box);
    		const dr = 3;
    		const smaller_box = [box[0] + dr, box[1] + dr, box[2] - 2 * dr, box[3] - 2 * dr];
    		let rgb = [sample[0] * 250, sample[1] * 240, sample[2] * 210];
    		ctx.fillStyle = `rgb(${rgb[0]}, ${rgb[1]}, ${rgb[2]})`;
    		ctx.fillRect(...smaller_box);
    	}

    	renderable((props, dt) => {
    		const { context, width, height } = props;
    		let snippets$1 = snippets();
    		let sample_idx$1 = sample_idx();

    		for (let i = 0; i < snippets$1.length; i++) {
    			const sample = snippets$1[i][sample_idx$1];
    			draw_sample(sample, i, width, context);
    		}
    	});

    	const writable_props = ['num_x_boxes'];

    	Object.keys($$props).forEach(key => {
    		if (!~writable_props.indexOf(key) && key.slice(0, 2) !== '$$' && key !== 'slot') console.warn(`<Snippets> was created with unknown prop '${key}'`);
    	});

    	$$self.$$set = $$props => {
    		if ('num_x_boxes' in $$props) $$invalidate(0, num_x_boxes = $$props.num_x_boxes);
    	};

    	$$self.$capture_state = () => ({
    		engine,
    		OUTER_PADDING,
    		BOX_PADDING_FRACTION,
    		BOX_Y_START,
    		num_x_boxes,
    		kth_box,
    		draw_sample
    	});

    	$$self.$inject_state = $$props => {
    		if ('num_x_boxes' in $$props) $$invalidate(0, num_x_boxes = $$props.num_x_boxes);
    	};

    	if ($$props && "$$inject" in $$props) {
    		$$self.$inject_state($$props.$$inject);
    	}

    	return [num_x_boxes];
    }

    class Snippets extends SvelteComponentDev {
    	constructor(options) {
    		super(options);
    		init(this, options, instance$2, create_fragment$2, safe_not_equal, { num_x_boxes: 0 });

    		dispatch_dev("SvelteRegisterComponent", {
    			component: this,
    			tagName: "Snippets",
    			options,
    			id: create_fragment$2.name
    		});
    	}

    	get num_x_boxes() {
    		throw new Error("<Snippets>: Props cannot be read directly from the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	set num_x_boxes(value) {
    		throw new Error("<Snippets>: Props cannot be set directly on the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}
    }

    /* src/Timeline.svelte generated by Svelte v3.49.0 */

    const { console: console_1 } = globals;

    function create_fragment$1(ctx) {
    	const block = {
    		c: noop,
    		l: function claim(nodes) {
    			throw new Error("options.hydrate only works if the component was compiled with the `hydratable: true` option");
    		},
    		m: noop,
    		p: noop,
    		i: noop,
    		o: noop,
    		d: noop
    	};

    	dispatch_dev("SvelteRegisterBlock", {
    		block,
    		id: create_fragment$1.name,
    		type: "component",
    		source: "",
    		ctx
    	});

    	return block;
    }

    function instance$1($$self, $$props, $$invalidate) {
    	let { $$slots: slots = {}, $$scope } = $$props;
    	validate_slots('Timeline', slots, []);
    	const top_left = [100, 20];
    	const width = 800;
    	const height = 40;
    	const PADDING = [0, 0];
    	const inner_width = width - PADDING[0];
    	const inner_height = 40 - PADDING[1];
    	let is_dragging = false;
    	let pre_down_state = is_paused();

    	function pos(px, py) {
    		const res = [
    			top_left[0] + px * inner_width + PADDING[0] / 2,
    			top_left[1] + py * inner_height + PADDING[0] / 2
    		];

    		return res;
    	}

    	function x_to_rel(x) {
    		x = x - (top_left[0] + PADDING[0] / 2);
    		x = Math.max(x, 0);
    		x = Math.min(x, inner_width);
    		const rel = x / inner_width;
    		return rel;
    	}

    	function draw_spike(sample_idx, width, ctx) {
    		// Line
    		const fraction = sample_idx / snippet_len();

    		const line_y = 0.5;

    		// Dot
    		const dot_y = line_y;

    		const dot_radius = 3;

    		// Text
    		const text_y = dot_y + 0.4;

    		const fontFamily = 'sans-serif';
    		const fontSize = 12;
    		const align = 'center';
    		const baseline = 'middle';

    		// Draw line
    		ctx.strokeStyle = 'white';

    		ctx.stroke();
    		ctx.beginPath();
    		ctx.moveTo(...pos(0, line_y));
    		ctx.lineTo(...pos(1, line_y));
    		ctx.stroke();

    		// Draw spike marker
    		ctx.strokeStyle = 'rgb(255, 150, 150)';

    		ctx.beginPath();
    		const spike_point_rel = 1 - snippet_pad() / snippet_len();
    		ctx.moveTo(...pos(spike_point_rel, 0.0));
    		ctx.lineTo(...pos(spike_point_rel, 1.0));
    		ctx.stroke();

    		// Draw moving dot
    		ctx.beginPath();

    		ctx.fillStyle = 'rgb(255, 255, 255)';
    		ctx.strokeStyle = ctx.fillStyle;
    		ctx.arc(...pos(fraction, dot_y), dot_radius, 0, 2 * Math.PI);

    		// Can't get fill working well.
    		ctx.fill();

    		// Draw moving text.
    		ctx.font = `${fontSize}px ${fontFamily}`;

    		ctx.textAlign = align;
    		ctx.textBaseline = baseline;
    		const text = `t = ${snippet_time().toFixed(2)} s`;
    		ctx.fillText(text, ...pos(fraction, text_y));
    	}

    	function is_inside(mouse_pos) {
    		const in_x = mouse_pos[0] > top_left[0] && mouse_pos[0] < top_left[0] + width;
    		const in_y = mouse_pos[1] > top_left[1] && mouse_pos[1] < top_left[1] + height;
    		return in_x && in_y;
    	}

    	function on_down(event) {
    		const pos = mouse_pos(event);
    		console.log(pos);

    		if (is_inside(pos)) {
    			event.preventDefault();
    			event.stopPropagation();
    			pre_down_state = pause_ctrl.current();
    			pause_ctrl.pause();
    			set_playback_time_rel(x_to_rel(pos[0]));
    			is_dragging = true;
    		}
    	}

    	function on_move(event) {
    		if (is_dragging) {
    			const pos = mouse_pos(event);
    			event.preventDefault();
    			event.stopPropagation();
    			set_playback_time_rel(x_to_rel(pos[0]));
    		}
    	}

    	function on_up(event) {
    		// Important to check is_dragging before resuming, as another control
    		// element may have done the pausing/unpausing.
    		if (is_dragging) {
    			// Only resume if it was running beforehand.
    			if (!pre_down_state) {
    				pause_ctrl.resume();
    			}

    			is_dragging = false;
    		}
    	}

    	renderable({
    		"render": (props, dt) => {
    			const { context, width, height } = props;
    			let sample_idx$1 = sample_idx();
    			draw_spike(sample_idx$1, width, context);
    		},
    		"setup": props => {
    			props.canvas.addEventListener('mousedown', on_down);
    			props.canvas.addEventListener('mouseup', on_up);
    			props.canvas.addEventListener('mousemove', on_move);
    			window.addEventListener('mouseup', on_up);
    		}
    	});

    	const writable_props = [];

    	Object.keys($$props).forEach(key => {
    		if (!~writable_props.indexOf(key) && key.slice(0, 2) !== '$$' && key !== 'slot') console_1.warn(`<Timeline> was created with unknown prop '${key}'`);
    	});

    	$$self.$capture_state = () => ({
    		engine,
    		top_left,
    		width,
    		height,
    		PADDING,
    		inner_width,
    		inner_height,
    		is_dragging,
    		pre_down_state,
    		pos,
    		x_to_rel,
    		draw_spike,
    		is_inside,
    		on_down,
    		on_move,
    		on_up
    	});

    	$$self.$inject_state = $$props => {
    		if ('is_dragging' in $$props) is_dragging = $$props.is_dragging;
    		if ('pre_down_state' in $$props) pre_down_state = $$props.pre_down_state;
    	};

    	if ($$props && "$$inject" in $$props) {
    		$$self.$inject_state($$props.$$inject);
    	}

    	return [top_left, width, height];
    }

    class Timeline extends SvelteComponentDev {
    	constructor(options) {
    		super(options);
    		init(this, options, instance$1, create_fragment$1, safe_not_equal, { top_left: 0, width: 1, height: 2 });

    		dispatch_dev("SvelteRegisterComponent", {
    			component: this,
    			tagName: "Timeline",
    			options,
    			id: create_fragment$1.name
    		});
    	}

    	get top_left() {
    		return this.$$.ctx[0];
    	}

    	set top_left(value) {
    		throw new Error("<Timeline>: Props cannot be set directly on the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	get width() {
    		return this.$$.ctx[1];
    	}

    	set width(value) {
    		throw new Error("<Timeline>: Props cannot be set directly on the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}

    	get height() {
    		return this.$$.ctx[2];
    	}

    	set height(value) {
    		throw new Error("<Timeline>: Props cannot be set directly on the component instance unless compiling with 'accessors: true' or '<svelte:options accessors/>'");
    	}
    }

    /* src/App.svelte generated by Svelte v3.49.0 */
    const file = "src/App.svelte";

    function get_each_context(ctx, list, i) {
    	const child_ctx = ctx.slice();
    	child_ctx[12] = list[i];
    	return child_ctx;
    }

    function get_each_context_1(ctx, list, i) {
    	const child_ctx = ctx.slice();
    	child_ctx[16] = list[i];
    	return child_ctx;
    }

    // (1:0) <script>  import * as engine from './engine.js';  // Sadly, I don't know how to use the $store syntax through a namespace.  import {pause_ctrl}
    function create_catch_block(ctx) {
    	const block = { c: noop, m: noop, p: noop, d: noop };

    	dispatch_dev("SvelteRegisterBlock", {
    		block,
    		id: create_catch_block.name,
    		type: "catch",
    		source: "(1:0) <script>  import * as engine from './engine.js';  // Sadly, I don't know how to use the $store syntax through a namespace.  import {pause_ctrl}",
    		ctx
    	});

    	return block;
    }

    // (58:0) {:then rec_id_names}
    function create_then_block(ctx) {
    	let select;
    	let mounted;
    	let dispose;
    	let each_value_1 = /*rec_id_names*/ ctx[15];
    	validate_each_argument(each_value_1);
    	let each_blocks = [];

    	for (let i = 0; i < each_value_1.length; i += 1) {
    		each_blocks[i] = create_each_block_1(get_each_context_1(ctx, each_value_1, i));
    	}

    	const block = {
    		c: function create() {
    			select = element("select");

    			for (let i = 0; i < each_blocks.length; i += 1) {
    				each_blocks[i].c();
    			}

    			if (/*recording_id*/ ctx[0] === void 0) add_render_callback(() => /*select_change_handler*/ ctx[6].call(select));
    			add_location(select, file, 58, 0, 1225);
    		},
    		m: function mount(target, anchor) {
    			insert_dev(target, select, anchor);

    			for (let i = 0; i < each_blocks.length; i += 1) {
    				each_blocks[i].m(select, null);
    			}

    			select_option(select, /*recording_id*/ ctx[0]);

    			if (!mounted) {
    				dispose = listen_dev(select, "change", /*select_change_handler*/ ctx[6]);
    				mounted = true;
    			}
    		},
    		p: function update(ctx, dirty) {
    			if (dirty & /*get_rec_names*/ 0) {
    				each_value_1 = /*rec_id_names*/ ctx[15];
    				validate_each_argument(each_value_1);
    				let i;

    				for (i = 0; i < each_value_1.length; i += 1) {
    					const child_ctx = get_each_context_1(ctx, each_value_1, i);

    					if (each_blocks[i]) {
    						each_blocks[i].p(child_ctx, dirty);
    					} else {
    						each_blocks[i] = create_each_block_1(child_ctx);
    						each_blocks[i].c();
    						each_blocks[i].m(select, null);
    					}
    				}

    				for (; i < each_blocks.length; i += 1) {
    					each_blocks[i].d(1);
    				}

    				each_blocks.length = each_value_1.length;
    			}

    			if (dirty & /*recording_id, get_rec_names*/ 1) {
    				select_option(select, /*recording_id*/ ctx[0]);
    			}
    		},
    		d: function destroy(detaching) {
    			if (detaching) detach_dev(select);
    			destroy_each(each_blocks, detaching);
    			mounted = false;
    			dispose();
    		}
    	};

    	dispatch_dev("SvelteRegisterBlock", {
    		block,
    		id: create_then_block.name,
    		type: "then",
    		source: "(58:0) {:then rec_id_names}",
    		ctx
    	});

    	return block;
    }

    // (60:0) {#each rec_id_names as rec}
    function create_each_block_1(ctx) {
    	let option;
    	let t0_value = /*rec*/ ctx[16].name + "";
    	let t0;
    	let t1;

    	const block = {
    		c: function create() {
    			option = element("option");
    			t0 = text(t0_value);
    			t1 = space();
    			option.__value = /*rec*/ ctx[16].id;
    			option.value = option.__value;
    			add_location(option, file, 60, 1, 1289);
    		},
    		m: function mount(target, anchor) {
    			insert_dev(target, option, anchor);
    			append_dev(option, t0);
    			append_dev(option, t1);
    		},
    		p: noop,
    		d: function destroy(detaching) {
    			if (detaching) detach_dev(option);
    		}
    	};

    	dispatch_dev("SvelteRegisterBlock", {
    		block,
    		id: create_each_block_1.name,
    		type: "each",
    		source: "(60:0) {#each rec_id_names as rec}",
    		ctx
    	});

    	return block;
    }

    // (56:24)  <p>Loading...</p> {:then rec_id_names}
    function create_pending_block(ctx) {
    	let p;

    	const block = {
    		c: function create() {
    			p = element("p");
    			p.textContent = "Loading...";
    			add_location(p, file, 56, 0, 1186);
    		},
    		m: function mount(target, anchor) {
    			insert_dev(target, p, anchor);
    		},
    		p: noop,
    		d: function destroy(detaching) {
    			if (detaching) detach_dev(p);
    		}
    	};

    	dispatch_dev("SvelteRegisterBlock", {
    		block,
    		id: create_pending_block.name,
    		type: "pending",
    		source: "(56:24)  <p>Loading...</p> {:then rec_id_names}",
    		ctx
    	});

    	return block;
    }

    // (71:0) {#each cluster_ids as c_id}
    function create_each_block(ctx) {
    	let div;
    	let button;
    	let t0_value = /*c_id*/ ctx[12] + "";
    	let t0;
    	let t1;
    	let div_class_value;
    	let mounted;
    	let dispose;

    	function click_handler() {
    		return /*click_handler*/ ctx[7](/*c_id*/ ctx[12]);
    	}

    	const block = {
    		c: function create() {
    			div = element("div");
    			button = element("button");
    			t0 = text(t0_value);
    			t1 = space();
    			attr_dev(button, "class", "svelte-1npolz2");
    			add_location(button, file, 72, 0, 1514);

    			attr_dev(div, "class", div_class_value = "" + (null_to_empty(/*c_id*/ ctx[12] == /*cluster_id*/ ctx[3]
    			? 'current cluster_box'
    			: 'cluster_box') + " svelte-1npolz2"));

    			add_location(div, file, 71, 0, 1439);
    		},
    		m: function mount(target, anchor) {
    			insert_dev(target, div, anchor);
    			append_dev(div, button);
    			append_dev(button, t0);
    			append_dev(div, t1);

    			if (!mounted) {
    				dispose = listen_dev(button, "click", click_handler, false, false, false);
    				mounted = true;
    			}
    		},
    		p: function update(new_ctx, dirty) {
    			ctx = new_ctx;
    			if (dirty & /*cluster_ids*/ 4 && t0_value !== (t0_value = /*c_id*/ ctx[12] + "")) set_data_dev(t0, t0_value);

    			if (dirty & /*cluster_ids, cluster_id*/ 12 && div_class_value !== (div_class_value = "" + (null_to_empty(/*c_id*/ ctx[12] == /*cluster_id*/ ctx[3]
    			? 'current cluster_box'
    			: 'cluster_box') + " svelte-1npolz2"))) {
    				attr_dev(div, "class", div_class_value);
    			}
    		},
    		d: function destroy(detaching) {
    			if (detaching) detach_dev(div);
    			mounted = false;
    			dispose();
    		}
    	};

    	dispatch_dev("SvelteRegisterBlock", {
    		block,
    		id: create_each_block.name,
    		type: "each",
    		source: "(71:0) {#each cluster_ids as c_id}",
    		ctx
    	});

    	return block;
    }

    // (87:0) <Canvas>
    function create_default_slot(ctx) {
    	let background;
    	let t0;
    	let timeline;
    	let t1;
    	let snippets;
    	let current;

    	background = new Background({
    			props: { color: "hsl(0, 0%, 10%)" },
    			$$inline: true
    		});

    	timeline = new Timeline({ $$inline: true });

    	snippets = new Snippets({
    			props: { num_x_boxes: /*num_x_boxes*/ ctx[1] },
    			$$inline: true
    		});

    	const block = {
    		c: function create() {
    			create_component(background.$$.fragment);
    			t0 = space();
    			create_component(timeline.$$.fragment);
    			t1 = space();
    			create_component(snippets.$$.fragment);
    		},
    		m: function mount(target, anchor) {
    			mount_component(background, target, anchor);
    			insert_dev(target, t0, anchor);
    			mount_component(timeline, target, anchor);
    			insert_dev(target, t1, anchor);
    			mount_component(snippets, target, anchor);
    			current = true;
    		},
    		p: function update(ctx, dirty) {
    			const snippets_changes = {};
    			if (dirty & /*num_x_boxes*/ 2) snippets_changes.num_x_boxes = /*num_x_boxes*/ ctx[1];
    			snippets.$set(snippets_changes);
    		},
    		i: function intro(local) {
    			if (current) return;
    			transition_in(background.$$.fragment, local);
    			transition_in(timeline.$$.fragment, local);
    			transition_in(snippets.$$.fragment, local);
    			current = true;
    		},
    		o: function outro(local) {
    			transition_out(background.$$.fragment, local);
    			transition_out(timeline.$$.fragment, local);
    			transition_out(snippets.$$.fragment, local);
    			current = false;
    		},
    		d: function destroy(detaching) {
    			destroy_component(background, detaching);
    			if (detaching) detach_dev(t0);
    			destroy_component(timeline, detaching);
    			if (detaching) detach_dev(t1);
    			destroy_component(snippets, detaching);
    		}
    	};

    	dispatch_dev("SvelteRegisterBlock", {
    		block,
    		id: create_default_slot.name,
    		type: "slot",
    		source: "(87:0) <Canvas>",
    		ctx
    	});

    	return block;
    }

    function create_fragment(ctx) {
    	let h2;
    	let t1;
    	let p;
    	let t3;
    	let h30;
    	let t5;
    	let ul;
    	let t6;
    	let h31;
    	let t8;
    	let div0;
    	let t9;
    	let div1;
    	let label;
    	let t10;
    	let input0;
    	let t11;
    	let input1;
    	let t12;
    	let button;
    	let t13_value = (/*_pause*/ ctx[4] ? '▶️' : '⏸️') + "";
    	let t13;
    	let t14;
    	let canvas;
    	let current;
    	let mounted;
    	let dispose;

    	let info = {
    		ctx,
    		current: null,
    		token: null,
    		hasCatch: false,
    		pending: create_pending_block,
    		then: create_then_block,
    		catch: create_catch_block,
    		value: 15
    	};

    	handle_promise(get_rec_names(), info);
    	let each_value = /*cluster_ids*/ ctx[2];
    	validate_each_argument(each_value);
    	let each_blocks = [];

    	for (let i = 0; i < each_value.length; i += 1) {
    		each_blocks[i] = create_each_block(get_each_context(ctx, each_value, i));
    	}

    	canvas = new Canvas({
    			props: {
    				$$slots: { default: [create_default_slot] },
    				$$scope: { ctx }
    			},
    			$$inline: true
    		});

    	const block = {
    		c: function create() {
    			h2 = element("h2");
    			h2.textContent = "MEA snippets";
    			t1 = space();
    			p = element("p");
    			p.textContent = "Snippets for each cluster.";
    			t3 = space();
    			h30 = element("h3");
    			h30.textContent = "Recordings";
    			t5 = space();
    			ul = element("ul");
    			info.block.c();
    			t6 = space();
    			h31 = element("h3");
    			h31.textContent = "Clusters";
    			t8 = space();
    			div0 = element("div");

    			for (let i = 0; i < each_blocks.length; i += 1) {
    				each_blocks[i].c();
    			}

    			t9 = space();
    			div1 = element("div");
    			label = element("label");
    			t10 = text("Horizontal boxes:\n");
    			input0 = element("input");
    			t11 = space();
    			input1 = element("input");
    			t12 = space();
    			button = element("button");
    			t13 = text(t13_value);
    			t14 = space();
    			create_component(canvas.$$.fragment);
    			add_location(h2, file, 51, 0, 1080);
    			add_location(p, file, 52, 0, 1102);
    			add_location(h30, file, 53, 0, 1136);
    			add_location(ul, file, 54, 0, 1156);
    			add_location(h31, file, 68, 0, 1370);
    			attr_dev(div0, "class", "clusters svelte-1npolz2");
    			add_location(div0, file, 69, 0, 1388);
    			attr_dev(input0, "type", "number");
    			attr_dev(input0, "name", "hbox-count");
    			attr_dev(input0, "min", "20");
    			attr_dev(input0, "max", "200");
    			attr_dev(input0, "class", "svelte-1npolz2");
    			add_location(input0, file, 79, 0, 1628);
    			attr_dev(input1, "type", "range");
    			attr_dev(input1, "name", "hbox-count-range");
    			attr_dev(input1, "min", "20");
    			attr_dev(input1, "max", "100");
    			attr_dev(input1, "class", "svelte-1npolz2");
    			add_location(input1, file, 80, 0, 1714);
    			attr_dev(button, "type", "button");
    			attr_dev(button, "name", "play-pause");
    			add_location(button, file, 81, 0, 1805);
    			attr_dev(label, "class", "svelte-1npolz2");
    			add_location(label, file, 78, 0, 1603);
    			attr_dev(div1, "class", "svelte-1npolz2");
    			add_location(div1, file, 77, 0, 1597);
    		},
    		l: function claim(nodes) {
    			throw new Error("options.hydrate only works if the component was compiled with the `hydratable: true` option");
    		},
    		m: function mount(target, anchor) {
    			insert_dev(target, h2, anchor);
    			insert_dev(target, t1, anchor);
    			insert_dev(target, p, anchor);
    			insert_dev(target, t3, anchor);
    			insert_dev(target, h30, anchor);
    			insert_dev(target, t5, anchor);
    			insert_dev(target, ul, anchor);
    			info.block.m(ul, info.anchor = null);
    			info.mount = () => ul;
    			info.anchor = null;
    			insert_dev(target, t6, anchor);
    			insert_dev(target, h31, anchor);
    			insert_dev(target, t8, anchor);
    			insert_dev(target, div0, anchor);

    			for (let i = 0; i < each_blocks.length; i += 1) {
    				each_blocks[i].m(div0, null);
    			}

    			insert_dev(target, t9, anchor);
    			insert_dev(target, div1, anchor);
    			append_dev(div1, label);
    			append_dev(label, t10);
    			append_dev(label, input0);
    			set_input_value(input0, /*num_x_boxes*/ ctx[1]);
    			append_dev(label, t11);
    			append_dev(label, input1);
    			set_input_value(input1, /*num_x_boxes*/ ctx[1]);
    			append_dev(label, t12);
    			append_dev(label, button);
    			append_dev(button, t13);
    			insert_dev(target, t14, anchor);
    			mount_component(canvas, target, anchor);
    			current = true;

    			if (!mounted) {
    				dispose = [
    					listen_dev(input0, "input", /*input0_input_handler*/ ctx[8]),
    					listen_dev(input1, "change", /*input1_change_input_handler*/ ctx[9]),
    					listen_dev(input1, "input", /*input1_change_input_handler*/ ctx[9]),
    					listen_dev(button, "click", /*click_handler_1*/ ctx[10], false, false, false)
    				];

    				mounted = true;
    			}
    		},
    		p: function update(new_ctx, [dirty]) {
    			ctx = new_ctx;
    			update_await_block_branch(info, ctx, dirty);

    			if (dirty & /*cluster_ids, cluster_id, load_cluster*/ 44) {
    				each_value = /*cluster_ids*/ ctx[2];
    				validate_each_argument(each_value);
    				let i;

    				for (i = 0; i < each_value.length; i += 1) {
    					const child_ctx = get_each_context(ctx, each_value, i);

    					if (each_blocks[i]) {
    						each_blocks[i].p(child_ctx, dirty);
    					} else {
    						each_blocks[i] = create_each_block(child_ctx);
    						each_blocks[i].c();
    						each_blocks[i].m(div0, null);
    					}
    				}

    				for (; i < each_blocks.length; i += 1) {
    					each_blocks[i].d(1);
    				}

    				each_blocks.length = each_value.length;
    			}

    			if (dirty & /*num_x_boxes*/ 2 && to_number(input0.value) !== /*num_x_boxes*/ ctx[1]) {
    				set_input_value(input0, /*num_x_boxes*/ ctx[1]);
    			}

    			if (dirty & /*num_x_boxes*/ 2) {
    				set_input_value(input1, /*num_x_boxes*/ ctx[1]);
    			}

    			if ((!current || dirty & /*_pause*/ 16) && t13_value !== (t13_value = (/*_pause*/ ctx[4] ? '▶️' : '⏸️') + "")) set_data_dev(t13, t13_value);
    			const canvas_changes = {};

    			if (dirty & /*$$scope, num_x_boxes*/ 524290) {
    				canvas_changes.$$scope = { dirty, ctx };
    			}

    			canvas.$set(canvas_changes);
    		},
    		i: function intro(local) {
    			if (current) return;
    			transition_in(canvas.$$.fragment, local);
    			current = true;
    		},
    		o: function outro(local) {
    			transition_out(canvas.$$.fragment, local);
    			current = false;
    		},
    		d: function destroy(detaching) {
    			if (detaching) detach_dev(h2);
    			if (detaching) detach_dev(t1);
    			if (detaching) detach_dev(p);
    			if (detaching) detach_dev(t3);
    			if (detaching) detach_dev(h30);
    			if (detaching) detach_dev(t5);
    			if (detaching) detach_dev(ul);
    			info.block.d();
    			info.token = null;
    			info = null;
    			if (detaching) detach_dev(t6);
    			if (detaching) detach_dev(h31);
    			if (detaching) detach_dev(t8);
    			if (detaching) detach_dev(div0);
    			destroy_each(each_blocks, detaching);
    			if (detaching) detach_dev(t9);
    			if (detaching) detach_dev(div1);
    			if (detaching) detach_dev(t14);
    			destroy_component(canvas, detaching);
    			mounted = false;
    			run_all(dispose);
    		}
    	};

    	dispatch_dev("SvelteRegisterBlock", {
    		block,
    		id: create_fragment.name,
    		type: "component",
    		source: "",
    		ctx
    	});

    	return block;
    }

    function get_rec_names() {
    	const res = fetch('/api/recordings').then(response => response.json());
    	return res;
    }

    function instance($$self, $$props, $$invalidate) {
    	let { $$slots: slots = {}, $$scope } = $$props;
    	validate_slots('App', slots, []);
    	let num_x_boxes = 30;
    	let recording_id = 0;
    	let cluster_ids = [];
    	let cluster_id = 0;

    	function update_cluster_ids() {
    		fetch(`/api/recording/${recording_id}/clusters`).then(response => response.json()).then(c_ids => {
    			c_ids.sort((a, b) => a - b);
    			$$invalidate(2, cluster_ids = c_ids);
    		});
    	}

    	async function load_cluster(idx) {
    		const res = await fetch(`/api/recording/${recording_id}/cluster/${idx}`);
    		let s = await res.json();
    		set_snippets(s);
    		$$invalidate(3, cluster_id = idx);
    		return s;
    	}

    	let _pause = false;

    	pause_ctrl.subscribe(value => {
    		$$invalidate(4, _pause = value);
    	});

    	const writable_props = [];

    	Object.keys($$props).forEach(key => {
    		if (!~writable_props.indexOf(key) && key.slice(0, 2) !== '$$' && key !== 'slot') console.warn(`<App> was created with unknown prop '${key}'`);
    	});

    	function select_change_handler() {
    		recording_id = select_value(this);
    		$$invalidate(0, recording_id);
    	}

    	const click_handler = c_id => load_cluster(c_id);

    	function input0_input_handler() {
    		num_x_boxes = to_number(this.value);
    		$$invalidate(1, num_x_boxes);
    	}

    	function input1_change_input_handler() {
    		num_x_boxes = to_number(this.value);
    		$$invalidate(1, num_x_boxes);
    	}

    	const click_handler_1 = () => pause_ctrl.toggle();

    	$$self.$capture_state = () => ({
    		engine,
    		pause_ctrl,
    		Canvas,
    		Background,
    		Snippets,
    		Timeline,
    		num_x_boxes,
    		recording_id,
    		cluster_ids,
    		cluster_id,
    		get_rec_names,
    		update_cluster_ids,
    		load_cluster,
    		_pause
    	});

    	$$self.$inject_state = $$props => {
    		if ('num_x_boxes' in $$props) $$invalidate(1, num_x_boxes = $$props.num_x_boxes);
    		if ('recording_id' in $$props) $$invalidate(0, recording_id = $$props.recording_id);
    		if ('cluster_ids' in $$props) $$invalidate(2, cluster_ids = $$props.cluster_ids);
    		if ('cluster_id' in $$props) $$invalidate(3, cluster_id = $$props.cluster_id);
    		if ('_pause' in $$props) $$invalidate(4, _pause = $$props._pause);
    	};

    	if ($$props && "$$inject" in $$props) {
    		$$self.$inject_state($$props.$$inject);
    	}

    	$$self.$$.update = () => {
    		if ($$self.$$.dirty & /*recording_id*/ 1) {
    			(update_cluster_ids());
    		}
    	};

    	return [
    		recording_id,
    		num_x_boxes,
    		cluster_ids,
    		cluster_id,
    		_pause,
    		load_cluster,
    		select_change_handler,
    		click_handler,
    		input0_input_handler,
    		input1_change_input_handler,
    		click_handler_1
    	];
    }

    class App extends SvelteComponentDev {
    	constructor(options) {
    		super(options);
    		init(this, options, instance, create_fragment, safe_not_equal, {});

    		dispatch_dev("SvelteRegisterComponent", {
    			component: this,
    			tagName: "App",
    			options,
    			id: create_fragment.name
    		});
    	}
    }

    var app = new App({
    	target: document.querySelector("#svelte-app")
    });

    return app;

})();
//# sourceMappingURL=bundle.js.map
