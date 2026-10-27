package device

import (
	"context"
	"errors"
	"fmt"
	"runtime"
	"sync"
	"sync/atomic"

	"overgo/internal/cuda/driver"
)

// State: contains resources that are valid only on Worker's locked OS thread
type State struct {
	Driver  *driver.Library
	Device  driver.Device
	Context driver.Context
	Stream  driver.Stream
	// Trace: optional launch recorder; launch sites append their
	// configuration and skip device work while Trace.Probe is set.
	Trace *LaunchTrace
}

type request struct {
	ctx      context.Context
	function func(*State) error
	result   chan error
}

// Worker: owns one CUDA context and serializes access to it on locked OS
// thread
type Worker struct {
	requests chan request
	stop     chan chan error
	done     chan struct{}
	close    func() error
	// spinning holds the worker polling handed for a request (Spin), which handing lets one caller at a time fill.
	spinning atomic.Bool
	handing  sync.Mutex
	handed   atomic.Pointer[handedRequest]
}

// handedRequest is a request handed over to a spinning worker: its function, its error once ran is set.
type handedRequest struct {
	function func(*State) error
	err      error
	ran      atomic.Bool
}

// workerPolls is the polls of handed a spinning worker, or a caller waiting on its request, makes between looks at
// its channels.
const workerPolls = 1 << 8

// New: starts CUDA worker for device ordinal
func New(ordinal int) (*Worker, error) {
	worker := &Worker{
		requests: make(chan request),
		stop:     make(chan chan error),
		done:     make(chan struct{}),
	}
	worker.close = sync.OnceValue(worker.stopAndWait)
	initialized := make(chan error, 1)
	go worker.run(ordinal, initialized)
	if err := <-initialized; err != nil {
		<-worker.done
		return nil, err
	}
	return worker, nil
}

// Do: executes function on worker's locked CUDA thread
func (w *Worker) Do(ctx context.Context, function func(*State) error) error {
	if function == nil {
		return errors.New("CUDA worker: nil function")
	}
	if w.spinning.Load() {
		if err := ctx.Err(); err != nil {
			return err
		}
		w.handing.Lock()
		if w.spinning.Load() {
			defer w.handing.Unlock()
			handed := &handedRequest{function: function}
			w.handed.Store(handed)
			for polls := 1; !handed.ran.Load(); polls++ {
				if polls%workerPolls == 0 {
					select {
					case <-w.done:
						return errors.New("CUDA worker closed before request completed")
					default:
					}
				}
			}
			return handed.err
		}
		w.handing.Unlock()
	}
	result := make(chan error, 1)
	req := request{ctx: ctx, function: function, result: result}
	select {
	case w.requests <- req:
	case <-ctx.Done():
		return ctx.Err()
	case <-w.done:
		return errors.New("CUDA worker is closed")
	}
	select {
	case err := <-result:
		return err
	case <-w.done:
		return errors.New("CUDA worker closed before request completed")
	}
}

// Spin starts or stops the worker polling for requests in place of waiting on its channel, as a decode loop's host
// thread calls the driver itself (Strata's verify): a request then reaches the worker and its answer the caller
// without either waiting for its thread to be woken, about 10 to 40 us a call, several a decode window. A spinning
// worker holds a processor; the caller stops it once its loop ends.
func (w *Worker) Spin(ctx context.Context, on bool) error {
	w.handing.Lock()
	changed := w.spinning.Load() != on
	w.spinning.Store(on)
	w.handing.Unlock()
	if !on || !changed {
		return nil
	}
	// the worker waits on its channel: a request through it wakes it to poll
	result := make(chan error, 1)
	select {
	case w.requests <- request{ctx: ctx, function: func(*State) error { return nil }, result: result}:
	case <-ctx.Done():
		return ctx.Err()
	case <-w.done:
		return errors.New("CUDA worker is closed")
	}
	select {
	case err := <-result:
		return err
	case <-w.done:
		return errors.New("CUDA worker closed before request completed")
	}
}

// Close synchronizes and destroys worker's CUDA resources
func (w *Worker) Close() error {
	return w.close()
}

func (w *Worker) stopAndWait() error {
	result := make(chan error, 1)
	select {
	case w.stop <- result:
		return <-result
	case <-w.done:
		return nil
	}
}

func (w *Worker) MemoryStats(ctx context.Context) (driver.MemoryStats, error) {
	var result driver.MemoryStats
	err := w.Do(ctx, func(state *State) error {
		result = state.Driver.MemoryStats()
		return nil
	})
	return result, err
}

func (w *Worker) ExecutionStats(ctx context.Context) (driver.ExecutionStats, error) {
	var result driver.ExecutionStats
	err := w.Do(ctx, func(state *State) error {
		result = state.Driver.ExecutionStats()
		return nil
	})
	return result, err
}

func (w *Worker) run(ordinal int, initialized chan<- error) {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	defer close(w.done)

	state, err := initialize(ordinal)
	initialized <- err
	if err != nil {
		return
	}

	serve := func(req request) {
		if err := req.ctx.Err(); err != nil {
			req.result <- err
			return
		}
		req.result <- callSafely(req.function, state)
	}
	for polls := 1; ; polls++ {
		// A spinning worker runs what is handed to it, and looks at its channels now and then; it runs a request still
		// handed over before it stops.
		if w.spinning.Load() || w.handed.Load() != nil {
			if handed := w.handed.Swap(nil); handed != nil {
				handed.err = callSafely(handed.function, state)
				handed.ran.Store(true)
				continue
			}
			if polls%workerPolls != 0 {
				continue
			}
			select {
			case req := <-w.requests:
				serve(req)
			case result := <-w.stop:
				result <- shutdown(state)
				return
			default:
			}
			continue
		}
		select {
		case req := <-w.requests:
			serve(req)
		case result := <-w.stop:
			result <- shutdown(state)
			return
		}
	}
}

func initialize(ordinal int) (*State, error) {
	lib, err := driver.Open()
	if err != nil {
		return nil, err
	}
	fail := func(cause error) (*State, error) {
		_ = lib.Close()
		return nil, cause
	}
	if err := lib.Init(); err != nil {
		return fail(err)
	}
	device, err := lib.Device(ordinal)
	if err != nil {
		return fail(err)
	}
	cudaContext, err := lib.ContextCreate(device, 0)
	if err != nil {
		return fail(err)
	}
	if err := lib.ContextSetCurrent(cudaContext); err != nil {
		_ = lib.ContextDestroy(cudaContext)
		return fail(err)
	}
	stream, err := lib.StreamCreate(0)
	if err != nil {
		_ = lib.ContextDestroy(cudaContext)
		return fail(err)
	}
	return &State{
		Driver:  lib,
		Device:  device,
		Context: cudaContext,
		Stream:  stream,
	}, nil
}

func shutdown(state *State) error {
	var errs []error
	if err := state.Driver.StreamSynchronize(state.Stream); err != nil {
		errs = append(errs, err)
	}
	if err := state.Driver.StreamDestroy(state.Stream); err != nil {
		errs = append(errs, err)
	}
	if err := state.Driver.ContextDestroy(state.Context); err != nil {
		errs = append(errs, err)
	}
	if err := state.Driver.Close(); err != nil {
		errs = append(errs, err)
	}
	return errors.Join(errs...)
}

func callSafely(function func(*State) error, state *State) (err error) {
	defer func() {
		if value := recover(); value != nil {
			err = fmt.Errorf("panic in CUDA worker request: %v", value)
		}
	}()
	return function(state)
}
