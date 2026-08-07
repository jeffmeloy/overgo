package device

import (
	"context"
	"errors"
	"fmt"
	"runtime"
	"sync"

	"overgo/internal/cuda/driver"
)

// State: contains resources that are valid only on Worker's locked OS thread
type State struct {
	Driver  *driver.Library
	Device  driver.Device
	Context driver.Context
	Stream  driver.Stream
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
	once     sync.Once
}

// New: starts CUDA worker for device ordinal
func New(ordinal int) (*Worker, error) {
	worker := &Worker{
		requests: make(chan request),
		stop:     make(chan chan error),
		done:     make(chan struct{}),
	}
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

// Close synchronizes and destroys worker's CUDA resources
func (w *Worker) Close() error {
	var closeErr error
	w.once.Do(func() {
		result := make(chan error, 1)
		select {
		case w.stop <- result:
			closeErr = <-result
		case <-w.done:
		}
	})
	return closeErr
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

	for {
		select {
		case req := <-w.requests:
			if err := req.ctx.Err(); err != nil {
				req.result <- err
				continue
			}
			req.result <- callSafely(req.function, state)
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
