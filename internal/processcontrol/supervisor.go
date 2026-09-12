// Package processcontrol is the one owner of external process
// lifecycle: bounded start, cooperative interrupt, process-tree
// termination, output drain, and a termination receipt. Windows
// contains descendants with a job object; Unix uses process groups.
// Cancellation never reports success until the tree and both pipes
// reach a terminal state.
package processcontrol

import (
	"context"
	"errors"
	"fmt"
	"io"
	"os/exec"
	"sync"
	"time"
)

// Command describes one supervised external process.
type Command struct {
	Path string
	Args []string
	Dir  string
	Env  []string
	// Stdin feeds the process; nil means no input. A caller streaming
	// input over the process lifetime passes the read side of a pipe.
	Stdin io.Reader
	// Stdout and Stderr receive drained output; nil discards.
	Stdout io.Writer
	Stderr io.Writer
}

// Receipt is the terminal evidence of one supervised execution: how it
// ended, whether the tree had to be terminated, how much output each
// pipe drained, and the measured wall time.
type Receipt struct {
	ExitCode       int
	Interrupted    bool
	TreeTerminated bool
	StdoutBytes    int64
	StderrBytes    int64
	WallNS         int64
}

// Supervised is one running process tree under supervision.
type Supervised struct {
	command *exec.Cmd
	tree    processTree
	started time.Time

	drain       sync.WaitGroup
	stdoutBytes int64
	stderrBytes int64

	// exited closes when the process and both pipes are terminal;
	// waitErr carries the raw wait result for Wait to interpret.
	exited  chan struct{}
	waitErr error

	mu          sync.Mutex
	interrupted bool
	terminated  bool
	finished    bool
	receipt     Receipt
}

// Start launches the command under tree containment and begins
// draining both pipes. A command that cannot start returns before any
// process exists.
func Start(ctx context.Context, command Command) (*Supervised, error) {
	if ctx == nil || command.Path == "" {
		return nil, errors.New("processcontrol: nil context or empty command path")
	}
	if err := inheritedResources(); err != nil {
		return nil, fmt.Errorf("processcontrol: inherited physical resources: %w", err)
	}
	run := exec.Command(command.Path, command.Args...)
	run.Dir = command.Dir
	run.Env = command.Env
	run.Stdin = command.Stdin
	configureSysProc(run)
	stdout, err := run.StdoutPipe()
	if err != nil {
		return nil, fmt.Errorf("processcontrol: stdout pipe: %w", err)
	}
	stderr, err := run.StderrPipe()
	if err != nil {
		return nil, fmt.Errorf("processcontrol: stderr pipe: %w", err)
	}
	resourceMu.Lock()
	defer resourceMu.Unlock()
	run.Env = resourceEnvironment(run.Environ())
	if err := run.Start(); err != nil {
		return nil, fmt.Errorf("processcontrol: start %s: %w", command.Path, err)
	}
	supervised := &Supervised{command: run, started: time.Now()}
	tree, err := newProcessTree(run)
	if err != nil {
		_ = run.Process.Kill()
		_ = run.Wait()
		return nil, fmt.Errorf("processcontrol: contain %s: %w", command.Path, err)
	}
	stdoutSink, stderrSink := command.Stdout, command.Stderr
	if stdoutSink != nil && stdoutSink == stderrSink {
		shared := &lockedWriter{sink: stdoutSink}
		stdoutSink, stderrSink = shared, shared
	}
	supervised.tree = tree
	supervised.exited = make(chan struct{})
	supervised.drain.Add(2)
	go supervised.drainPipe(stdout, stdoutSink, &supervised.stdoutBytes)
	go supervised.drainPipe(stderr, stderrSink, &supervised.stderrBytes)
	go func() {
		treeErr := supervised.tree.wait()
		supervised.drain.Wait()
		supervised.waitErr = run.Wait()
		if treeErr != nil {
			supervised.waitErr = treeErr
		}
		close(supervised.exited)
	}()
	return supervised, nil
}

// lockedWriter serializes two drains aimed at one shared sink, so a
// caller asking for combined output never races its own buffer.
type lockedWriter struct {
	mu   sync.Mutex
	sink io.Writer
}

// Write forwards under the lock.
func (w *lockedWriter) Write(data []byte) (int, error) {
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.sink.Write(data)
}

func (s *Supervised) drainPipe(pipe io.Reader, sink io.Writer, counter *int64) {
	defer s.drain.Done()
	if sink == nil {
		sink = io.Discard
	}
	copied, _ := io.Copy(sink, pipe)
	s.mu.Lock()
	*counter = copied
	s.mu.Unlock()
}

// Interrupt asks the process to stop cooperatively where the platform
// can, recording the request; on platforms without a cooperative
// signal it is a no-op that still marks the receipt.
func (s *Supervised) Interrupt() error {
	s.mu.Lock()
	s.interrupted = true
	s.mu.Unlock()
	return s.tree.interrupt(s.command)
}

// Terminate kills the entire process tree, descendants included.
func (s *Supervised) Terminate() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.finished {
		return nil
	}
	s.terminated = true
	return s.tree.terminate()
}

// Wait blocks until the whole tree and both pipes reach a terminal
// state, then returns the termination receipt exactly once. A context
// deadline terminates the tree and still waits for the terminal state:
// success is never reported while anything is running.
func (s *Supervised) Wait(ctx context.Context) (Receipt, error) {
	if ctx == nil {
		return Receipt{}, errors.New("processcontrol: nil wait context")
	}
	s.mu.Lock()
	if s.finished {
		receipt := s.receipt
		s.mu.Unlock()
		return receipt, nil
	}
	s.mu.Unlock()

	select {
	case <-s.exited:
	case <-ctx.Done():
		_ = s.Terminate()
		<-s.exited
	}
	waitErr := s.waitErr

	s.mu.Lock()
	defer s.mu.Unlock()
	if s.finished {
		return s.receipt, nil
	}
	_ = s.tree.close()
	s.finished = true
	s.receipt = Receipt{
		ExitCode:       s.command.ProcessState.ExitCode(),
		Interrupted:    s.interrupted,
		TreeTerminated: s.terminated,
		StdoutBytes:    s.stdoutBytes,
		StderrBytes:    s.stderrBytes,
		WallNS:         time.Since(s.started).Nanoseconds(),
	}
	if _, ok := errors.AsType[*exec.ExitError](waitErr); waitErr != nil && !ok {
		return s.receipt, fmt.Errorf("processcontrol: wait: %w", waitErr)
	}
	if ctx.Err() != nil {
		return s.receipt, fmt.Errorf("processcontrol: deadline terminated the tree: %w", ctx.Err())
	}
	return s.receipt, nil
}

// Exited reports, without blocking, whether the process tree and its
// pipes have reached a terminal state.
func (s *Supervised) Exited() bool {
	select {
	case <-s.exited:
		return true
	default:
		return false
	}
}

// Run supervises one command to completion: start, wait for the tree
// and pipes, and return the receipt. The context deadline terminates
// the whole tree.
func Run(ctx context.Context, command Command) (Receipt, error) {
	supervised, err := Start(ctx, command)
	if err != nil {
		return Receipt{}, err
	}
	return supervised.Wait(ctx)
}
