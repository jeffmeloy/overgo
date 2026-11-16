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

	"overgo/internal/processmeasure"
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
	// bounded reads the tree's bound in boundOf's place, so a test bounds one tree without bounding another.
	bounded func(*exec.Cmd) (jobBound, error)
}

// Receipt is the terminal evidence of one supervised execution: how it
// ended, whether the tree had to be terminated, how much output each
// pipe drained, the measured wall time, the most memory any one process
// of the tree committed, zero where the operating system does not say, and
// the tree's memory limit (jobBound), zero where it had none, and whether it
// reached it.
type Receipt struct {
	ExitCode         int
	Interrupted      bool
	TreeTerminated   bool
	StdoutBytes      int64
	StderrBytes      int64
	WallNS           int64
	PeakProcessBytes uint64
	MemoryLimit      uint64
	MemoryLimited    bool
}

// Supervised is one running process tree under supervision.
type Supervised struct {
	command *exec.Cmd
	tree    processTree
	clock   processmeasure.Stopwatch

	drain       sync.WaitGroup
	stdoutBytes int64
	stderrBytes int64

	// exited closes when the process and both pipes are terminal;
	// waitErr carries the raw wait result for Wait to interpret.
	exited  chan struct{}
	waitErr error
	// limited says a commit of the tree would have passed its bound, read as the tree ended.
	limited bool

	mu         sync.Mutex
	terminated bool
	finished   bool
	receipt    Receipt
	receiptErr error
	// peak is the tree's peak process memory, read as it ended, before its job closed.
	peak uint64
}

// Start launches the command under tree containment and begins
// draining both pipes. A command that cannot start returns before any
// process exists. The tree commits no more than its command's recorded
// need and the commit free as it starts (jobBound).
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
	bounded := command.bounded
	if bounded == nil {
		bounded = boundOf
	}
	bound, err := bounded(run)
	if err != nil {
		return nil, err
	}
	resourceMu.Lock()
	defer resourceMu.Unlock()
	run.Env = resourceEnvironment(run.Environ())
	if err := run.Start(); err != nil {
		return nil, fmt.Errorf("processcontrol: start %s: %w", command.Path, err)
	}
	supervised := &Supervised{command: run, clock: processmeasure.NewStopwatch()}
	tree, err := newProcessTree(run, bound)
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
		limited, treeErr := supervised.tree.wait()
		supervised.limited = limited
		supervised.drain.Wait()
		supervised.waitErr = run.Wait()
		if treeErr != nil {
			supervised.waitErr = treeErr
		}
		close(supervised.exited)
	}()
	return supervised, nil
}

// Running says whether the process of pid that started at start (ProcessStart) still runs: a running process with the
// id that started at another time is another process, and the one asked about has ended. On 2026-11-12 the 307th
// process started after one with id 13696 had exited was given 13696, and ProcessAlive said the first ran; a landing's
// test stage starts hundreds. A start of zero, a record written before starts were kept, asks about the id alone.
func Running(pid int, start int64) bool {
	if !ProcessAlive(pid) {
		return false
	}
	if start == 0 {
		return true
	}
	now, read := ProcessStart(pid)
	return read && now == start
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
		err := s.receiptErr
		s.mu.Unlock()
		return receipt, err
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
		return s.receipt, s.receiptErr
	}
	s.peak = s.tree.peakProcessBytes()
	jobPeak := s.tree.peakJobBytes()
	_ = s.tree.close()
	s.finished = true
	wall, wallErr := s.clock.Elapsed()
	s.receipt = Receipt{
		ExitCode:         s.command.ProcessState.ExitCode(),
		TreeTerminated:   s.terminated,
		StdoutBytes:      s.stdoutBytes,
		StderrBytes:      s.stderrBytes,
		WallNS:           int64(wall),
		PeakProcessBytes: s.peak,
		MemoryLimit:      s.tree.bound.limit,
		MemoryLimited:    s.limited,
	}
	if wallErr != nil {
		s.receiptErr = fmt.Errorf("processcontrol: wall: %w", wallErr)
	}
	if _, ok := errors.AsType[*exec.ExitError](waitErr); waitErr != nil && !ok {
		s.receiptErr = errors.Join(s.receiptErr, fmt.Errorf("processcontrol: wait: %w", waitErr))
	}
	if ctx.Err() != nil {
		s.receiptErr = errors.Join(s.receiptErr, fmt.Errorf("processcontrol: cancellation terminated the tree: %w", ctx.Err()))
	}
	switch {
	case s.limited:
		s.receiptErr = errors.Join(s.receiptErr, s.tree.bound.reached())
	case s.receipt.ExitCode == 0 && s.receiptErr == nil && jobPeak > 0:
		// A run that passed by itself records what its command needs, the bound of its next run.
		recordJobNeed(s.tree.bound.key, jobPeak)
	}
	return s.receipt, s.receiptErr
}

// PeakProcessBytes is the most memory any one process of the tree has committed so far, or, once the tree has ended,
// at most: a reader of a long run learns which part of it raised the peak as the part ends.
func (s *Supervised) PeakProcessBytes() uint64 {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.finished {
		return s.peak
	}
	return s.tree.peakProcessBytes()
}

// Done closes when the process tree and its pipes reach a terminal state.
func (s *Supervised) Done() <-chan struct{} { return s.exited }

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
