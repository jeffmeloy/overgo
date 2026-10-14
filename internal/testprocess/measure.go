package testprocess

import (
	"bytes"
	"errors"
	"fmt"
	"os/exec"
	"time"

	"overgo/internal/processmeasure"
)

// Result is the wall time, the peak working set and the combined output of one measured child.
type Result struct {
	Wall               time.Duration
	PeakWorkingSetByte uint64
	Output             []byte
}

// Measure runs one configured child and reads its peak working set from the
// kernel's own high-water mark once it has exited: a handle retained across
// the exit keeps the counters readable, and nothing samples the child.
func Measure(command *exec.Cmd) (Result, error) {
	if command == nil || command.Process != nil {
		return Result{}, errors.New("process measure: command is nil or already started")
	}
	var output bytes.Buffer
	command.Stdout, command.Stderr = &output, &output
	started, err := processmeasure.Counter()
	if err != nil {
		return Result{}, err
	}
	if err := command.Start(); err != nil {
		return Result{}, err
	}
	peak, peakErr := retainPeak(command.Process)
	waitErr := command.Wait()
	ended, clockErr := processmeasure.Counter()
	result := Result{Wall: ended - started, Output: bytes.Clone(output.Bytes())}
	if peakErr != nil {
		return result, errors.Join(waitErr, clockErr, fmt.Errorf("process measure: peak unavailable: %w", peakErr))
	}
	result.PeakWorkingSetByte, peakErr = peak.read()
	closeErr := peak.close()
	if peakErr != nil {
		return result, errors.Join(waitErr, clockErr, closeErr, fmt.Errorf("process measure: peak unavailable: %w", peakErr))
	}
	return result, errors.Join(waitErr, clockErr, closeErr)
}
