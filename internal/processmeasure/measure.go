// Package processmeasure owns child-process wall and peak-memory measurement.
package processmeasure

import (
	"bytes"
	"errors"
	"fmt"
	"os/exec"
	"time"
)

const sampleInterval = 2 * time.Millisecond

type Result struct {
	Wall               time.Duration
	PeakWorkingSetByte uint64
	Output             []byte
}

type peakSample struct {
	bytes   uint64
	sampled bool
	err     error
}

// Measure runs one configured child and samples its process peak.
func Measure(command *exec.Cmd) (Result, error) {
	if command == nil || command.Process != nil {
		return Result{}, errors.New("process measure: command is nil or already started")
	}
	var output bytes.Buffer
	command.Stdout, command.Stderr = &output, &output
	started := time.Now()
	if err := command.Start(); err != nil {
		return Result{}, err
	}
	stop := make(chan struct{})
	finished := make(chan peakSample, 1)
	go samplePeak(command, stop, finished)
	waitErr := command.Wait()
	close(stop)
	sample := <-finished
	result := Result{
		Wall: time.Since(started), PeakWorkingSetByte: sample.bytes,
		Output: bytes.Clone(output.Bytes()),
	}
	if !sample.sampled {
		return result, errors.Join(waitErr, fmt.Errorf("process measure: peak unavailable: %w", sample.err))
	}
	return result, waitErr
}

func samplePeak(command *exec.Cmd, stop <-chan struct{}, finished chan<- peakSample) {
	ticker := time.Tick(sampleInterval)
	var result peakSample
	for {
		peak, err := peakWorkingSet(command.Process)
		if err == nil {
			result.sampled = true
			result.bytes = max(result.bytes, peak)
		} else {
			result.err = err
		}
		select {
		case <-stop:
			finished <- result
			return
		case <-ticker:
		}
	}
}
