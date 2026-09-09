package testevidence

import (
	"errors"
	"sync"
)

// queuedPackageUpdates keeps persistence off the process-output drain. Only
// verdict transitions queue, never raw output; flush joins the worker and
// reports every publication error before command evidence can be accepted.
func queuedPackageUpdates(observe func(string, bool) error) (func(string, bool) error, func() error) {
	if observe == nil {
		return nil, func() error { return nil }
	}
	type update struct {
		name   string
		passed bool
	}
	var mutex sync.Mutex
	wake := sync.NewCond(&mutex)
	var pending []update
	closed := false
	done := make(chan struct{})
	var result error
	go func() {
		var failures []error
		for {
			mutex.Lock()
			for len(pending) == 0 && !closed {
				wake.Wait()
			}
			batch, finished := pending, closed
			pending = nil
			mutex.Unlock()
			for _, change := range batch {
				if err := observe(change.name, change.passed); err != nil {
					failures = append(failures, err)
				}
			}
			if finished {
				result = errors.Join(failures...)
				close(done)
				return
			}
		}
	}()
	return func(name string, passed bool) error {
			mutex.Lock()
			pending = append(pending, update{name: name, passed: passed})
			wake.Signal()
			mutex.Unlock()
			return nil
		}, func() error {
			mutex.Lock()
			closed = true
			wake.Signal()
			mutex.Unlock()
			<-done
			return result
		}
}

// packageUpdates publishes terminal verdict changes; a later contradiction
// revokes the receipt. Unfinished siblings never alter completed packages.
type packageUpdates struct {
	observe func(string, bool) error
	passed  map[string]bool
}

func includePackageResult(entry packageEvidence, result *testResult) packageEvidence {
	entry.incomplete = entry.incomplete || !result.started || result.ineligible || result.Action != "pass" || result.Unavailable != ""
	entry.passed = entry.passed || result.Name == "" && result.Action == "pass"
	entry.tested = entry.tested || result.Name != "" && result.Action == "pass"
	return entry
}

func (updates *packageUpdates) update(results map[string]*testResult, name string, valid bool) error {
	terminal := results[name+"\x00"]
	if updates.observe == nil || terminal == nil || terminal.Action == "" {
		return nil
	}
	entry := packageEvidence{}
	for _, result := range results {
		if result.Package == name {
			entry = includePackageResult(entry, result)
		}
	}
	passed := valid && name != "" && entry.passed && entry.tested && !entry.incomplete
	if passed == updates.passed[name] {
		return nil
	}
	if err := updates.observe(name, passed); err != nil {
		return err
	}
	updates.passed[name] = passed
	return nil
}

func (updates *packageUpdates) invalidate() error {
	var result error
	for name, passed := range updates.passed {
		if passed {
			result = errors.Join(result, updates.observe(name, false))
			updates.passed[name] = false
		}
	}
	return result
}
