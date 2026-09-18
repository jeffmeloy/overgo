package testevidence

import (
	"errors"
	"maps"
	"sync"
)

// queuedPackageUpdates keeps persistence off the process-output drain. Only
// verdict transitions queue, never raw output; flush joins the worker and
// reports every publication error before command evidence can be accepted.
func queuedPackageUpdates(observe func(string, bool, map[string]string) error) (func(string, bool, map[string]string) error, func() error) {
	if observe == nil {
		return nil, func() error { return nil }
	}
	type update struct {
		name   string
		passed bool
		tests  map[string]string
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
				if err := observe(change.name, change.passed, change.tests); err != nil {
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
	return func(name string, passed bool, tests map[string]string) error {
			mutex.Lock()
			pending = append(pending, update{name: name, passed: passed, tests: maps.Clone(tests)})
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
	observe func(string, bool, map[string]string) error
	passed  map[string]bool
}

func includePackageResult(entry packageEvidence, result *testResult) packageEvidence {
	entry.incomplete = entry.incomplete || !result.started || result.ineligible || result.Action != "pass" && !result.excluded || result.Unavailable != ""
	entry.passed = entry.passed || result.Name == "" && result.Action == "pass" && !result.ineligible && result.Unavailable == ""
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
	var tests map[string]string
	if passed {
		tests = map[string]string{}
		for _, result := range results {
			if result.Package == name && result.Name != "" {
				tests[result.Name] = result.Action
			}
		}
	}
	if err := updates.observe(name, passed, tests); err != nil {
		return err
	}
	updates.passed[name] = passed
	return nil
}

func (updates *packageUpdates) invalidate() error {
	var result error
	for name, passed := range updates.passed {
		if passed {
			result = errors.Join(result, updates.observe(name, false, nil))
			updates.passed[name] = false
		}
	}
	return result
}
