package automationcheck

import (
	"context"
	"errors"
	"fmt"
	"maps"
	"sync"
)

// Executor runs one DAG invocation. Callers may inject exact cache handling.
type Executor func(context.Context, Invocation) (Evidence, error)

// DAGResult retains declaration order even when execution is concurrent.
type DAGResult struct {
	Invocation Invocation
	Evidence   Evidence
	Err        error
}

// ExecuteDAG runs ready independent checks concurrently while respecting
// dependencies and exclusive resource declarations. satisfied names work
// completed by an earlier policy barrier.
func ExecuteDAG(ctx context.Context, invocations []Invocation, satisfied map[string]bool, execute Executor) ([]DAGResult, error) {
	if ctx == nil || execute == nil {
		return nil, errors.New("automation check: DAG requires context and executor")
	}
	done := make(map[string]bool, len(satisfied)+len(invocations))
	maps.Copy(done, satisfied)
	seen := make(map[string]bool, len(invocations))
	for _, invocation := range invocations {
		if invocation.Check.Name == "" || seen[invocation.Check.Name] {
			return nil, fmt.Errorf("automation check: DAG duplicate or empty name %q", invocation.Check.Name)
		}
		seen[invocation.Check.Name] = true
	}
	results := make([]DAGResult, len(invocations))
	pending := make(map[int]bool, len(invocations))
	for index := range invocations {
		pending[index] = true
	}
	for len(pending) > 0 {
		wave := selectDAGWave(invocations, pending, done)
		if len(wave) == 0 {
			return nil, errors.New("automation check: DAG has unsatisfied dependencies or resource deadlock")
		}
		var wait sync.WaitGroup
		for _, index := range wave {
			delete(pending, index)
			wait.Go(func() {
				evidence, err := execute(ctx, invocations[index])
				results[index] = DAGResult{Invocation: invocations[index], Evidence: evidence, Err: err}
			})
		}
		wait.Wait()
		failed := false
		for _, index := range wave {
			if results[index].Err != nil {
				failed = true
				continue
			}
			done[invocations[index].Check.Name] = true
		}
		if failed {
			break
		}
	}
	return results, nil
}

func selectDAGWave(invocations []Invocation, pending map[int]bool, done map[string]bool) []int {
	type resourceUse struct {
		exclusive bool
		count     int
	}
	resources := map[string]resourceUse{}
	var wave []int
	for index, invocation := range invocations {
		if !pending[index] || !dependenciesDone(invocation.Check.Dependencies, done) {
			continue
		}
		compatible := true
		for _, resource := range invocation.Check.Resources {
			use := resources[resource.Name]
			if use.count > 0 && (use.exclusive || resource.Exclusive) {
				compatible = false
				break
			}
		}
		if !compatible {
			continue
		}
		wave = append(wave, index)
		for _, resource := range invocation.Check.Resources {
			use := resources[resource.Name]
			use.count++
			use.exclusive = use.exclusive || resource.Exclusive
			resources[resource.Name] = use
		}
	}
	return wave
}
