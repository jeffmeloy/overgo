package automationcheck

import (
	"context"
	"slices"
	"sync"
	"sync/atomic"
	"testing"

	"overgo/internal/runrecord"
)

func TestResourceDAG(t *testing.T) {
	checks := []Check{
		dagFixtureCheck("cpu-a", nil), dagFixtureCheck("cpu-b", nil),
		dagFixtureCheck("device-a", []Resource{{Name: "device", Exclusive: true}}),
		dagFixtureCheck("device-b", []Resource{{Name: "device", Exclusive: true}}),
	}
	planned, err := Plan(checks, Impact{})
	if err != nil {
		t.Fatal(err)
	}
	started, release := make(chan struct{}, 2), make(chan struct{})
	var deviceActive, deviceMaximum atomic.Int32
	done := make(chan error, 1)
	go func() {
		_, err := ExecuteDAG(context.Background(), planned, nil, func(ctx context.Context, invocation Invocation) (Evidence, error) {
			if invocation.Check.Name == "cpu-a" || invocation.Check.Name == "cpu-b" {
				started <- struct{}{}
				<-release
			}
			if invocation.Check.Resources != nil {
				active := deviceActive.Add(1)
				for maximum := deviceMaximum.Load(); active > maximum && !deviceMaximum.CompareAndSwap(maximum, active); maximum = deviceMaximum.Load() {
				}
				deviceActive.Add(-1)
			}
			return Run(ctx, invocation)
		})
		done <- err
	}()
	<-started
	<-started
	close(release)
	if err := <-done; err != nil {
		t.Fatal(err)
	}
	if deviceMaximum.Load() != 1 {
		t.Fatalf("exclusive device concurrency = %d", deviceMaximum.Load())
	}
}

func TestPolicyBarrier(t *testing.T) {
	checks := []Check{
		dagFixtureCheck("policy", nil),
		dagFixtureCheckAfter("work", "policy"),
		dagFixtureCheckAfter("commit", "work"),
	}
	planned, err := Plan(checks, Impact{})
	if err != nil {
		t.Fatal(err)
	}
	var mutex sync.Mutex
	var order []string
	_, err = ExecuteDAG(context.Background(), planned, nil, func(ctx context.Context, invocation Invocation) (Evidence, error) {
		mutex.Lock()
		order = append(order, invocation.Check.Name)
		mutex.Unlock()
		return Run(ctx, invocation)
	})
	if err != nil || !slices.Equal(order, []string{"policy", "work", "commit"}) {
		t.Fatalf("barrier order = %v, %v", order, err)
	}
}

func TestExcludedDependencySatisfiesPolicyBarrier(t *testing.T) {
	optional := dagFixtureCheck("optional", nil)
	optional.Descriptor.Always = false
	optional.Descriptor.Triggers = []Fact{"optional"}
	optional.Descriptor.Inapplicable = "fixture exclusion"
	checks := []Check{optional, dagFixtureCheckAfter("commit", "optional")}
	impact := Impact{Exclusions: []Exclusion{{Check: "optional", Reason: "fixture proves independence"}}}
	planned, err := Plan(checks, impact)
	if err != nil {
		t.Fatal(err)
	}
	results, err := ExecuteDAG(context.Background(), planned, map[string]bool{"optional": true}, func(ctx context.Context, invocation Invocation) (Evidence, error) {
		return Run(ctx, invocation)
	})
	if err != nil || len(results) != 1 || results[0].Invocation.Check.Name != "commit" {
		t.Fatalf("results=%+v err=%v", results, err)
	}
}

func dagFixtureCheck(name string, resources []Resource) Check {
	return Check{
		Descriptor: Descriptor{Name: name, Phase: runrecord.PhaseTest, Always: true, Resources: resources},
		Run:        func(context.Context, Invocation) (bool, string, error) { return false, "", nil },
	}
}

func dagFixtureCheckAfter(name, dependency string) Check {
	check := dagFixtureCheck(name, nil)
	check.Descriptor.Dependencies = []string{dependency}
	return check
}
