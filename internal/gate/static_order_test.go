package gate

import (
	"cmp"
	"context"
	"errors"
	"slices"
	"sync"
	"testing"

	"overgo/internal/automationcheck"
	"overgo/internal/runrecord"
)

// TestStaticFailurePreventsAcceptance exercises the executable dependency graph,
// including subordinate batches: failed build/vet must never start model work.
func TestStaticFailurePreventsAcceptance(t *testing.T) {
	t.Parallel()
	for _, failed := range []string{modernCensusCheckName, "modern-go", "vet", "build", ""} {
		for _, batched := range []bool{false, true} {
			name := cmp.Or(failed, "success")
			if batched {
				name += "/batch"
			}
			t.Run(name, func(t *testing.T) {
				g, batch, _ := verificationBatchFixture(t, "pass")
				checks := g.pipelineChecks()
				if batched {
					var err error
					checks, err = g.batchAcceptanceChecks(checks, batch)
					if err != nil {
						t.Fatal(err)
					}
				}
				var mutex sync.Mutex
				var executed []string
				for i := range checks {
					checks[i].Run = func(_ context.Context, invocation automationcheck.Invocation) (bool, string, error) {
						mutex.Lock()
						defer mutex.Unlock()
						executed = append(executed, invocation.Check.Name)
						if invocation.Check.Name == failed {
							return false, "", errors.New("injected static failure")
						}
						return false, "", nil
					}
				}
				invocations, err := automationcheck.Plan(checks, automationcheck.Impact{})
				if err != nil {
					t.Fatal(err)
				}
				_, err = automationcheck.ExecuteDAG(t.Context(), invocations, nil, automationcheck.Run)
				if err != nil {
					t.Fatal(err)
				}
				if failed != "" && !slices.Contains(executed, failed) {
					t.Fatalf("injected static failure did not execute: %v", executed)
				}
				for _, check := range checks {
					name := check.Descriptor.Name
					if check.Descriptor.Phase != runrecord.PhaseTest && check.Descriptor.Phase != runrecord.PhasePackage {
						continue
					}
					position := slices.Index(executed, name)
					if failed != "" && position >= 0 {
						t.Fatalf("%s ran despite %s failure: %v", name, failed, executed)
					}
					vet, build := slices.Index(executed, "vet"), slices.Index(executed, "build")
					if failed == "" && (position < 0 || vet < 0 || build < 0 || vet > position || build > position) {
						t.Fatalf("%s did not run after static checks: %v", name, executed)
					}
				}
			})
		}
	}
}
