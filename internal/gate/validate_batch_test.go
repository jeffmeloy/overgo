package gate

import (
	"context"
	"errors"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"overgo/internal/automationcheck"
)

// TestValidateChecksRunFirstConcurrently pins the validate wave: protection
// and scope admit the candidate first, every static check with no data
// dependency on another then runs in one wave, the store writer never
// overlaps a store reader, vet and build follow the whole wave, and two
// failures inside the wave are both reported.
func TestValidateChecksRunFirstConcurrently(t *testing.T) {
	g := &gateContext{repo: t.TempDir(), paths: []string{"internal/gate/gate.go"}}
	checks := g.pipelineChecks()
	byName := map[string]automationcheck.Descriptor{}
	for _, check := range checks {
		byName[check.Descriptor.Name] = check.Descriptor
	}
	if !slices.Equal(byName["scope"].Dependencies, []string{"protection"}) {
		t.Fatalf("scope depends on %v", byName["scope"].Dependencies)
	}
	for _, name := range validateWave {
		if !slices.Equal(byName[name].Dependencies, []string{"scope"}) {
			t.Fatalf("%s depends on %v, want the admission checks only", name, byName[name].Dependencies)
		}
	}
	for _, static := range []string{"vet", "build"} {
		if !slices.Equal(byName[static].Dependencies, validateWave) {
			t.Fatalf("%s depends on %v, want the whole validate wave", static, byName[static].Dependencies)
		}
	}
	type span struct{ start, end time.Time }
	run := func(failing ...string) (map[string]span, []automationcheck.DAGResult) {
		var mutex sync.Mutex
		spans := map[string]span{}
		for index := range checks {
			checks[index].Run = func(_ context.Context, invocation automationcheck.Invocation) (bool, string, error) {
				started := time.Now()
				time.Sleep(60 * time.Millisecond)
				mutex.Lock()
				spans[invocation.Check.Name] = span{started, time.Now()}
				mutex.Unlock()
				if slices.Contains(failing, invocation.Check.Name) {
					return false, "", errors.New("injected " + invocation.Check.Name + " failure")
				}
				return false, "", nil
			}
		}
		invocations, err := automationcheck.Plan(checks, automationcheck.Impact{})
		if err != nil {
			t.Fatal(err)
		}
		results, err := automationcheck.ExecuteDAG(t.Context(), invocations, nil, automationcheck.Run)
		if err != nil {
			t.Fatal(err)
		}
		return spans, results
	}
	overlaps := func(spans map[string]span, left, right string) bool {
		return spans[left].start.Before(spans[right].end) && spans[right].start.Before(spans[left].end)
	}
	spans, _ := run()
	if spans["scope"].start.Before(spans["protection"].end) {
		t.Fatal("scope started before protection ended")
	}
	for _, name := range validateWave {
		if spans[name].start.Before(spans["scope"].end) {
			t.Fatalf("%s started before scope ended", name)
		}
		for _, static := range []string{"vet", "build"} {
			if spans[static].start.Before(spans[name].end) {
				t.Fatalf("%s started before %s ended", static, name)
			}
		}
		if name == "modern-go" || name == "docs" || name == "published" {
			continue
		}
		if !overlaps(spans, name, "modern-go") {
			t.Fatalf("%s did not run in the modern-Go wave", name)
		}
	}
	for _, reader := range []string{"docs", "published"} {
		if overlaps(spans, "magics", reader) {
			t.Fatalf("the store writer magics overlapped the store reader %s", reader)
		}
	}
	if spans["acceptance"].start.Before(spans["vet"].end) || spans["acceptance"].start.Before(spans["build"].end) {
		t.Fatal("acceptance started before vet and build ended")
	}
	spans, results := run("fmt", "style")
	err := checkFailures(results)
	if err == nil || !strings.Contains(err.Error(), "fmt: injected fmt failure") || !strings.Contains(err.Error(), "style: injected style failure") {
		t.Fatalf("wave failures = %v, want both findings", err)
	}
	if _, ran := spans["vet"]; ran {
		t.Fatal("vet ran after the validate wave failed")
	}
}
