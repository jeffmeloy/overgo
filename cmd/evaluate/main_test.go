package main

import (
	"context"
	"errors"
	"reflect"
	"testing"

	"overgo/internal/testcheck"
)

type fixtureSession struct {
	tasks  *[]string
	closed *int
	fail   string
}

func (s fixtureSession) Evaluate(_ context.Context, task string) error {
	*s.tasks = append(*s.tasks, task)
	if task == s.fail {
		return errors.New("task failed")
	}
	return nil
}

func (s fixtureSession) Close() error {
	*s.closed++
	return nil
}

func TestEvaluateCommandIsolatesModelsAndReportsTaskFailure(t *testing.T) {
	compiled, err := compileManifest(manifest{
		Repository: "repo", Catalog: "catalog.json", CodeCommit: "commit", Device: 0,
		Models: []modelRequest{
			{Path: "first.gguf", Suites: []string{"a.json"}},
			{Path: "second.gguf", Suites: []string{"b.json"}},
			{Path: "first.gguf", Suites: []string{"fail.json", "unreached.json"}},
		},
	})
	testcheck.NoError(t, err)
	var launched []int
	err = runParent(t.Context(), compiled, func(_ context.Context, index int) error {
		launched = append(launched, index)
		if index == 1 {
			return errors.New("worker failed")
		}
		return nil
	})
	if err == nil || !reflect.DeepEqual(launched, []int{0, 1}) || len(compiled.Models) != 2 {
		t.Fatalf("parent launch = %v models=%d err=%v", launched, len(compiled.Models), err)
	}

	var tasks []string
	openCount, closeCount := 0, 0
	err = executeModel(t.Context(), compiled, compiled.Models[0], func(
		context.Context, manifest, modelRequest,
	) (evaluationSession, error) {
		openCount++
		return fixtureSession{tasks: &tasks, closed: &closeCount, fail: "fail.json"}, nil
	})
	if err == nil || openCount != 1 || closeCount != 1 ||
		!reflect.DeepEqual(tasks, []string{"a.json", "fail.json"}) {
		t.Fatalf("worker opens/closes/tasks = %d/%d/%v err=%v", openCount, closeCount, tasks, err)
	}
}

// derivedFixtureSession is a session that campaigns derived suites, recording each family it is given.
type derivedFixtureSession struct {
	fixtureSession
	families *[]string
}

func (s derivedFixtureSession) EvaluateDerived(_ context.Context, family string) error {
	*s.families = append(*s.families, family)
	return nil
}

// TestAModelRequestCampaignsItsDerivedFamily holds a manifest's derived family to its model's session: a request
// naming only a family compiles and campaigns it after any suites, through the session's derived evaluation; a
// session that campaigns no derived suites is refused for it, closed all the same; a request naming neither suites
// nor a family, or a family on a model's later request, is refused.
func TestAModelRequestCampaignsItsDerivedFamily(t *testing.T) {
	t.Parallel()
	compiled := testcheck.Must(compileManifest(manifest{
		Repository: "repo", Catalog: "catalog.json", CodeCommit: "commit",
		Models: []modelRequest{{Path: "model.gguf", Suites: []string{"a.json"}, Family: "bbh"}},
	}))(t)
	var tasks, families []string
	closed := 0
	testcheck.NoError(t, executeModel(t.Context(), compiled, compiled.Models[0], func(context.Context, manifest, modelRequest) (evaluationSession, error) {
		return derivedFixtureSession{fixtureSession: fixtureSession{tasks: &tasks, closed: &closed}, families: &families}, nil
	}))
	if !reflect.DeepEqual(tasks, []string{"a.json"}) || !reflect.DeepEqual(families, []string{"bbh"}) || closed != 1 {
		t.Fatalf("the worker evaluated %v and the families %v, closing %d times; want the suite, then the family, closed once", tasks, families, closed)
	}
	familyOnly := testcheck.Must(compileManifest(manifest{
		Repository: "repo", Catalog: "catalog.json", CodeCommit: "commit", Models: []modelRequest{{Path: "model.gguf", Family: "bbh"}},
	}))(t)
	if err := executeModel(t.Context(), familyOnly, familyOnly.Models[0], func(context.Context, manifest, modelRequest) (evaluationSession, error) {
		return fixtureSession{tasks: &tasks, closed: &closed}, nil
	}); err == nil || closed != 2 {
		t.Fatalf("a session campaigning no derived suites was given a family: %v, closed %d times", err, closed)
	}
	for _, models := range [][]modelRequest{
		{{Path: "model.gguf"}},
		{{Path: "model.gguf", Suites: []string{"a.json"}}, {Path: "model.gguf", Family: "bbh"}},
	} {
		if _, err := compileManifest(manifest{Repository: "repo", Catalog: "catalog.json", CodeCommit: "commit", Models: models}); err == nil {
			t.Errorf("the manifest %+v compiled", models)
		}
	}
}
