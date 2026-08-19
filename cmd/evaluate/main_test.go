package main

import (
	"context"
	"errors"
	"reflect"
	"testing"
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
	if err != nil {
		t.Fatal(err)
	}
	var launched []int
	err = runParent(context.Background(), compiled, func(_ context.Context, index int) error {
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
	err = executeModel(context.Background(), compiled, compiled.Models[0], func(
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
