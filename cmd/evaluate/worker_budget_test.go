package main

import (
	"context"
	"errors"
	"testing"
	"time"
)

func TestWorkerBudgetBoundsThePass(t *testing.T) {
	// A pass that outlives the budget is cut and recorded, not failed.
	err := runBudgetedPass(t.Context(), time.Millisecond, "model.gguf", func(ctx context.Context) error {
		<-ctx.Done()
		return ctx.Err()
	})
	if err != nil {
		t.Fatalf("budget exhaustion returned %v, want the recorded outcome", err)
	}
	// A pass that fails inside the budget keeps its own error.
	failure := errors.New("suite refused")
	err = runBudgetedPass(t.Context(), time.Hour, "model.gguf", func(context.Context) error { return failure })
	if !errors.Is(err, failure) {
		t.Fatalf("pass failure returned %v, want %v", err, failure)
	}
	// A pass that completes inside the budget completes.
	if err := runBudgetedPass(t.Context(), time.Hour, "model.gguf", func(context.Context) error { return nil }); err != nil {
		t.Fatal(err)
	}
	if err := runBudgetedPass(t.Context(), 0, "model.gguf", func(context.Context) error { return nil }); err == nil {
		t.Fatal("a zero budget was accepted")
	}
}
