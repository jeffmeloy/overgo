package server

import (
	"testing"

	"overgo/internal/inference"
)

type continuousFactoryGenerator struct {
	*fakeGenerator
	options inference.ContinuousGeneratorOptions
	created *inference.ContinuousGenerator
}

func (f *continuousFactoryGenerator) NewContinuousGenerator(
	options inference.ContinuousGeneratorOptions,
) (*inference.ContinuousGenerator, error) {
	f.options = options
	f.created = &inference.ContinuousGenerator{}
	return f.created, nil
}

func TestServerUsesModelSessionDirectorAsSoleSessionAuthority(t *testing.T) {
	factory := &continuousFactoryGenerator{fakeGenerator: &fakeGenerator{}}
	handler, err := New(Config{MaxConcurrent: 3, ContextShift: true}, factory)
	if err != nil {
		t.Fatal(err)
	}
	defer handler.Close()
	if factory.options.MaxSequences != 3 || !factory.options.ContextShift {
		t.Fatalf("scheduler options = %+v", factory.options)
	}
	lease, ok := handler.sessions.TryLease(-1)
	if !ok || lease.Model() != factory.created {
		t.Fatal("director did not admit the compiled generation session")
	}
	lease.Release()
}
