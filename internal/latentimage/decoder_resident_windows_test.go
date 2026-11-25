//go:build windows

package latentimage

import (
	"bytes"
	"context"
	"testing"

	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/processmeasure"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

// TestTheDecoderIsCompiledOnceASession holds decoder-resident/do: a MOSS session compiles its decoder at its first
// decode and keeps it, so the second request it draws compiles none, and that request's picture is the one a fresh
// session draws for it, bit for bit; another session compiles its own. The decoder had been compiled with its static
// inputs bound and released at every decode, 77 to 79 ms of each MOSS Ultra section (png-level/clock), where Krea's
// resident pipeline keeps its VAE.
func TestTheDecoderIsCompiledOnceASession(t *testing.T) {
	cudatest.Require(t)
	root, _ := compactGeneratorFixture(t)
	profile := testcheck.Must(ResolveProfile(root))(t)
	load := func(request Request) *Generator {
		t.Helper()
		g := testcheck.Must(LoadGenerator(t.Context(), root, profile, request))(t)
		t.Cleanup(testutil.Closer(t, context.Background(), g))
		return g
	}
	// draw draws request in g and gives its picture and how many times the decoder was compiled for it.
	draw := func(g *Generator, request Request) ([]byte, int) {
		t.Helper()
		staged, stages := processmeasure.WithStages(t.Context())
		session := testcheck.Must(g.prepare(staged, request))(t)
		testcheck.Must(g.integrate(staged, session))(t)
		picture := testcheck.Must(g.decode(staged, session))(t)
		compiled := 0
		for _, part := range testcheck.Must(stages.Parts())(t) {
			if part.Name == "rotary image decoder compiled" {
				compiled += part.Times
			}
		}
		return picture.Data, compiled
	}
	first := Request{Prompt: "blue cube", NegativePrompt: "red", Width: 48, Height: 32, Steps: 3, Seed: 42}
	second := first
	second.Seed = 7
	g := load(first)
	if _, compiled := draw(g, first); compiled != 1 {
		t.Fatalf("a session's first decode compiled the decoder %d times; want once", compiled)
	}
	testcheck.NoError(t, g.Reset(t.Context(), second))
	again, compiled := draw(g, second)
	if compiled != 0 {
		t.Errorf("a session's second decode compiled the decoder %d times; want none, the first's kept", compiled)
	}
	fresh, freshCompiled := draw(load(second), second)
	if freshCompiled != 1 {
		t.Errorf("a fresh session compiled the decoder %d times; want its own once", freshCompiled)
	}
	if !bytes.Equal(again, fresh) {
		t.Error("the kept decoder's picture differs from the one a fresh session draws for the same request")
	}
}
