//go:build windows

package latentimage

import (
	"bytes"
	"context"
	"errors"
	"testing"

	"overgo/internal/artifact"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/graphruntime"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

// finishedRequestBytes bounds what a finished request leaves on the device, the
// executor's own small handles (512 bytes when written).
const finishedRequestBytes = 1 << 16

// TestMOSSBoundedResidency holds a session to its bounded stages: a request
// leaves no device memory behind, a prompt met again places no text encoder on
// the device, two new prompts place it once, and none of this changes a
// picture from a fresh session's.
func TestMOSSBoundedResidency(t *testing.T) {
	cudatest.Require(t)
	root, _ := compactGeneratorFixture(t)
	profile, err := ResolveProfile(root)
	testcheck.NoError(t, err)
	base := Request{Prompt: "blue cube", NegativePrompt: "red", Width: 48, Height: 32, Steps: 3, Seed: 42}
	load := func(request Request) *Generator {
		t.Helper()
		g, err := LoadGenerator(t.Context(), root, profile, request)
		testcheck.NoError(t, err)
		t.Cleanup(testutil.Closer(t, context.Background(), g))
		return g
	}
	stats := func(g *Generator) graphruntime.ResidentStats {
		t.Helper()
		s, err := g.rotary.runtime.Stats(t.Context())
		testcheck.NoError(t, err)
		return s
	}
	finish := func(g *Generator) []byte {
		t.Helper()
		session, err := g.integrate(t.Context(), g)
		testcheck.NoError(t, err)
		image, err := g.decode(t.Context(), session)
		testcheck.NoError(t, err)
		if s := stats(g); s.StaticBytes != 0 || s.Device.CurrentBytes > finishedRequestBytes {
			t.Fatalf("a finished request left %d bytes of weights and %d bytes on the device", s.StaticBytes, s.Device.CurrentBytes)
		}
		if peak, err := g.PeakDeviceBytes(); err != nil || peak == 0 {
			t.Fatalf("a finished request read a device peak of %d: %v", peak, err)
		}
		if g.rotary.conditioned.held > g.rotary.conditioned.limit {
			t.Fatal("conditioning cache exceeded its limit", g.rotary.conditioned.held)
		}
		return image.Data
	}
	prepare := func(g *Generator, request Request) uint64 {
		t.Helper()
		before := stats(g)
		if _, err := g.prepare(t.Context(), request); err != nil {
			t.Fatal(err)
		}
		return stats(g).Execution.HostToDeviceBytes - before.Execution.HostToDeviceBytes
	}
	g := load(base)
	bothNew := prepare(g, base)
	native := finish(g)
	id, err := artifact.IdentifyBytes(artifact.KindOutput, native)
	if err != nil || id.DigestHex() != "99062f8a13f008e4c3f14f45e5a2b063e2ce4e910a59f90b3b2bdc48d32fdeda" {
		t.Fatal("native compact image differs", id, err)
	}
	unit := float32(1)
	cases := []struct {
		name    string
		request Request
		cached  bool
		before  func()
	}{
		{name: "same request", request: base, cached: true},
		{name: "new seed", request: Request{Prompt: "blue cube", NegativePrompt: "red", Width: 48, Height: 32, Steps: 3, Seed: 7}, cached: true},
		{name: "new positive", request: Request{Prompt: "green tree", NegativePrompt: "red", Width: 48, Height: 32, Steps: 3, Seed: 7}},
		{name: "new negative", request: Request{Prompt: "green tree", NegativePrompt: "blue sky", Width: 48, Height: 32, Steps: 3, Seed: 7}},
		{name: "guidance one", request: Request{Prompt: "green tree", NegativePrompt: "never encoded", GuidanceScale: &unit, Width: 48, Height: 32, Steps: 3, Seed: 7}, cached: true},
		{name: "evicted", request: base, before: func() { g.rotary.conditioned = conditioningCache{limit: g.rotary.conditioned.limit} }},
	}
	for _, test := range cases {
		t.Run(test.name, func(t *testing.T) {
			if test.before != nil {
				test.before()
			}
			testcheck.NoError(t, g.Reset(t.Context(), test.request))
			transferred := prepare(g, test.request)
			if test.cached && transferred != 0 {
				t.Fatalf("a prompt met again uploaded %d bytes", transferred)
			}
			if !test.cached && transferred == 0 {
				t.Fatal("a new prompt did not encode")
			}
			actual := finish(g)
			cold := load(test.request)
			if _, err := cold.prepare(t.Context(), test.request); err != nil {
				t.Fatal(err)
			}
			if !bytes.Equal(actual, finish(cold)) {
				t.Fatal("session reuse changed the fresh image")
			}
			testcheck.NoError(t, cold.Close(t.Context()))
		})
	}
	t.Run("two prompts place the encoder once", func(t *testing.T) {
		testcheck.NoError(t, g.Reset(t.Context(), Request{Prompt: "yellow hat", NegativePrompt: "red", Width: 48, Height: 32, Steps: 3, Seed: 7}))
		onePrompt := prepare(g, Request{Prompt: "yellow hat", NegativePrompt: "red", Width: 48, Height: 32, Steps: 3, Seed: 7})
		finish(g)
		// The second prompt of a two-prompt request adds its token rows, never
		// the weights again.
		if bothNew >= onePrompt+onePrompt/2 {
			t.Fatalf("two prompts uploaded %d bytes, one uploads %d", bothNew, onePrompt)
		}
	})
	cancelled, cancel := context.WithCancel(t.Context())
	cancel()
	before := stats(g)
	if err := g.Reset(cancelled, base); !errors.Is(err, context.Canceled) {
		t.Fatal("cancelled reset accepted", err)
	}
	if after := stats(g); after.Execution != before.Execution {
		t.Fatal("cancelled reset performed device work")
	}
	testcheck.NoError(t, g.Reset(t.Context(), base))
	if _, err := g.prepare(cancelled, base); !errors.Is(err, context.Canceled) {
		t.Fatal("cancelled preparation accepted", err)
	}
	if g.prepared {
		t.Fatal("cancelled preparation published completion")
	}
	testcheck.NoError(t, g.Close(cancelled))
	if g.rotary != nil {
		t.Fatal("closed session retained conditioning")
	}
}
