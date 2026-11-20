//go:build modeltest && windows

package latentimage

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"math"
	"testing"

	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/testcheck"
)

// fusionSteps are the steps each section of TestTheMOSSPassFusesWithoutChangingABit draws.
const fusionSteps = 2

// fusionLatents are the SHA-256 of the final latent's F32 bits, little-endian, of each section the fusions are held on,
// fusionSteps steps of it asked as the workflow's pass-2 recording asks it: drawn at 15ecde8d3, before moss-fusion fused
// a kernel, as ddc71d81 (text to image) and 08e52361 (depth control), which moss-fusion/copies (e849f2f97) kept bit
// for bit, and drawn again when moss-checkpoint-regression made the attention in registers the native kernel's
// arithmetic, bit for bit, which changed every latent; copies' fusions compute what they fuse exactly, whatever the
// attention.
var fusionLatents = map[string]string{
	"text-to-image": "4eb4a963d352ba09a29c0e701f308a70037cafd9decf18a9cc52751bd7b31c8a",
	"depth-control": "8e74944f150b1a7f2382deceafe543c51dcd3a6417f66df7c4f20f318465f39e",
}

// TestTheMOSSPassFusesWithoutChangingABit holds moss-fusion's fusions to the arithmetic they fuse: a fused pass
// computes each value as the operations it fuses computed it, rounded where they rounded, so text to image (its
// projections in int8) and depth control (its adapters merged into them, its control beside the latent), each drawn
// fusionSteps steps at the workflow's size, end in the final latent overgo drew before any fusion, bit for bit.
func TestTheMOSSPassFusesWithoutChangingABit(t *testing.T) {
	cudatest.Require(t)
	ctx := t.Context()
	root := ultraModelRoot(t)
	ultraDepth(t, root)
	profile := testcheck.Must(ResolveProfile(root))(t)
	var generator *Generator
	defer func() {
		if generator != nil {
			closeReporting(t, context.WithoutCancel(ctx), generator)
		}
	}()
	for _, section := range workflowSections(t) {
		want, held := fusionLatents[section.slug]
		if !held {
			continue
		}
		recording := readComfySection(t, "pass-2", section.slug)
		request := section.request(t, recording)
		if section.slug == "depth-control" {
			// The pass is held, not the estimate: depth control reads the map made before it, as when its hash was drawn,
			// where the workflow asks it from its picture (moss-workflow-clock).
			request, _ = depthControlSection(t, recording)
		}
		request.Steps = fusionSteps
		if generator == nil || generator.Reset(ctx, request) != nil {
			if generator != nil {
				closeReporting(t, context.WithoutCancel(ctx), generator)
			}
			generator = testcheck.Must(LoadGenerator(ctx, root, profile, request))(t)
		}
		session := testcheck.Must(generator.prepare(ctx, request))(t)
		testcheck.Must(generator.integrate(ctx, session))(t)
		final := testcheck.Must(generator.rotary.latent.outputs.CopyToHost(ctx, generator.rotary.latent.node))(t)
		testcheck.Must(generator.decode(ctx, session))(t)
		if got := f32Bits(final.Data); got != want {
			t.Errorf("%s's final latent after %d steps hashes %s; before any fusion it hashed %s", section.slug, fusionSteps, got, want)
		}
	}
}

// f32Bits is the SHA-256 of values' F32 bits, little-endian.
func f32Bits(values []float32) string {
	bits := make([]byte, 4*len(values))
	for index, value := range values {
		binary.LittleEndian.PutUint32(bits[4*index:], math.Float32bits(value))
	}
	sum := sha256.Sum256(bits)
	return hex.EncodeToString(sum[:])
}
