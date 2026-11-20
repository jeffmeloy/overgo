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
// fusionSteps steps of it asked as the workflow's pass-2 recording asks it, as overgo drew them at 15ecde8d3, before
// moss-fusion fused a kernel.
var fusionLatents = map[string]string{
	"text-to-image": "ddc71d81902dabf7c224b64a0eeac559f90c956d39e642edfb0ca5d991dc161a",
	"depth-control": "08e5236167844f130e7a274258f8a4f8254120658bf370799ab6fa9c064c3b79",
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
		request := section.request(t, readComfySection(t, "pass-2", section.slug))
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
