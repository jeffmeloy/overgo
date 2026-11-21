//go:build modeltest && windows

package latentimage

import (
	"context"
	"encoding/binary"
	"math"
	"slices"
	"testing"

	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/testcheck"
)

// ultraConditioningSHA256 is the SHA-256 of the MOSS Ultra workflow's text to image conditioning, its prompt's hidden
// rows and then its negative prompt's as little-endian F32 bits, as overgo encoded it at 93888b740, where the host
// decoded each of the encoder's int8 ConvRot weights (comfyquant.DecodeRow).
const ultraConditioningSHA256 = "4b3428732aca2b4b344a6608ba05ec44994a1a7dead88b9d961a54d1415163a8"

// TestTheUltraEncoderDecodesItsWeightsOnTheCard holds the MOSS Ultra encoder's preparation to its weights' int8 ConvRot
// rows (moss-first-run/do): a session's first preparation of the workflow's text to image request, which places the
// encoder's weights block by block, copies to the card under three fifths of the BF16 bytes those weights decode to,
// its ConvRot weights decoded there from their int8 rows and scales, and its conditioning is the host decode's, bit
// for bit (ultraConditioningSHA256). The host had decoded them on one core and copied each at twice its stored bytes:
// 51 s of the workflow's first run went to that preparation, 41 s of it to the decode (moss-first-run/quiet's split).
func TestTheUltraEncoderDecodesItsWeightsOnTheCard(t *testing.T) {
	cudatest.Require(t)
	ctx := t.Context()
	root := ultraModelRoot(t)
	profile := testcheck.Must(ResolveProfile(root))(t)
	at := slices.IndexFunc(goldenCases, func(golden goldenCase) bool { return golden.name == "text to image" })
	request := goldenCases[at].request(t, readComfySection(t, "pass-1", "text-to-image"))
	generator := testcheck.Must(LoadGenerator(ctx, root, profile, request))(t)
	defer closeReporting(t, context.WithoutCancel(ctx), generator)
	before := testcheck.Must(generator.rotary.runtime.Stats(ctx))(t).Execution.HostToDeviceBytes
	testcheck.Must(generator.prepare(ctx, request))(t)
	copied := testcheck.Must(generator.rotary.runtime.Stats(ctx))(t).Execution.HostToDeviceBytes - before
	var decoded uint64
	for _, binding := range generator.rotary.conditioner.catalog.Bindings() {
		decoded += binding.TargetBytes
	}
	var bits []byte
	for _, row := range [][]float32{generator.rotary.positive, generator.rotary.negative} {
		for _, value := range row {
			bits = binary.LittleEndian.AppendUint32(bits, math.Float32bits(value))
		}
	}
	conditioning := sha256Hex(bits)
	t.Logf("the text to image preparation copied %.2f GiB to the card, its encoder's weights %.2f GiB in BF16; its conditioning hashes %s",
		float64(copied)/(1<<30), float64(decoded)/(1<<30), conditioning)
	if copied*5 >= decoded*3 {
		t.Errorf("the encoder's preparation copied %d bytes, its weights %d in BF16; want under three fifths, its ConvRot weights decoded on the card from their int8 rows", copied, decoded)
	}
	if conditioning != ultraConditioningSHA256 {
		t.Errorf("the conditioning hashes %s; want the host decode's %s, bit for bit", conditioning, ultraConditioningSHA256)
	}
}
