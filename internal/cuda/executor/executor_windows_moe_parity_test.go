//go:build windows

package executor

import (
	"testing"

	"overgo/internal/tensor/dtype"
)

// TestFlashNextExpertsAnswerAlikeOnTheCardAndTheHost holds a routed layer of each of Flash-Next's formats to one
// answer wherever its experts run (flashnext-projection-exact/host-accuracy): the layer's output with expert 0 on the card and
// the rest computed on the host stays as near the f32 reference as the output with every expert on the card, within
// twice the card's largest difference, so which experts the card holds, which its free memory sets, moves an answer no
// more than the card's own rounding does. Flash-Next's exact projection suite answered " D" on the full quiet card and
// "D" under a 47 GiB budget at 1251fe747; the host quantizes an expert's activations as Q8_K or Q8_0, as llama.cpp's CPU
// backend does, and measured at 1251fe747 it stood 0.056 to 0.074 from the reference where the card stood 0.0009 to
// 0.0014, sixty times nearer. With each quantized row split in two, the values' and what their rounding left
// (hostexpert), the host stands 0.0003 to 0.0007 from it; either split alone left it at 0.024 to 0.068.
func TestFlashNextExpertsAnswerAlikeOnTheCardAndTheHost(t *testing.T) {
	// Serial: device fixtures share the one card.
	// cardRounding is how many times the card's largest difference from the reference the host's may reach.
	const cardRounding = 2
	for _, layout := range flashNextExpertLayouts {
		t.Run(layout.name, func(t *testing.T) {
			c, limit := layout.hostCase(t)
			c.reference = true
			card, want := c.run(t, newFixtureExecutor(t), dtype.F32)
			c.reference, c.offload, c.limit = false, true, limit
			host, _ := c.run(t, newFixtureExecutor(t), dtype.F32)
			apart, at := maxAbsDifference(t, card, host)
			cardOff, _ := maxAbsDifference(t, card, want)
			hostOff, _ := maxAbsDifference(t, host, want)
			t.Logf("%s: the card's output against the host's differs by %g at most (%g against %g at %d); from the f32 reference the card's by %g, the host's by %g",
				layout.name, apart, card[at], host[at], at, cardOff, hostOff)
			if hostOff > cardRounding*cardOff {
				t.Errorf("%s on the host stands %g from the f32 reference where on the card it stands %g; want the host within %d times the card's", layout.name, hostOff, cardOff, cardRounding)
			}
		})
	}
}
