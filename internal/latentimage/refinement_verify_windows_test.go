//go:build windows

package latentimage

import (
	"io"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestMOSSRefinementVerificationRejectsChangedContract(t *testing.T) {
	for _, name := range []string{"checkpoint.json", "oracle-contract.json"} {
		t.Run(name, func(t *testing.T) {
			root := t.TempDir()
			for _, file := range []string{"checkpoint.json", "oracle-contract.json"} {
				data, err := os.ReadFile(filepath.Join("..", "..", "fixtures", "moss", file))
				if err != nil {
					t.Fatal(err)
				}
				if file == name {
					data = append(data, ' ')
				}
				if err := os.WriteFile(filepath.Join(root, file), data, 0600); err != nil {
					t.Fatal(err)
				}
			}
			err := VerifyMOSSTransformer(t.Context(), "absent-checkpoint", "absent-reference", root, "refinement", io.Discard)
			if err == nil || !strings.Contains(err.Error(), "identity differs") || !strings.Contains(err.Error(), name) {
				t.Fatalf("changed %s reached model/device execution: %v", name, err)
			}
		})
	}
}

// TestMOSSVerifierFailsExactBoundaries holds the generation verifier to the bits
// of an exact boundary: either signed zero against the other fails though the
// distance is zero and both passes repeat it, equal bits pass, and a boundary
// with a frozen tolerance keeps it.
func TestMOSSVerifierFailsExactBoundaries(t *testing.T) {
	t.Parallel()
	negative := float32(math.Copysign(0, -1))
	tolerance := mossStageTolerance{Absolute: 1e-3}
	for _, test := range []struct {
		boundary     string
		actual, want float32
		accepted     bool
	}{
		{"initial_latents", negative, 0, false},
		{"schedule.sigmas", 0, negative, false},
		{"conditioning.0.hidden", negative, 0, false},
		{"initial_latents", 1, 1, true},
		{"latents.3", negative, 0, true},
		{"latents.3", 1, 1.0005, true},
		{"latents.3", 1, 1.01, false},
	} {
		applied := tolerance
		if mossExactBoundary(test.boundary) {
			applied = mossStageTolerance{}
		}
		failed, unequal, _ := compareMOSSValues([]float32{test.actual}, []float32{test.want}, applied)
		if got := mossBoundaryAccepted(test.boundary, failed, unequal, true); got != test.accepted {
			t.Fatalf("%s %v against %v accepted %v, want %v", test.boundary, test.actual, test.want, got, test.accepted)
		}
	}
	if mossBoundaryAccepted("latents.3", 0, 0, false) {
		t.Fatal("a boundary that did not repeat was accepted")
	}
}
