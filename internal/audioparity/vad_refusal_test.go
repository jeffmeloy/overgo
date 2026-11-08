package audioparity

import (
	"archive/zip"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/speechactivity"
	"overgo/internal/testcheck"
)

// verifyVADArtifactRefusals runs once, as TestVADOfflineAcceptance's
// artifact-refusals subtest.
func verifyVADArtifactRefusals(t *testing.T) {
	t.Parallel()
	reference, store, _ := loadVADReference(t)
	l := newVADLifecycle(t, store, reference.Models[0], false)
	// No model is ever distributed, so the detector loads whatever its licence: one whose licence file is missing is
	// admitted (the owner's rule, 2026-11-08).
	t.Run("missing-license", func(t *testing.T) {
		testcheck.Must(speechactivity.LoadDetector(t.Context(), &vadFaultRepository{Repository: l.store, hidden: l.profile.License}, l.recipe, vadReferenceBytes))(t)
	})
	for _, name := range []string{"tensor-identity", "numeric-budget"} {
		t.Run(name, func(t *testing.T) {
			fault := &vadFaultRepository{Repository: l.store}
			budget := uint64(vadReferenceBytes)
			switch name {
			case "tensor-identity":
				fault.manifestDrift = true
			case "numeric-budget":
				budget = 1
			}
			if _, err := speechactivity.LoadDetector(t.Context(), fault, l.recipe, budget); err == nil {
				t.Fatal("invalid model admission succeeded")
			}
		})
	}
	for _, name := range []string{"unsafe-constructor", "metadata-bound"} {
		t.Run(name, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "refused.pth")
			file, err := os.Create(path)
			testcheck.NoError(t, err)
			archive := zip.NewWriter(file)
			if name == "metadata-bound" {
				// A ZIP64 metadata entry advertises a host-unrepresentable length;
				// refusal must precede decompression or allocation of that extent.
				_, err = archive.CreateRaw(&zip.FileHeader{Name: "archive/data.pkl", Method: zip.Store, UncompressedSize64: math.MaxUint64})
			} else {
				var entry interface{ Write([]byte) (int, error) }
				entry, err = archive.CreateHeader(&zip.FileHeader{Name: "archive/data.pkl", Method: zip.Store})
				if err == nil {
					// PROTO 2, unknown GLOBAL, EMPTY_TUPLE, NEWOBJ, STOP.
					_, err = entry.Write([]byte("\x80\x02cexample\nConstructor\n)\x81."))
				}
			}
			testcheck.NoError(t, err)
			testcheck.NoError(t, archive.Close())
			testcheck.NoError(t, file.Close())
			_, err = speechactivity.LoadNetwork(t.Context(), path, l.profile.Network, vadReferenceBytes)
			if err == nil {
				t.Fatal("unsafe or oversized metadata admitted")
			}
			if name == "unsafe-constructor" && !strings.Contains(err.Error(), "NEWOBJ") {
				t.Fatalf("refused for unrelated reason: %v", err)
			}
			t.Log(err)
		})
	}
}
