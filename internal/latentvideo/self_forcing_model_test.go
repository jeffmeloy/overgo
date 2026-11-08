//go:build modeltest

package latentvideo

import (
	"path/filepath"
	"testing"

	"overgo/internal/modeltest"
	"overgo/internal/testcheck"
)

// TestSelfForcingIsAWan13BDiT holds the Self-Forcing student to the model it was distilled from: decision
// self-forcing-download fetched self_forcing_dmd.pt, 5.68 GB, the size of the local F32 Wan DiT, its tensors unread.
// Each of its tensors is one of Wan2.1-T2V-1.3B's DiT, of the same shape, and LiveEdit's checkpoint, whose patch
// embedding also reads a source clip's channels, is refused.
func TestSelfForcingIsAWan13BDiT(t *testing.T) {
	t.Parallel()
	wan := modeltest.Directory(t, "Wan2.1-T2V-1.3B")
	student := modeltest.Directory(t, "Self-Forcing")
	testcheck.NoError(t, matchesWanDiT(filepath.Join(student, "checkpoints", "self_forcing_dmd.pt"), wan))
	editor := modeltest.Directory(t, "LiveEdit")
	if err := matchesWanDiT(filepath.Join(editor, liveEditCheckpoint), wan); err == nil {
		t.Fatal("LiveEdit's checkpoint, which reads a source clip beside the noise, matched the text-to-video DiT")
	}
}
