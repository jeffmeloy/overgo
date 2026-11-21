package speechrecognitiontest

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/testcheck"
)

// locationReader is a capture source that records each capture's locations and nothing else, failing where refusal is
// set.
type locationReader struct {
	artifact.Reader
	locations map[artifact.ID][]artifact.Location
	refusal   error
}

func (r locationReader) Locations(_ context.Context, id artifact.ID) ([]artifact.Location, error) {
	return r.locations[id], r.refusal
}

// TestACaptureWhoseBytesAreGoneIsDeclaredWhileItIsRemade holds speech-hold's declaration to the captures it names: a
// capture the row remaking it names, with no recorded location left, gives that row and no error, so its test is
// declared inapplicable; the same capture with its bytes in place is read where they lie; a capture no row names fails
// when its bytes are gone; and a source that cannot say where a capture lies fails whatever the declaration. Each of
// the three captures the declaration names, its bytes gone, gives speech-captures, the row that remakes it.
func TestACaptureWhoseBytesAreGoneIsDeclaredWhileItIsRemade(t *testing.T) {
	t.Parallel()
	root := t.TempDir()
	present := filepath.Join(root, "tensors.safetensors")
	testcheck.NoError(t, os.WriteFile(present, []byte("captured"), 0o600))
	declared := testcheck.Must(artifact.ParseID("file:sha256:" + strings.Repeat("a", 64)))(t)
	undeclared := testcheck.Must(artifact.ParseID("file:sha256:" + strings.Repeat("b", 64)))(t)
	kept := testcheck.Must(artifact.ParseID("file:sha256:" + strings.Repeat("c", 64)))(t)
	gone := func(id artifact.ID) []artifact.Location {
		return []artifact.Location{{Artifact: id, Kind: artifact.LocationFile, Value: filepath.Join(root, "removed", "tensors.safetensors")}}
	}
	reader := locationReader{locations: map[artifact.ID][]artifact.Location{
		declared: gone(declared), undeclared: gone(undeclared),
		kept: {{Artifact: kept, Kind: artifact.LocationFile, Value: present}},
	}}
	remade := map[string]string{declared.String(): "the remaking row", kept.String(): "the remaking row"}
	if path, row, err := capturePath(t.Context(), reader, declared, remade); path != "" || row != "the remaking row" || err != nil {
		t.Errorf("a declared capture with its bytes gone gives %q, row %q, %v; want the row that remakes it", path, row, err)
	}
	if path, row, err := capturePath(t.Context(), reader, kept, remade); !strings.HasSuffix(path, "tensors.safetensors") || row != "" || err != nil {
		t.Errorf("a declared capture with its bytes in place gives %q, row %q, %v; want its file", path, row, err)
	}
	if _, row, err := capturePath(t.Context(), reader, undeclared, remade); row != "" || err == nil {
		t.Errorf("a capture no row names, its bytes gone, gives row %q and %v; want an error", row, err)
	}
	refusing := reader
	refusing.refusal = errors.New("the store cannot be read")
	if _, row, err := capturePath(t.Context(), refusing, declared, remade); row != "" || err == nil {
		t.Errorf("a source that cannot say where a declared capture lies gives row %q and %v; want its error", row, err)
	}
	for raw := range remadeCaptures {
		id := testcheck.Must(artifact.ParseID(raw))(t)
		removed := locationReader{locations: map[artifact.ID][]artifact.Location{id: gone(id)}}
		if _, row, err := capturePath(t.Context(), removed, id, remadeCaptures); row != "speech-captures" || err != nil {
			t.Errorf("%s, its bytes gone, gives row %q and %v; want speech-captures, which remakes it", raw, row, err)
		}
	}
}

// TestAnAcceptanceWhoseCapturesAreAllBeingRemadeIsDeclaredWhole holds an acceptance's whole declaration (SQA-450): where
// every capture it reads is gone while a row remakes it, the acceptance is declared inapplicable before its cases run,
// so it cannot pass as a parent whose cases each declared themselves; where any capture is in place it runs, its cases
// saying the rest; a capture no row names, its bytes gone, fails it; and an acceptance reading none runs.
func TestAnAcceptanceWhoseCapturesAreAllBeingRemadeIsDeclaredWhole(t *testing.T) {
	t.Parallel()
	root := t.TempDir()
	present := filepath.Join(root, "tensors.safetensors")
	testcheck.NoError(t, os.WriteFile(present, []byte("captured"), 0o600))
	id := func(digit string) artifact.ID {
		return testcheck.Must(artifact.ParseID("file:sha256:" + strings.Repeat(digit, 64)))(t)
	}
	first, second, kept, stray := id("a"), id("b"), id("c"), id("d")
	gone := func(id artifact.ID) []artifact.Location {
		return []artifact.Location{{Artifact: id, Kind: artifact.LocationFile, Value: filepath.Join(root, "removed", id.String()[12:20])}}
	}
	reader := locationReader{locations: map[artifact.ID][]artifact.Location{
		first: gone(first), second: gone(second), stray: gone(stray),
		kept: {{Artifact: kept, Kind: artifact.LocationFile, Value: present}},
	}}
	remade := map[string]string{first.String(): "the remaking row", second.String(): "the remaking row", kept.String(): "the remaking row"}
	if row, err := capturesRemade(t.Context(), reader, []artifact.ID{first, second}, remade); row != "the remaking row" || err != nil {
		t.Errorf("an acceptance whose captures are all being remade gives row %q and %v; want the row, declaring it whole", row, err)
	}
	if row, err := capturesRemade(t.Context(), reader, []artifact.ID{first, kept}, remade); row != "" || err != nil {
		t.Errorf("an acceptance with a capture in place gives row %q and %v; want it to run", row, err)
	}
	if row, err := capturesRemade(t.Context(), reader, []artifact.ID{first, stray}, remade); row != "" || err == nil {
		t.Errorf("an acceptance reading a capture no row names, its bytes gone, gives row %q and %v; want an error", row, err)
	}
	if row, err := capturesRemade(t.Context(), reader, nil, remade); row != "" || err != nil {
		t.Errorf("an acceptance reading no capture gives row %q and %v; want it to run", row, err)
	}
}
