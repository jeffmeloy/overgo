package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/testcheck"
	"overgo/internal/trainingworkflow"
)

// TestARouteIsIdentifiedByTheBytesItReads holds the MOSS and codec routes to a dataset identity of the bytes they read,
// parsed as they parse their manifests, read once (SQA-507). Two MOSS corpora whose manifest names " red.png", one
// leading space, beside one red.png and other pictures named " red.png" are two datasets, each holding the bytes of
// " red.png", the picture it trains on: the identity had trimmed the line and hashed red.png for both. A picture changed
// after its manifest was read is trained and judged as read, and a clip changed after admission is refused, not decoded
// under the admitted identity. Two corpora with one manifest text and other media are two datasets, and a corpus copied
// to another directory is one.
func TestARouteIsIdentifiedByTheBytesItReads(t *testing.T) {
	t.Parallel()
	corpus := func(files map[string]string, manifest string) string {
		directory := t.TempDir()
		for name, content := range files {
			testcheck.NoError(t, os.WriteFile(filepath.Join(directory, name), []byte(content), 0o600))
		}
		path := filepath.Join(directory, "manifest.tsv")
		testcheck.NoError(t, os.WriteFile(path, []byte(manifest), 0o600))
		return path
	}
	dataset := func(identity []byte) string {
		id, _, _, err := trainingworkflow.TextAuthority(identity)
		testcheck.NoError(t, err)
		return id.String()
	}
	pictures := func(manifest string) ([]trainingworkflow.ManifestPicture, string) {
		read, identity, err := trainingworkflow.PictureManifest(manifest)
		testcheck.NoError(t, err)
		return read, dataset(identity)
	}

	spaced := " red.png\ta red square\n"
	first, firstID := pictures(corpus(map[string]string{"red.png": "red pixels", " red.png": "blue pixels"}, spaced))
	_, secondID := pictures(corpus(map[string]string{"red.png": "red pixels", " red.png": "green pixels"}, spaced))
	if firstID == secondID {
		t.Error("two corpora whose manifest names \" red.png\" beside one red.png and other \" red.png\" pictures are one dataset; want two")
	}
	if len(first) != 1 || string(first[0].Data) != "blue pixels" || filepath.Base(first[0].Path) != " red.png" {
		t.Errorf("the manifest naming \" red.png\" reads %+v; want \" red.png\" and its bytes", first)
	}

	two := "red.png\ta red square\ngreen.png\ta green square\n"
	media := map[string]string{"red.png": "red pixels", "green.png": "green pixels"}
	original := corpus(media, two)
	read, originalID := pictures(original)
	_, copiedID := pictures(corpus(media, two))
	_, swappedID := pictures(corpus(map[string]string{"red.png": "green pixels", "green.png": "green pixels"}, two))
	if originalID != copiedID {
		t.Error("one corpus in two directories is two datasets; want one")
	}
	if originalID == swappedID {
		t.Error("two corpora with one manifest text and red.png's bytes changed are one dataset; want two")
	}
	testcheck.NoError(t, os.WriteFile(filepath.Join(filepath.Dir(original), "red.png"), []byte("blue pixels"), 0o600))
	if string(read[0].Data) != "red pixels" {
		t.Errorf("red.png changed after its manifest was read is trained as %q; want the bytes read, \"red pixels\"", read[0].Data)
	}

	clips := " clip-0.mp4\nclip-1.mp4\n"
	clipped := corpus(map[string]string{"clip-0.mp4": "a", "clip-1.mp4": "b"}, clips)
	admitted := testcheck.Must(readClipCorpus(clipped))(t)
	other := testcheck.Must(readClipCorpus(corpus(map[string]string{"clip-0.mp4": "a", "clip-1.mp4": "c"}, clips)))(t)
	if dataset(admitted.identity) == dataset(other.identity) {
		t.Error("two clip lists of one text naming clips of other bytes are one dataset; want two")
	}
	if !bytes.Equal(admitted.identity, trainingworkflow.MediaIdentity([]byte(clips), [][]byte{[]byte("a"), []byte("b")})) {
		t.Error("the codec's identity is not its manifest and the clips it names, as it parses them")
	}
	if path := testcheck.Must(admitted.admitted(0))(t); filepath.Base(path) != "clip-0.mp4" {
		t.Errorf("the admitted clip-0.mp4 is decoded from %s", path)
	}
	testcheck.NoError(t, os.WriteFile(filepath.Join(filepath.Dir(clipped), "clip-1.mp4"), []byte("d"), 0o600))
	if _, err := admitted.admitted(1); err == nil || !strings.Contains(err.Error(), "clip-1.mp4 is not the clip admitted") {
		t.Errorf("clip-1.mp4 changed after admission = %v; want it refused, naming it", err)
	}
}
