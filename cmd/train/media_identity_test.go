package main

import (
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/testcheck"
	"overgo/internal/trainingworkflow"
)

// TestTwoCorporaWithOneManifestTextAreTwoDatasets holds a route that trains from media to a dataset identity of the
// media it reads (SQA-507): two directories with one pictures.tsv, red.png holding green's bytes in the second, are two
// datasets, as are a codec clip list naming clips of other bytes; the same files copied to another directory are one.
// The manifest's bytes alone had named both corpora alike, and so had their split and recipe.
func TestTwoCorporaWithOneManifestTextAreTwoDatasets(t *testing.T) {
	t.Parallel()
	corpus := func(files map[string]string, manifest string) string {
		directory := t.TempDir()
		for name, content := range files {
			testcheck.NoError(t, os.WriteFile(filepath.Join(directory, name), []byte(content), 0o600))
		}
		path := filepath.Join(directory, "pictures.tsv")
		testcheck.NoError(t, os.WriteFile(path, []byte(manifest), 0o600))
		return path
	}
	dataset := func(manifest string) string {
		bytes := testcheck.Must(trainingworkflow.MediaManifestBytes(manifest))(t)
		id, _, _, err := trainingworkflow.TextAuthority(bytes)
		testcheck.NoError(t, err)
		return id.String()
	}
	pictures := "red.png\ta red square\ngreen.png\ta green square\n"
	first := corpus(map[string]string{"red.png": "red pixels", "green.png": "green pixels"}, pictures)
	copied := corpus(map[string]string{"red.png": "red pixels", "green.png": "green pixels"}, pictures)
	swapped := corpus(map[string]string{"red.png": "green pixels", "green.png": "green pixels"}, pictures)
	if dataset(first) != dataset(copied) {
		t.Error("one corpus in two directories is two datasets; want one")
	}
	if dataset(first) == dataset(swapped) {
		t.Error("two corpora with one manifest text and red.png's bytes changed are one dataset; want two")
	}
	clips := "clip-0.mp4\nclip-1.mp4\n"
	if dataset(corpus(map[string]string{"clip-0.mp4": "a", "clip-1.mp4": "b"}, clips)) == dataset(corpus(map[string]string{"clip-0.mp4": "a", "clip-1.mp4": "c"}, clips)) {
		t.Error("two clip lists of one text naming clips of other bytes are one dataset; want two")
	}
}
