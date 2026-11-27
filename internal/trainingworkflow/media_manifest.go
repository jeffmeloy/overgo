package trainingworkflow

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"os"
	"path/filepath"
	"slices"
)

// MediaManifestBytes are the bytes a route's dataset is identified by where its manifest names a media file a line, the
// file first and any caption after a tab, relative to the manifest: the manifest's own bytes and, after them, the SHA-256
// of each file it names, in its order. Two corpora with one manifest text and other media are two datasets, and a corpus
// copied to another directory is one: the manifest's bytes alone had named both corpora alike, MOSS's pictures and the
// codec route's clips (SQA-507).
func MediaManifestBytes(manifest string) ([]byte, error) {
	raw, err := os.ReadFile(manifest)
	if err != nil {
		return nil, err
	}
	identity := append(slices.Clone(raw), "\x00media"...)
	for line := range bytes.Lines(raw) {
		named, _, _ := bytes.Cut(bytes.TrimSpace(line), []byte{'\t'})
		if len(named) == 0 {
			continue
		}
		data, err := os.ReadFile(filepath.Join(filepath.Dir(manifest), string(named)))
		if err != nil {
			return nil, err
		}
		sum := sha256.Sum256(data)
		identity = hex.AppendEncode(append(identity, '\n'), sum[:])
	}
	return identity, nil
}
