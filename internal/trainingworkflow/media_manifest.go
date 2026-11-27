package trainingworkflow

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"os"
	"path/filepath"
	"slices"
)

// MediaIdentity are the bytes a route's dataset is identified by where its manifest names media files: the manifest's
// own bytes and, after them, the SHA-256 of each file's bytes as the route read them, in the manifest's order. Two
// corpora with one manifest text and other media are two datasets, and a corpus copied to another directory is one
// (SQA-507); a route that hashes the bytes it trains on names no file it did not read.
func MediaIdentity(manifest []byte, files [][]byte) []byte {
	identity := append(slices.Clone(manifest), "\x00media"...)
	for _, data := range files {
		sum := sha256.Sum256(data)
		identity = hex.AppendEncode(append(identity, '\n'), sum[:])
	}
	return identity
}

// ManifestPicture is a line of a pictures manifest: the picture's path, its caption, and its bytes, read once.
type ManifestPicture struct {
	Path, Caption string
	Data          []byte
}

// PictureManifest reads a manifest of a picture and its caption a line, tab-separated, the picture relative to the
// manifest and kept as written, each picture's bytes read once, and the bytes its dataset is identified by
// (MediaIdentity): MOSS's route trains on those bytes and its proof is grounded on the same identity, where the identity
// had trimmed a line the route did not and both read the pictures again after hashing them (SQA-507).
func PictureManifest(manifest string) ([]ManifestPicture, []byte, error) {
	raw, err := os.ReadFile(manifest)
	if err != nil {
		return nil, nil, err
	}
	var pictures []ManifestPicture
	var files [][]byte
	for line := range bytes.Lines(raw) {
		line = bytes.TrimRight(line, "\r\n")
		if len(line) == 0 {
			continue
		}
		path, caption, found := bytes.Cut(line, []byte{'\t'})
		if !found || len(path) == 0 || len(caption) == 0 {
			return nil, nil, fmt.Errorf("manifest %s: a line needs a picture, a tab and a caption: %q", manifest, line)
		}
		picture := ManifestPicture{Path: filepath.Join(filepath.Dir(manifest), string(path)), Caption: string(caption)}
		if picture.Data, err = os.ReadFile(picture.Path); err != nil {
			return nil, nil, err
		}
		pictures, files = append(pictures, picture), append(files, picture.Data)
	}
	if len(pictures) == 0 {
		return nil, nil, fmt.Errorf("manifest %s names no pictures", manifest)
	}
	return pictures, MediaIdentity(raw, files), nil
}
