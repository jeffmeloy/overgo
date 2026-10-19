package testutil

import (
	"archive/tar"
	"compress/gzip"
	"io"
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/artifact"
)

const (
	// fixtureDirectoryMode gives the test owner traversal and write access only.
	fixtureDirectoryMode = 0o700
	// fixtureFileMode gives the test owner read and write access only.
	fixtureFileMode = 0o600
)

// ExtractTarGzip verifies a pinned fixture and extracts regular files into a test directory.
func ExtractTarGzip(t testing.TB, path, digest string) string {
	t.Helper()
	id, err := artifact.IdentifyFile(artifact.KindFile, path)
	if err != nil || id.DigestHex() != digest {
		t.Fatal("fixture identity", path, err)
	}
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	gz, err := gzip.NewReader(f)
	if err != nil {
		t.Fatal(err)
	}
	defer gz.Close()
	tr, root := tar.NewReader(gz), t.TempDir()
	for {
		h, err := tr.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			t.Fatal(err)
		}
		if h.Typeflag != tar.TypeReg || !filepath.IsLocal(h.Name) {
			t.Fatal("unsafe fixture member", h.Name)
		}
		p := filepath.Join(root, h.Name)
		if err = os.MkdirAll(filepath.Dir(p), fixtureDirectoryMode); err != nil {
			t.Fatal(err)
		}
		out, err := os.OpenFile(p, os.O_WRONLY|os.O_CREATE|os.O_EXCL, fixtureFileMode)
		if err != nil {
			t.Fatal(err)
		}
		_, copyErr := io.Copy(out, tr)
		closeErr := out.Close()
		if copyErr != nil || closeErr != nil {
			t.Fatal(copyErr, closeErr)
		}
	}
	return root
}
