package hfrepo

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"slices"
	"strings"

	"overgo/internal/gitauthority"
	"overgo/internal/pathidentity"
)

// SourceReceipt binds a repository revision to its complete downloaded file inventory.
// Verification checks local bytes and does not execute source or activate a model.
type SourceReceipt struct {
	Source   string       `json:"source"`
	Revision string       `json:"revision"`
	Files    []SourceFile `json:"files"`
}

// SourceFile identifies one repository-relative payload.
type SourceFile struct {
	Path   string `json:"path"`
	SHA256 string `json:"sha256"`
	Bytes  int64  `json:"bytes"`
}

// Covers reports whether a payload has exactly the declared source facts.
func (r SourceReceipt) Covers(file SourceFile) bool { return slices.Contains(r.Files, file) }

// Verify checks every declared payload and returns a canonical receipt.
// Source authenticity belongs to the pinned acquisition that produced the receipt.
// Additional files cannot acquire provenance from this receipt.
func (r SourceReceipt) Verify(ctx context.Context, directory string) (SourceReceipt, error) {
	if ctx == nil || strings.TrimSpace(r.Source) == "" || !gitauthority.ValidObjectID(r.Revision) || len(r.Files) == 0 {
		return SourceReceipt{}, errors.New("model repository: source receipt requires context, repository, immutable revision and files")
	}
	r.Files = slices.Clone(r.Files)
	slices.SortFunc(r.Files, func(a, b SourceFile) int { return strings.Compare(a.Path, b.Path) })
	previous := ""
	r.Revision = strings.ToLower(r.Revision)
	for index := range r.Files {
		file := &r.Files[index]
		file.SHA256 = strings.ToLower(file.SHA256)
		if err := ctx.Err(); err != nil {
			return SourceReceipt{}, err
		}
		digest, err := hex.DecodeString(file.SHA256)
		if file.Path == "" || file.Path == previous || !filepath.IsLocal(file.Path) || filepath.ToSlash(filepath.Clean(file.Path)) != file.Path || strings.ContainsAny(file.Path, "\\:\x00") || err != nil || len(digest) != sha256.Size || file.Bytes < 0 {
			return SourceReceipt{}, fmt.Errorf("model repository: invalid or duplicate source payload %q", file.Path)
		}
		previous = file.Path
		filename := filepath.Join(directory, filepath.FromSlash(file.Path))
		contained, err := pathidentity.Contains(directory, filename)
		if err != nil || !contained {
			return SourceReceipt{}, fmt.Errorf("model repository: source payload %q escapes its directory", file.Path)
		}
		handle, err := os.Open(filename)
		if err != nil {
			return SourceReceipt{}, err
		}
		hash := sha256.New()
		// Context checks run between bounded reads rather than after an entire shard.
		size, copyErr := io.Copy(hash, sourceReader{ctx: ctx, reader: handle})
		closeErr := handle.Close()
		if err := errors.Join(copyErr, closeErr); err != nil {
			return SourceReceipt{}, err
		}
		if size != file.Bytes || !strings.EqualFold(hex.EncodeToString(hash.Sum(nil)), file.SHA256) {
			return SourceReceipt{}, fmt.Errorf("model repository: source payload %q differs from declared size or SHA-256", file.Path)
		}
	}
	return r, ctx.Err()
}

type sourceReader struct {
	ctx    context.Context
	reader io.Reader
}

// Read checks cancellation before reading the next payload chunk.
func (r sourceReader) Read(p []byte) (int, error) {
	if err := r.ctx.Err(); err != nil {
		return 0, err
	}
	return r.reader.Read(p)
}
