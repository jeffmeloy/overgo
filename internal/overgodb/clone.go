package overgodb

import (
	"errors"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
)

// CloneReport counts one store clone.
type CloneReport struct {
	LinkedBlobs int
	CopiedFiles int
	CopiedBytes int64
}

// CloneStore clones the store under sourceRoot into a fresh destinationRoot:
// blobs become hard links, everything else -- journal, segments, snapshots,
// checkpoints -- is copied. Blob files are never modified in place
// (publication stages then renames, release unlinks), so a shared inode is
// safe and a candidate store costs seconds instead of a full byte copy. The
// lock, a backup seal and staging leftovers are not carried: the clone is a
// store, not a backup. Both roots must sit on one volume.
func CloneStore(sourceRoot, destinationRoot string) (CloneReport, error) {
	report := CloneReport{}
	if _, err := os.Lstat(destinationRoot); err == nil {
		return report, fmt.Errorf("overgodb: clone destination %q already exists", destinationRoot)
	} else if !errors.Is(err, os.ErrNotExist) {
		return report, err
	}
	blobPrefix := blobDirectory + "/"
	err := filepath.WalkDir(sourceRoot, func(path string, entry fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		relative, err := filepath.Rel(sourceRoot, path)
		if err != nil {
			return err
		}
		target := filepath.Join(destinationRoot, relative)
		if entry.IsDir() {
			return os.MkdirAll(target, storeDirectoryMode)
		}
		name := filepath.ToSlash(relative)
		if name == lockFilename || name == backupSealFilename || strings.HasPrefix(entry.Name(), ".staging-") {
			return nil
		}
		if strings.HasPrefix(name, blobPrefix) {
			report.LinkedBlobs++
			return os.Link(path, target)
		}
		copied, err := copyStoreFile(path, target)
		report.CopiedFiles++
		report.CopiedBytes += copied
		return err
	})
	return report, err
}

// copyStoreFile copies one file into a path that must not exist.
func copyStoreFile(source, target string) (int64, error) {
	input, err := os.Open(source)
	if err != nil {
		return 0, err
	}
	defer input.Close()
	output, err := os.OpenFile(target, os.O_CREATE|os.O_EXCL|os.O_WRONLY, storeFileMode)
	if err != nil {
		return 0, err
	}
	copied, err := io.Copy(output, input)
	return copied, errors.Join(err, output.Close())
}
