package overgodb

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/fsatomic"
	"overgo/internal/jsonfile"
	"overgo/internal/processlock"
)

// BackupReport binds publication to its captured identity and physical IO.
// Counters cover store files; seal IO and metadata replay are excluded.
type BackupReport struct {
	Head                                             artifact.CommitID
	Sequence                                         uint64
	Extent                                           int64
	BytesRead, BytesCopied, BytesReused, BytesSynced int64
	FilesRead, FilesCopied, FilesReused, FilesSynced int64
	Elapsed                                          time.Duration
}

// Backup resumes an unpublished copy, verifies bytes, then publishes its head.
// Published destinations are immutable; callers must retire them before reuse.
func (s *Store) Backup(ctx context.Context, destinationRoot string) (*BackupReport, error) {
	report := &BackupReport{}
	started := time.Now()
	defer func() { report.Elapsed = time.Since(started) }()
	if destinationRoot == "" {
		return report, errors.New("overgodb: empty backup destination")
	}
	sourceRoot, err := filepath.Abs(s.root)
	if err != nil {
		return report, err
	}
	destinationRoot, err = filepath.Abs(destinationRoot)
	if err != nil {
		return report, err
	}
	if err := os.MkdirAll(filepath.Dir(destinationRoot), storeDirectoryMode); err != nil {
		return report, err
	}
	sourceRoot, err = filepath.EvalSymlinks(sourceRoot)
	if err != nil {
		return report, err
	}
	parent, err := filepath.EvalSymlinks(filepath.Dir(destinationRoot))
	if err != nil {
		return report, err
	}
	destinationRoot = filepath.Join(parent, filepath.Base(destinationRoot))
	for _, candidate := range []string{destinationRoot, destinationRoot + ".partial"} {
		for _, pair := range [][2]string{{sourceRoot, candidate}, {candidate, sourceRoot}} {
			relative, err := filepath.Rel(pair[0], pair[1])
			if err == nil && relative != ".." && !strings.HasPrefix(relative, ".."+string(filepath.Separator)) {
				return report, errors.New("overgodb: backup source and destination overlap")
			}
		}
	}
	destinationLock, err := processlock.AcquireContext(ctx, destinationRoot+".lock", storeFileMode)
	if err != nil {
		return report, err
	}
	defer destinationLock.Close()
	if _, err := os.Lstat(destinationRoot); err == nil {
		return report, fmt.Errorf("overgodb: backup destination %q already exists", destinationRoot)
	} else if !os.IsNotExist(err) {
		return report, err
	}
	// Pin mutable journal and checkpoint paths through capture.
	lock, err := processlock.AcquireContext(ctx, filepath.Join(s.root, lockFilename), storeFileMode)
	if err != nil {
		return report, err
	}
	defer lock.Close()
	s.mu.Lock()
	if err := s.ready(false); err != nil {
		s.mu.Unlock()
		return report, err
	}
	if err := s.refreshLocked(); err != nil {
		s.mu.Unlock()
		return report, err
	}
	report.Head, report.Sequence, report.Extent = s.head, s.sequence, s.replayEnd
	requiredBlobs := make([]artifact.ID, 0, len(s.state.contents.locators))
	for id, locator := range s.state.contents.locators {
		if locator.blob {
			requiredBlobs = append(requiredBlobs, id)
		}
	}
	s.mu.Unlock()
	if !report.Head.Valid() {
		return report, errors.New("overgodb: refusing to back up a store with no commits")
	}
	partial := destinationRoot + ".partial"
	if err := backupDirectory(partial); err != nil {
		return report, err
	}
	sealPath := filepath.Join(partial, "backup-seal.json")
	var seal backupSeal
	durable := jsonfile.DecodeStrict(sealPath, &seal) == nil && seal.Source == sourceRoot && seal.Head.Valid() && seal.Sequence != 0 && seal.Extent > 0
	// Invalidate durability before changing any file. A crash requires resync.
	if err := fsatomic.Remove(sealPath); err != nil {
		return report, err
	}
	if err := report.copyTree(ctx, filepath.Join(sourceRoot, segmentDirectory), filepath.Join(partial, segmentDirectory), durable); err != nil {
		return report, err
	}
	if err := report.copyFile(ctx, filepath.Join(sourceRoot, storeFilename), filepath.Join(partial, storeFilename), report.Extent, "", durable); err != nil {
		return report, err
	}
	if err := report.copyTree(ctx, filepath.Join(sourceRoot, checkpointDirectory), filepath.Join(partial, checkpointDirectory), durable); err != nil {
		return report, err
	}
	// Published blobs are immutable and retained across commits and rotation.
	// The captured metadata and blob identities no longer depend on the writer.
	if err := lock.Close(); err != nil {
		return report, err
	}
	sourceBlobs, destinationBlobs := newBlobStore(sourceRoot), newBlobStore(partial)
	if err := backupDirectory(destinationBlobs.root); err != nil {
		return report, err
	}
	retainedBlobs := make(map[string]bool, len(requiredBlobs))
	for _, id := range requiredBlobs {
		sourcePath, err := sourceBlobs.path(id)
		if err != nil {
			return report, err
		}
		destinationPath, err := destinationBlobs.path(id)
		if err != nil {
			return report, err
		}
		retainedBlobs[destinationPath] = true
		if err := backupDirectory(filepath.Dir(filepath.Dir(destinationPath))); err != nil {
			return report, err
		}
		info, err := os.Stat(sourcePath)
		if err != nil {
			return report, fmt.Errorf("overgodb: backup blob %s: %w", id, err)
		}
		if err := report.copyFile(ctx, sourcePath, destinationPath, info.Size(), id.DigestHex(), durable); err != nil {
			return report, err
		}
	}
	// The seal covers only verified files. Retire scratch/orphan copies so a
	// later head cannot mistake an unverified extra for previously synced data.
	if err := filepath.WalkDir(destinationBlobs.root, func(path string, entry fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if err := ctx.Err(); err != nil {
			return err
		}
		if entry.IsDir() || retainedBlobs[path] {
			return nil
		}
		return fsatomic.Remove(path)
	}); err != nil {
		return report, err
	}
	if err := ctx.Err(); err != nil {
		return report, err
	}
	copyHead, copySequence, err := replayedHead(partial)
	if err != nil {
		return report, fmt.Errorf("overgodb: backup verification: %w", err)
	}
	if copyHead != report.Head || copySequence != report.Sequence {
		return report, fmt.Errorf("overgodb: backup replay head %s@%d does not match captured %s@%d", copyHead, copySequence, report.Head, report.Sequence)
	}
	if err := ctx.Err(); err != nil {
		return report, err
	}
	if err := jsonfile.Write(sealPath, backupSeal{sourceRoot, report.Head, report.Sequence, report.Extent}, storeFileMode); err != nil {
		return report, err
	}
	if err := ctx.Err(); err != nil {
		return report, err
	}
	return report, fsatomic.Replace(partial, destinationRoot)
}

// backupDirectory refuses links before traversing reusable scratch space.
func backupDirectory(path string) error {
	info, err := os.Lstat(path)
	if os.IsNotExist(err) {
		parent := filepath.Dir(path)
		if parent != path {
			if err := backupDirectory(parent); err != nil {
				return err
			}
		}
		if err := os.Mkdir(path, storeDirectoryMode); err != nil {
			return err
		}
		return fsatomic.SyncDirectory(parent)
	}
	if err != nil {
		return err
	}
	if !info.IsDir() || info.Mode()&os.ModeSymlink != 0 {
		return fmt.Errorf("overgodb: backup directory %q is not a real directory", path)
	}
	return nil
}

func (report *BackupReport) copyTree(ctx context.Context, source, destination string, durable bool) error {
	entries, err := os.ReadDir(source)
	if err != nil && !os.IsNotExist(err) {
		return err
	}
	if err := backupDirectory(destination); err != nil {
		return err
	}
	retained := make(map[string]bool, len(entries))
	for _, entry := range entries {
		if err := ctx.Err(); err != nil {
			return err
		}
		retained[entry.Name()] = true
		sourcePath, destinationPath := filepath.Join(source, entry.Name()), filepath.Join(destination, entry.Name())
		if entry.IsDir() {
			if err := report.copyTree(ctx, sourcePath, destinationPath, durable); err != nil {
				return err
			}
			continue
		}
		info, err := entry.Info()
		if err != nil {
			return err
		}
		if !info.Mode().IsRegular() {
			return fmt.Errorf("overgodb: backup source %q is not a regular file", sourcePath)
		}
		if err := report.copyFile(ctx, sourcePath, destinationPath, info.Size(), "", durable); err != nil {
			return err
		}
	}
	previous, err := os.ReadDir(destination)
	if err != nil {
		return err
	}
	for _, entry := range previous {
		if err := ctx.Err(); err != nil {
			return err
		}
		if !retained[entry.Name()] {
			if err := os.RemoveAll(filepath.Join(destination, entry.Name())); err != nil {
				return err
			}
		}
	}
	return fsatomic.SyncDirectory(destination)
}

func (report *BackupReport) copyFile(ctx context.Context, source, destination string, extent int64, expected string, durable bool) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if err := backupDirectory(filepath.Dir(destination)); err != nil {
		return err
	}
	if info, err := os.Lstat(destination); err == nil {
		if !info.Mode().IsRegular() {
			return fmt.Errorf("overgodb: backup candidate %q is not a regular file", destination)
		}
		sourceInfo, err := os.Stat(source)
		if err != nil {
			return err
		}
		if info.Size() == extent && !os.SameFile(info, sourceInfo) {
			candidate, err := report.hashFile(ctx, destination, extent)
			if err != nil {
				return err
			}
			identity := expected
			if identity == "" {
				identity, err = report.hashFile(ctx, source, extent)
				if err != nil {
					return err
				}
			}
			if candidate == identity {
				// Unsealed attempts may have stopped before syncing.
				if !durable {
					if err := fsatomic.SyncFile(destination); err != nil {
						return err
					}
					report.FilesSynced++
					report.BytesSynced += extent
				}
				report.FilesReused++
				report.BytesReused += extent
				return nil
			}
		}
	} else if !os.IsNotExist(err) {
		return err
	}
	input, err := os.Open(source)
	if err != nil {
		return err
	}
	defer input.Close()
	output, err := os.CreateTemp(filepath.Dir(destination), ".backup-*")
	if err != nil {
		return err
	}
	defer func() { _ = output.Close(); _ = os.Remove(output.Name()) }()
	hash := sha256.New()
	n, err := io.CopyN(io.MultiWriter(output, hash), &backupReader{ctx: ctx, source: input}, extent)
	report.FilesRead++
	report.BytesRead += n
	report.FilesCopied++
	report.BytesCopied += n
	if err != nil {
		return err
	}
	if expected != "" && hex.EncodeToString(hash.Sum(nil)) != expected {
		return fmt.Errorf("overgodb: backup source %q bytes do not hash to their identity", source)
	}
	if err := output.Sync(); err != nil {
		return err
	}
	report.FilesSynced++
	report.BytesSynced += extent
	if err := output.Close(); err != nil {
		return err
	}
	return fsatomic.Replace(output.Name(), destination)
}

func (report *BackupReport) hashFile(ctx context.Context, path string, extent int64) (string, error) {
	file, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer file.Close()
	hash := sha256.New()
	n, err := io.CopyN(hash, &backupReader{ctx: ctx, source: file}, extent)
	report.FilesRead++
	report.BytesRead += n
	return hex.EncodeToString(hash.Sum(nil)), err
}

type backupReader struct {
	ctx    context.Context
	source io.Reader
}

// Read checks cancellation between source reads.
func (reader *backupReader) Read(buffer []byte) (int, error) {
	if err := reader.ctx.Err(); err != nil {
		return 0, err
	}
	return reader.source.Read(buffer)
}

func replayedHead(root string) (artifact.CommitID, uint64, error) {
	store, err := OpenReadOnly(root)
	if err != nil {
		return artifact.CommitID{}, 0, err
	}
	head, sequence := store.Head()
	return head, sequence, store.Close()
}

// A seal attests completed file syncs, never substitutes for byte verification.
type backupSeal struct {
	Source   string            `json:"source"`
	Head     artifact.CommitID `json:"head"`
	Sequence uint64            `json:"sequence"`
	Extent   int64             `json:"extent"`
}
