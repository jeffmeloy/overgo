package main

import (
	"context"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/cuda/kernel"
	"overgo/internal/dataroot"
	"overgo/internal/gitauthority"
	"overgo/internal/jsonfile"
	"overgo/internal/media"
	"overgo/internal/overgodb"
	"overgo/internal/plan"
	"overgo/internal/testevidence"
	"overgo/internal/testutil"
)

// The backup owner writes this seal after byte verification and replay. Check
// its captured commit, rather than comparing with the changing source head.
type imageVideoBackupIdentity struct {
	Source   string            `json:"source"`
	Head     artifact.CommitID `json:"head"`
	Sequence uint64            `json:"sequence"`
	Extent   int64             `json:"extent"`
}

func checkImageVideoWorktreeReadiness(ctx context.Context, root string, roots dataroot.Roots, census artifact.ID) (imageVideoBackupIdentity, error) {
	var identity imageVideoBackupIdentity
	canonical := filepath.Join(root, gitauthority.CanonicalOvergoDBDirectory)
	for _, path := range []string{canonical, roots.Store, roots.Models, roots.Datasets, roots.Checkpoints} {
		info, err := os.Stat(path)
		if err != nil || !info.IsDir() {
			return identity, fmt.Errorf("media readiness: missing directory %q: %v", path, err)
		}
	}
	resolved, err := filepath.EvalSymlinks(roots.Store)
	if err != nil {
		return identity, err
	}
	resolved, err = filepath.Abs(resolved)
	if err != nil {
		return identity, err
	}
	canonical, err = filepath.Abs(canonical)
	if err != nil {
		return identity, err
	}
	if resolved != canonical {
		return identity, errors.New("media readiness: writable store must resolve to this worktree's canonical directory")
	}
	if err := jsonfile.DecodeStrict(filepath.Join(canonical, "backup-seal.json"), &identity); err != nil {
		return identity, fmt.Errorf("media readiness: backup publication identity: %w", err)
	}
	if !identity.Head.Valid() || identity.Sequence == 0 || identity.Extent <= 0 || !filepath.IsAbs(identity.Source) {
		return identity, errors.New("media readiness: incomplete backup identity")
	}
	// Check all copied files, not just the directory name: a junction or a
	// hardlinked journal/blob would let local writes affect the source store.
	err = filepath.WalkDir(canonical, func(path string, entry fs.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		if err := ctx.Err(); err != nil {
			return err
		}
		if entry.Type()&os.ModeSymlink != 0 {
			return fmt.Errorf("media readiness: linked store entry %q", path)
		}
		if entry.IsDir() {
			return nil
		}
		relative, err := filepath.Rel(canonical, path)
		if err != nil {
			return err
		}
		source, err := os.Stat(filepath.Join(identity.Source, relative))
		if os.IsNotExist(err) {
			return nil // Files created locally after the snapshot have no source counterpart.
		}
		if err != nil {
			return err
		}
		local, err := entry.Info()
		if err != nil {
			return err
		}
		if os.SameFile(local, source) {
			return fmt.Errorf("media readiness: file shares source storage: %q", path)
		}
		return nil
	})
	if err != nil {
		return identity, err
	}
	store, err := overgodb.OpenReadOnly(canonical)
	if err != nil {
		return identity, err
	}
	defer store.Close()
	// One exact sequence has one commit. Local appends may advance the head
	// without changing the imported snapshot's identity.
	result, err := store.Query(ctx, overgodb.Query{FromSequence: identity.Sequence, ToSequence: identity.Sequence, Projection: overgodb.ProjectCommits, MaxResults: 1})
	if err != nil {
		return identity, err
	}
	if result.Truncated || len(result.Commits) != 1 || result.Commits[0].ID != identity.Head {
		return identity, errors.New("media readiness: backup commit does not match local replay")
	}
	if census.Kind() != artifact.KindEvidence {
		return identity, errors.New("media readiness: inherited census is not evidence")
	}
	if _, found, err := artifact.ReadContent(ctx, store, census); err != nil || !found {
		return identity, fmt.Errorf("media readiness: inherited census content unavailable: %v", err)
	}
	return identity, nil
}

func TestImageVideoWorktreeReadiness(t *testing.T) {
	if testing.Short() {
		t.Skip(testevidence.ShortIntegrationSkip)
	}
	if os.Getenv(dataroot.Env) == "" {
		t.Skip("integration: set OVERGO_DATA_ROOT for image/video worktree readiness")
	}
	root := testutil.RepoRoot(t)
	document, err := plan.Load(filepath.Join(root, plan.Path))
	if err != nil {
		t.Fatal(err)
	}
	if document.Lane != "image_video_gen" {
		t.Skip("integration: readiness applies to the image_video_gen campaign")
	}
	if document.Census == nil {
		t.Fatal("campaign has no inherited census identity")
	}
	roots, err := dataroot.Resolve(root)
	if err != nil {
		t.Fatal(err)
	}
	// The gate verifies an immutable candidate checkout while explicitly
	// binding external data to the original worktree through OVERGO_DATA_ROOT.
	dataRoot, err := gitauthority.RepositoryRoot(t.Context(), os.Getenv(dataroot.Env))
	if err != nil {
		t.Fatal(err)
	}
	identity, err := checkImageVideoWorktreeReadiness(t.Context(), dataRoot, roots, *document.Census)
	if err != nil {
		t.Fatal(err)
	}
	if err := kernel.ValidateAssets(); err != nil {
		t.Fatal(err)
	}
	ffmpeg, err := media.ResolveFFmpeg("")
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("private store=%s imported head=%s sequence=%d; models=%s datasets=%s checkpoints=%s; FFmpeg=%s; no model execution", roots.Store, identity.Head, identity.Sequence, roots.Models, roots.Datasets, roots.Checkpoints, ffmpeg)
}

func TestImageVideoWorktreeReadinessRejectsInvalidState(t *testing.T) {
	root := t.TempDir()
	source, err := overgodb.Open(filepath.Join(t.TempDir(), "source"))
	if err != nil {
		t.Fatal(err)
	}
	defer source.Close()
	payload := []byte("retained census")
	content := artifact.Content{Descriptor: artifact.Descriptor{ID: testutil.ArtifactBytesID(t, artifact.KindEvidence, payload), Size: uint64(len(payload)), MediaType: "text/plain"}, Data: payload}
	if _, err := source.Commit(t.Context(), artifact.Batch{Key: "readiness/source", Contents: []artifact.Content{content}}); err != nil {
		t.Fatal(err)
	}
	roots := dataroot.Roots{Store: filepath.Join(root, gitauthority.CanonicalOvergoDBDirectory), Models: t.TempDir(), Datasets: t.TempDir(), Checkpoints: t.TempDir()}
	if _, err := source.Backup(t.Context(), roots.Store); err != nil {
		t.Fatal(err)
	}
	identity, err := checkImageVideoWorktreeReadiness(t.Context(), root, roots, content.Descriptor.ID)
	if err != nil {
		t.Fatal(err)
	}
	for _, field := range []string{"store", "models", "datasets", "checkpoints"} {
		t.Run("missing-"+field, func(t *testing.T) {
			bad := roots
			fields := map[string]*string{"store": &bad.Store, "models": &bad.Models, "datasets": &bad.Datasets, "checkpoints": &bad.Checkpoints}
			*fields[field] = filepath.Join(root, "absent")
			if _, err := checkImageVideoWorktreeReadiness(t.Context(), root, bad, content.Descriptor.ID); err == nil {
				t.Fatal("missing root accepted")
			}
		})
	}
	t.Run("external-store", func(t *testing.T) {
		bad := roots
		bad.Store = identity.Source
		if _, err := checkImageVideoWorktreeReadiness(t.Context(), root, bad, content.Descriptor.ID); err == nil || !strings.Contains(err.Error(), "canonical directory") {
			t.Fatalf("external writable store: %v", err)
		}
	})
	t.Run("wrong-backup-head", func(t *testing.T) {
		bad := identity
		bad.Sequence++
		sealPath := filepath.Join(roots.Store, "backup-seal.json")
		if err := jsonfile.Write(sealPath, bad, 0o600); err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() {
			if err := jsonfile.Write(sealPath, identity, 0o600); err != nil {
				t.Error(err)
			}
		})
		if _, err := checkImageVideoWorktreeReadiness(t.Context(), root, roots, content.Descriptor.ID); err == nil || !strings.Contains(err.Error(), "backup commit") {
			t.Fatalf("mismatched snapshot: %v", err)
		}
	})
	t.Run("missing-census", func(t *testing.T) {
		missing := testutil.ArtifactBytesID(t, artifact.KindEvidence, []byte("absent"))
		if _, err := checkImageVideoWorktreeReadiness(t.Context(), root, roots, missing); err == nil || !strings.Contains(err.Error(), "census content unavailable") {
			t.Fatalf("missing inherited content: %v", err)
		}
	})
	t.Run("hardlinked-file", func(t *testing.T) {
		linked := filepath.Join(roots.Store, "overgodb.log.shared")
		sourceFile := filepath.Join(identity.Source, "overgodb.log.shared")
		if err := os.WriteFile(sourceFile, []byte("shared"), 0o600); err != nil {
			t.Fatal(err)
		}
		if err := os.Link(sourceFile, linked); err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { _ = os.Remove(linked) })
		if _, err := checkImageVideoWorktreeReadiness(t.Context(), root, roots, content.Descriptor.ID); err == nil || !strings.Contains(err.Error(), "shares source storage") {
			t.Fatalf("hardlinked source storage: %v", err)
		}
	})
	t.Run("local-append", func(t *testing.T) {
		local, err := overgodb.Open(roots.Store)
		if err != nil {
			t.Fatal(err)
		}
		_, commitErr := local.Commit(t.Context(), artifact.Batch{Key: "readiness/local", Aliases: []artifact.AliasBinding{{Name: "readiness/local", Target: content.Descriptor.ID}}})
		if err := errors.Join(commitErr, local.Close()); err != nil {
			t.Fatal(err)
		}
		if _, err := checkImageVideoWorktreeReadiness(t.Context(), root, roots, content.Descriptor.ID); err != nil {
			t.Fatalf("valid local append invalidated the imported snapshot: %v", err)
		}
	})
}
