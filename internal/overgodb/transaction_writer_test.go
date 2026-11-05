package overgodb

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"testing"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/processlock"
	"overgo/internal/testcheck"
)

func TestTransactionWriter(t *testing.T) {
	t.Parallel()
	for _, cancelGap := range []bool{false, true} {
		t.Run(fmt.Sprintf("cutover owns the rename gap/cancel=%t", cancelGap), func(t *testing.T) {
			live, candidate := filepath.Join(t.TempDir(), "live"), filepath.Join(t.TempDir(), "candidate")
			store, err := Create(live)
			testcheck.NoError(t, err)
			batch := fixtureBatch(t)
			original, err := store.Commit(t.Context(), batch)
			testcheck.NoError(t, errors.Join(err, store.Close()))
			if _, err := CloneStore(live, candidate); err != nil {
				t.Fatal(err)
			}
			store, err = Create(candidate)
			testcheck.NoError(t, err)
			prepared, err := store.Commit(t.Context(), artifact.Batch{Key: "prepared", Artifacts: []artifact.Descriptor{fixtureDescriptor(t, artifact.KindOutput, "candidate")}})
			testcheck.NoError(t, errors.Join(err, store.Close()))
			ctx, cancel := context.WithCancelCause(t.Context())
			defer cancel(context.Canceled)
			superseded := fmt.Sprintf("%s.superseded-%.12s", live, original)
			observed := false
			boundary := &fileBoundary{Context: ctx, path: superseded}
			boundary.check = sync.OnceFunc(func() {
				observed = true
				// Startup must share exclusion even while live does not exist.
				for _, root := range []string{live, candidate} {
					if writer, err := Create(root); !errors.Is(err, processlock.ErrBusy) {
						if writer != nil {
							writer.Close()
						}
						t.Errorf("writer entered during replacement at %s: %v", root, err)
					}
				}
				if _, err := os.Stat(live); !os.IsNotExist(err) {
					t.Errorf("queued startup recreated live: %v", err)
				}
				queued, abandon := context.WithCancelCause(t.Context())
				abandon(context.Canceled)
				if writer, err := OpenContext(queued, live); !errors.Is(err, context.Canceled) {
					if writer != nil {
						writer.Close()
					}
					t.Errorf("cancelled startup during cutover: %v", err)
				}
				if cancelGap {
					cancel(context.Canceled)
				}
			})
			path, err := ReplaceStore(boundary, live, candidate, original.String())
			if !observed || cancelGap && (!errors.Is(err, context.Canceled) || path != "") || !cancelGap && (err != nil || path != superseded) {
				t.Fatalf("cutover observed=%t path=%s error=%v", observed, path, err)
			}
			store, err = Create(live)
			testcheck.NoError(t, err)
			defer store.Close()
			want := prepared
			if cancelGap {
				want = original
			}
			if head, _ := store.Head(); head != want {
				t.Fatalf("live head = %s, want %s", head, want)
			}
			if _, found, err := store.Artifact(t.Context(), batch.Artifacts[0].ID); err != nil || !found {
				t.Fatalf("cutover lost original artifact: %t %v", found, err)
			}
			if _, err := store.Commit(t.Context(), artifact.Batch{Key: "after-cutover", Artifacts: []artifact.Descriptor{fixtureDescriptor(t, artifact.KindOutput, "after cutover")}}); err != nil {
				t.Fatalf("cutover retained writer ownership: %v", err)
			}
			retained := candidate
			if !cancelGap {
				retained = superseded
				want = original
			} else {
				want = prepared
			}
			reader, err := OpenReadOnly(retained)
			testcheck.NoError(t, err)
			defer reader.Close()
			if head, _ := reader.Head(); head != want {
				t.Fatalf("retained head = %s, want %s", head, want)
			}
		})
	}
	t.Run("idle handle cannot publish into a displaced journal", func(t *testing.T) {
		live, candidate := filepath.Join(t.TempDir(), "live"), filepath.Join(t.TempDir(), "candidate")
		idle, err := Create(live)
		testcheck.NoError(t, err)
		defer idle.Close()
		original, err := idle.Commit(t.Context(), fixtureBatch(t))
		testcheck.NoError(t, err)
		if _, err := CloneStore(live, candidate); err != nil {
			t.Fatal(err)
		}
		_, err = ReplaceStore(t.Context(), live, candidate, original.String())
		if runtime.GOOS == "windows" {
			// Win32 refuses renaming the open journal; neither store may move.
			if err == nil {
				t.Fatal("replacement moved an open Windows journal")
			}
			for _, root := range []string{live, candidate} {
				if _, err := os.Stat(filepath.Join(root, storeFilename)); err != nil {
					t.Fatal(err)
				}
			}
			return
		}
		testcheck.NoError(t, err)
		if _, err := idle.Commit(t.Context(), artifact.Batch{Key: "displaced", Artifacts: []artifact.Descriptor{fixtureDescriptor(t, artifact.KindOutput, "must not publish")}}); err == nil || !strings.Contains(err.Error(), "was replaced") {
			t.Fatalf("displaced writer: %v", err)
		}
		if err := idle.Refresh(t.Context()); err == nil {
			t.Fatal("displaced view silently refreshed")
		}
		reader, err := OpenReadOnly(live)
		testcheck.NoError(t, err)
		defer reader.Close()
		if head, _ := reader.Head(); head != original {
			t.Fatalf("displaced write changed live head: %s", head)
		}
	})
	t.Run("backup refreshes a stale source", func(t *testing.T) {
		root := t.TempDir()
		source, err := Create(root)
		testcheck.NoError(t, err)
		defer source.Close()
		writer, err := Create(root)
		testcheck.NoError(t, err)
		defer writer.Close()
		head, err := writer.Commit(t.Context(), fixtureBatch(t))
		testcheck.NoError(t, err)
		destination := filepath.Join(t.TempDir(), "backup")
		backupReport, err := source.Backup(t.Context(), destination)
		got, sequence := backupReport.Head, backupReport.Sequence
		if err != nil || got != head || sequence != 1 {
			t.Fatalf("backup: %s@%d, %v", got, sequence, err)
		}
	})
	t.Run("interrupted rotation refuses mismatched bytes", func(t *testing.T) {
		root := t.TempDir()
		store, err := Create(root)
		testcheck.NoError(t, err)
		defer store.Close()
		if _, err := store.Commit(t.Context(), fixtureBatch(t)); err != nil {
			t.Fatal(err)
		}
		active, err := os.ReadFile(filepath.Join(root, storeFilename))
		testcheck.NoError(t, err)
		if _, err := store.Snapshot(t.Context()); err != nil {
			t.Fatal(err)
		}
		active[len(active)-1] ^= 1
		testcheck.NoError(t, os.WriteFile(filepath.Join(root, storeFilename), active, storeFileMode))
		if recovered, err := Create(root); err == nil {
			recovered.Close()
			t.Fatal("mismatched active segment was accepted")
		}
		got, err := os.ReadFile(filepath.Join(root, storeFilename))
		if err != nil || !bytes.Equal(got, active) {
			t.Fatalf("refusal modified uncertain bytes: %v", err)
		}
	})
	t.Run("measure cold replay and resumed publication", func(t *testing.T) {
		root := t.TempDir()
		store, err := Create(root)
		testcheck.NoError(t, err)
		defer store.Close()
		for index := range fixtureWorkerCount {
			descriptor := fixtureDescriptor(t, artifact.KindOutput, fmt.Sprintf("timing/%d", index))
			if _, err := store.Commit(t.Context(), artifact.Batch{Key: fmt.Sprintf("timing/%d", index), Artifacts: []artifact.Descriptor{descriptor}}); err != nil {
				t.Fatal(err)
			}
		}
		started := time.Now()
		cold, err := Create(root)
		testcheck.NoError(t, err)
		coldWall := time.Since(started)
		defer cold.Close()
		started = time.Now()
		lock, err := processlock.AcquireContext(t.Context(), writerLockPath(root), storeFileMode)
		testcheck.NoError(t, err)
		lockWall := time.Since(started)
		testcheck.NoError(t, lock.Close())
		started = time.Now()
		testcheck.NoError(t, cold.Refresh(t.Context()))
		refreshWall := time.Since(started)
		descriptor := fixtureDescriptor(t, artifact.KindOutput, "timing/publication")
		started = time.Now()
		if _, err := cold.Commit(t.Context(), artifact.Batch{Key: "timing/publication", Artifacts: []artifact.Descriptor{descriptor}}); err != nil {
			t.Fatal(err)
		}
		publishWall := time.Since(started)
		testcheck.NoError(t, store.Refresh(t.Context()))
		if head, sequence := store.Head(); !head.Valid() || sequence != fixtureWorkerCount+1 {
			t.Fatalf("timing probe lost publication: %s@%d", head, sequence)
		}
		t.Logf("fixture_commits=%d cold_open_replay=%s uncontended_lock=%s resumed_refresh_including_lock=%s publication_including_refresh_lock_sync=%s; local fixture timings, not model-validation throughput", fixtureWorkerCount, coldWall, lockWall, refreshWall, publishWall)
	})
	t.Run("independent processes preserve writes and reject competing aliases", testTransactionProcesses)
	t.Run("rotation refreshes idle handles even when active size matches", func(t *testing.T) {
		root := t.TempDir()
		writer, err := Create(root)
		testcheck.NoError(t, err)
		defer writer.Close()
		if _, err := writer.Commit(t.Context(), fixtureBatch(t)); err != nil {
			t.Fatal(err)
		}
		if _, err := writer.Snapshot(t.Context()); err != nil {
			t.Fatal(err)
		}
		idle, err := Create(root)
		testcheck.NoError(t, err)
		defer idle.Close()
		reader, err := OpenReadOnly(root)
		testcheck.NoError(t, err)
		defer reader.Close()
		descriptor := fixtureDescriptor(t, artifact.KindOutput, "rotated")
		if _, err := writer.Commit(t.Context(), artifact.Batch{Key: "rotation", Artifacts: []artifact.Descriptor{descriptor}}); err != nil {
			t.Fatal(err)
		}
		if _, err := writer.Snapshot(t.Context()); err != nil {
			t.Fatal(err)
		}
		for _, view := range []*Store{idle, reader} {
			testcheck.NoError(t, view.Refresh(t.Context()))
			if _, found, err := view.Artifact(t.Context(), descriptor.ID); err != nil || !found {
				t.Fatalf("rotated write absent: %v, %v", found, err)
			}
		}
		if _, err := idle.Commit(t.Context(), artifact.Batch{Key: "after-rotation", Artifacts: []artifact.Descriptor{fixtureDescriptor(t, artifact.KindOutput, "after rotation")}}); err != nil {
			t.Fatal(err)
		}
		if _, err := idle.Snapshot(t.Context()); err != nil {
			t.Fatal(err)
		}
	})
	t.Run("cancel queued transaction without releasing owner", func(t *testing.T) {
		root := t.TempDir()
		store, err := Create(root)
		testcheck.NoError(t, err)
		defer store.Close()
		lock, err := processlock.Acquire(writerLockPath(root), storeFileMode)
		testcheck.NoError(t, err)
		defer lock.Close()
		ctx, cancel := context.WithCancelCause(t.Context())
		done := make(chan error, 1)
		batch := fixtureBatch(t)
		// The transaction queues behind the live OS owner; a local read must
		// answer beside it, and the cancelled transaction drains its wait.
		go func() { _, err := store.Commit(ctx, batch); done <- err }()
		store.Head()
		cancel(context.Canceled)
		if err := <-done; !errors.Is(err, context.Canceled) {
			t.Fatalf("cancel = %v", err)
		}
		if other, err := processlock.Acquire(writerLockPath(root), storeFileMode); !errors.Is(err, processlock.ErrBusy) {
			if other != nil {
				other.Close()
			}
			t.Fatalf("cancel disturbed owner: %v", err)
		}
		testcheck.NoError(t, lock.Close())
		if _, err := store.Commit(t.Context(), batch); err != nil {
			t.Fatal(err)
		}
	})
	t.Run("release refuses an interleaved write", func(t *testing.T) {
		root := t.TempDir()
		store, err := Create(root)
		testcheck.NoError(t, err)
		defer store.Close()
		other, err := Create(root)
		testcheck.NoError(t, err)
		defer other.Close()
		first := retentionContent(t, artifact.KindOutput, map[string]any{"release": "first"})
		second := retentionContent(t, artifact.KindOutput, map[string]any{"release": "second"})
		if _, err := store.Commit(t.Context(), artifact.Batch{Key: "release/seed", Contents: []artifact.Content{first, second}}); err != nil {
			t.Fatal(err)
		}
		head, sequence := store.Head()
		writer := releaseWriter{ctx: t.Context(), store: store, head: head, sequence: sequence}
		testcheck.NoError(t, writer.commit([]artifact.ID{first.Descriptor.ID}))
		if _, err := other.Commit(t.Context(), artifact.Batch{Key: "interleaved", Artifacts: []artifact.Descriptor{fixtureDescriptor(t, artifact.KindOutput, "interleaved")}}); err != nil {
			t.Fatal(err)
		}
		if err := writer.commit([]artifact.ID{second.Descriptor.ID}); !errors.Is(err, ErrHeadConflict) {
			t.Fatalf("interleaved release = %v", err)
		}
		if has, err := store.HasContent(t.Context(), second.Descriptor.ID); err != nil || !has {
			t.Fatalf("refused release published: %v, %v", has, err)
		}
	})
	t.Run("idle handles refresh before publication", func(t *testing.T) {
		root := t.TempDir()
		first, err := Create(root)
		testcheck.NoError(t, err)
		defer first.Close()
		second, err := Create(root)
		if err != nil {
			t.Fatalf("idle handle reserves writer: %v", err)
		}
		defer second.Close()
		batch := fixtureBatch(t)
		id, err := first.Commit(t.Context(), batch)
		testcheck.NoError(t, err)
		retry, err := second.Commit(t.Context(), batch)
		if err != nil || retry != id {
			t.Fatalf("refreshed retry = %s, %v", retry, err)
		}
		descriptor := fixtureDescriptor(t, artifact.KindOutput, "independent writer")
		if _, err := second.Commit(t.Context(), artifact.Batch{Key: "second", Artifacts: []artifact.Descriptor{descriptor}}); err != nil {
			t.Fatal(err)
		}
		if _, err := first.Commit(t.Context(), artifact.Batch{Key: "stale", ExpectedHead: new(id), Artifacts: []artifact.Descriptor{descriptor}}); !errors.Is(err, ErrHeadConflict) {
			t.Fatalf("stale head = %v", err)
		}
		testcheck.NoError(t, first.Refresh(t.Context()))
		if _, found, err := first.Artifact(t.Context(), descriptor.ID); err != nil || !found {
			t.Fatalf("independent write lost: %v, %v", found, err)
		}
		if _, seq := first.Head(); seq != 2 {
			t.Fatalf("sequence = %d", seq)
		}
	})
}

// fileBoundary observes a created file at an existing cancellation check.
type fileBoundary struct {
	context.Context
	path  string
	check func()
}

func (c *fileBoundary) Err() error {
	if _, err := os.Stat(c.path); err == nil {
		c.check()
	}
	return c.Context.Err()
}

type transactionRequest struct{ Action, Key string }
type transactionReply struct {
	Result   string
	Sequence uint64
}

func TestTransactionWriterProcess(t *testing.T) {
	t.Parallel()
	root := os.Getenv("OVERGO_TRANSACTION_TEST_ROOT")
	if root == "" {
		return
	}
	store, err := Create(root)
	testcheck.NoError(t, err)
	encoder, decoder := json.NewEncoder(os.Stdout), json.NewDecoder(os.Stdin)
	testcheck.NoError(t, encoder.Encode(transactionReply{Result: "ready"}))
	for {
		var request transactionRequest
		if err := decoder.Decode(&request); errors.Is(err, io.EOF) {
			break
		} else if err != nil {
			t.Fatal(err)
		}
		if request.Action == "tear" {
			lock, err := processlock.AcquireContext(t.Context(), writerLockPath(root), storeFileMode)
			testcheck.NoError(t, err)
			defer lock.Close()
			file, err := os.OpenFile(filepath.Join(root, storeFilename), os.O_WRONLY|os.O_APPEND, storeFileMode)
			testcheck.NoError(t, err)
			if _, err := file.Write([]byte("torn")); err != nil {
				t.Fatal(err)
			}
			testcheck.NoError(t, file.Sync())
			testcheck.NoError(t, encoder.Encode(transactionReply{Result: "locked-torn-tail"}))
			// Parent kills this process while the OS lock and torn tail are live.
			var next transactionRequest
			_ = decoder.Decode(&next)
			t.Fatal("crash helper resumed")
		}
		descriptor := fixtureDescriptor(t, artifact.KindOutput, request.Key)
		batch := artifact.Batch{Key: request.Key, Artifacts: []artifact.Descriptor{descriptor}}
		if request.Action == "cas" {
			batch.Aliases = []artifact.AliasBinding{{Name: "contended", Target: descriptor.ID}}
		}
		_, err := store.Commit(t.Context(), batch)
		result := "committed"
		if errors.Is(err, ErrAliasConflict) {
			result = "conflict"
		} else if err != nil {
			t.Fatal(err)
		}
		_, sequence := store.Head()
		testcheck.NoError(t, encoder.Encode(transactionReply{Result: result, Sequence: sequence}))
	}
	testcheck.NoError(t, store.Close())
	os.Exit(0)
}

func testTransactionProcesses(t *testing.T) {
	root := t.TempDir()
	store, err := Create(root)
	testcheck.NoError(t, err)
	defer store.Close()
	executable, err := os.Executable()
	testcheck.NoError(t, err)
	type peer struct {
		cmd    *exec.Cmd
		input  *json.Encoder
		output *json.Decoder
	}
	start := func() peer {
		cmd := exec.CommandContext(t.Context(), executable, "-test.run=^TestTransactionWriterProcess$", "-test.timeout=30s")
		cmd.Env = append(os.Environ(), "OVERGO_TRANSACTION_TEST_ROOT="+root)
		input, err := cmd.StdinPipe()
		testcheck.NoError(t, err)
		output, err := cmd.StdoutPipe()
		testcheck.NoError(t, err)
		cmd.Stderr = os.Stderr
		testcheck.NoError(t, cmd.Start())
		t.Cleanup(func() { _ = input.Close(); _ = cmd.Process.Kill(); _ = cmd.Wait() })
		peer := peer{cmd, json.NewEncoder(input), json.NewDecoder(output)}
		var reply transactionReply
		if err := peer.output.Decode(&reply); err != nil || reply.Result != "ready" {
			t.Fatalf("ready: %+v, %v", reply, err)
		}
		return peer
	}
	first, second := start(), start()
	read := func(peer peer) transactionReply {
		var reply transactionReply
		testcheck.NoError(t, peer.output.Decode(&reply))
		return reply
	}
	send := func(peer peer, action, key string) {
		testcheck.NoError(t, peer.input.Encode(transactionRequest{action, key}))
	}
	for index, peer := range []peer{first, second} {
		send(peer, "write", fmt.Sprintf("independent/%d", index))
	}
	for _, peer := range []peer{first, second} {
		if reply := read(peer); reply.Result != "committed" {
			t.Fatal(reply)
		}
	}
	for index, peer := range []peer{first, second} {
		send(peer, "cas", fmt.Sprintf("competing/%d", index))
	}
	results := map[string]int{}
	for _, peer := range []peer{first, second} {
		results[read(peer).Result]++
	}
	if results["committed"] != 1 || results["conflict"] != 1 {
		t.Fatalf("CAS = %v", results)
	}
	send(first, "tear", "")
	if reply := read(first); reply.Result != "locked-torn-tail" {
		t.Fatal(reply)
	}
	if lock, err := processlock.Acquire(writerLockPath(root), storeFileMode); !errors.Is(err, processlock.ErrBusy) {
		if lock != nil {
			lock.Close()
		}
		t.Fatalf("live writer exclusion: %v", err)
	}
	testcheck.NoError(t, first.cmd.Process.Kill())
	_ = first.cmd.Wait()
	send(second, "write", "after-crash")
	if reply := read(second); reply.Result != "committed" || reply.Sequence != 4 {
		t.Fatalf("recovery = %+v", reply)
	}
	testcheck.NoError(t, store.Refresh(t.Context()))
	for _, key := range []string{"independent/0", "independent/1", "after-crash"} {
		id := fixtureDescriptor(t, artifact.KindOutput, key).ID
		if _, found, err := store.Artifact(t.Context(), id); err != nil || !found {
			t.Fatalf("lost %s: %v, %v", key, found, err)
		}
	}
}
