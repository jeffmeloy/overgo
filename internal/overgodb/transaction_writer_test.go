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
	"testing"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/processlock"
)

func TestTransactionWriter(t *testing.T) {
	t.Run("backup refreshes a stale source", func(t *testing.T) {
		root := t.TempDir()
		source, err := Open(root)
		if err != nil {
			t.Fatal(err)
		}
		defer source.Close()
		writer, err := Open(root)
		if err != nil {
			t.Fatal(err)
		}
		defer writer.Close()
		head, err := writer.Commit(t.Context(), fixtureBatch(t))
		if err != nil {
			t.Fatal(err)
		}
		destination := filepath.Join(t.TempDir(), "backup")
		got, sequence, err := source.Backup(destination)
		if err != nil || got != head || sequence != 1 {
			t.Fatalf("backup: %s@%d, %v", got, sequence, err)
		}
	})
	t.Run("interrupted rotation refuses mismatched bytes", func(t *testing.T) {
		root := t.TempDir()
		store, err := Open(root)
		if err != nil {
			t.Fatal(err)
		}
		defer store.Close()
		if _, err := store.Commit(t.Context(), fixtureBatch(t)); err != nil {
			t.Fatal(err)
		}
		active, err := os.ReadFile(filepath.Join(root, storeFilename))
		if err != nil {
			t.Fatal(err)
		}
		if _, err := store.Snapshot(t.Context()); err != nil {
			t.Fatal(err)
		}
		active[len(active)-1] ^= 1
		if err := os.WriteFile(filepath.Join(root, storeFilename), active, storeFileMode); err != nil {
			t.Fatal(err)
		}
		if recovered, err := Open(root); err == nil {
			recovered.Close()
			t.Fatal("mismatched active segment was accepted")
		}
		got, err := os.ReadFile(filepath.Join(root, storeFilename))
		if err != nil || !bytes.Equal(got, active) {
			t.Fatalf("refusal modified uncertain bytes: %v", err)
		}
	})
	t.Run("recover rotation published before active reset", func(t *testing.T) {
		root := t.TempDir()
		store, err := Open(root)
		if err != nil {
			t.Fatal(err)
		}
		defer store.Close()
		if _, err := store.Commit(t.Context(), fixtureBatch(t)); err != nil {
			t.Fatal(err)
		}
		active, err := os.ReadFile(filepath.Join(root, storeFilename))
		if err != nil {
			t.Fatal(err)
		}
		if _, err := store.Snapshot(t.Context()); err != nil {
			t.Fatal(err)
		}
		// Restore the pre-reset active bytes: the exact crash boundary between
		// durable sealed publication and active truncation, with no live owner.
		if err := os.WriteFile(filepath.Join(root, storeFilename), active, storeFileMode); err != nil {
			t.Fatal(err)
		}
		reader, err := OpenReadOnly(root)
		if err != nil {
			t.Fatalf("read interrupted rotation: %v", err)
		}
		defer reader.Close()
		var descriptors []artifact.Descriptor
		for index := range fixtureWorkerCount {
			descriptors = append(descriptors, fixtureDescriptor(t, artifact.KindOutput, fmt.Sprintf("after interrupted rotation/%d", index)))
		}
		if _, err := store.Commit(t.Context(), artifact.Batch{Key: "rotation-recovery", Artifacts: descriptors}); err != nil {
			t.Fatalf("recover interrupted rotation: %v", err)
		}
		info, err := os.Stat(filepath.Join(root, storeFilename))
		if err != nil || info.Size() <= int64(len(active)) {
			t.Fatalf("probe must refill beyond old reader offset: %v, %v", info, err)
		}
		if err := reader.Refresh(t.Context()); err != nil {
			t.Fatal(err)
		}
		if _, sequence := reader.Head(); sequence != 2 {
			t.Fatalf("rotation recovery sequence = %d", sequence)
		}
	})
	t.Run("measure cold replay and resumed publication", func(t *testing.T) {
		root := t.TempDir()
		store, err := Open(root)
		if err != nil {
			t.Fatal(err)
		}
		defer store.Close()
		for index := range fixtureWorkerCount {
			descriptor := fixtureDescriptor(t, artifact.KindOutput, fmt.Sprintf("timing/%d", index))
			if _, err := store.Commit(t.Context(), artifact.Batch{Key: fmt.Sprintf("timing/%d", index), Artifacts: []artifact.Descriptor{descriptor}}); err != nil {
				t.Fatal(err)
			}
		}
		started := time.Now()
		cold, err := Open(root)
		if err != nil {
			t.Fatal(err)
		}
		coldWall := time.Since(started)
		defer cold.Close()
		started = time.Now()
		lock, err := processlock.AcquireContext(t.Context(), filepath.Join(root, lockFilename), storeFileMode)
		if err != nil {
			t.Fatal(err)
		}
		lockWall := time.Since(started)
		if err := lock.Close(); err != nil {
			t.Fatal(err)
		}
		started = time.Now()
		if err := cold.Refresh(t.Context()); err != nil {
			t.Fatal(err)
		}
		refreshWall := time.Since(started)
		descriptor := fixtureDescriptor(t, artifact.KindOutput, "timing/publication")
		started = time.Now()
		if _, err := cold.Commit(t.Context(), artifact.Batch{Key: "timing/publication", Artifacts: []artifact.Descriptor{descriptor}}); err != nil {
			t.Fatal(err)
		}
		publishWall := time.Since(started)
		if err := store.Refresh(t.Context()); err != nil {
			t.Fatal(err)
		}
		if head, sequence := store.Head(); !head.Valid() || sequence != fixtureWorkerCount+1 {
			t.Fatalf("timing probe lost publication: %s@%d", head, sequence)
		}
		t.Logf("fixture_commits=%d cold_open_replay=%s uncontended_lock=%s resumed_refresh_including_lock=%s publication_including_refresh_lock_sync=%s; local fixture timings, not model-validation throughput", fixtureWorkerCount, coldWall, lockWall, refreshWall, publishWall)
	})
	t.Run("independent processes preserve writes and reject competing aliases", testTransactionProcesses)
	t.Run("rotation refreshes idle handles even when active size matches", func(t *testing.T) {
		root := t.TempDir()
		writer, err := Open(root)
		if err != nil {
			t.Fatal(err)
		}
		defer writer.Close()
		if _, err := writer.Commit(t.Context(), fixtureBatch(t)); err != nil {
			t.Fatal(err)
		}
		if _, err := writer.Snapshot(t.Context()); err != nil {
			t.Fatal(err)
		}
		idle, err := Open(root)
		if err != nil {
			t.Fatal(err)
		}
		defer idle.Close()
		reader, err := OpenReadOnly(root)
		if err != nil {
			t.Fatal(err)
		}
		defer reader.Close()
		descriptor := fixtureDescriptor(t, artifact.KindOutput, "rotated")
		if _, err := writer.Commit(t.Context(), artifact.Batch{Key: "rotation", Artifacts: []artifact.Descriptor{descriptor}}); err != nil {
			t.Fatal(err)
		}
		if _, err := writer.Snapshot(t.Context()); err != nil {
			t.Fatal(err)
		}
		for _, view := range []*Store{idle, reader} {
			if err := view.Refresh(t.Context()); err != nil {
				t.Fatal(err)
			}
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
		store, err := Open(root)
		if err != nil {
			t.Fatal(err)
		}
		defer store.Close()
		lock, err := processlock.Acquire(filepath.Join(root, lockFilename), storeFileMode)
		if err != nil {
			t.Fatal(err)
		}
		defer lock.Close()
		ctx, cancel := context.WithCancelCause(t.Context())
		done := make(chan error, 1)
		batch := fixtureBatch(t)
		go func() { _, err := store.Commit(ctx, batch); done <- err }()
		// Let the transaction queue behind a live OS owner, then cancel it.
		time.Sleep(20 * time.Millisecond)
		read := make(chan struct{})
		go func() { store.Head(); close(read) }()
		select {
		case <-read:
		case <-time.After(5 * time.Second):
			cancel(context.Canceled)
			t.Fatal("queued writer blocks local readers")
		}
		cancel(context.Canceled)
		select {
		case err := <-done:
			if !errors.Is(err, context.Canceled) {
				t.Fatalf("cancel = %v", err)
			}
		case <-time.After(5 * time.Second):
			t.Fatal("cancel did not drain lock wait")
		}
		if other, err := processlock.Acquire(filepath.Join(root, lockFilename), storeFileMode); !errors.Is(err, processlock.ErrBusy) {
			if other != nil {
				other.Close()
			}
			t.Fatalf("cancel disturbed owner: %v", err)
		}
		if err := lock.Close(); err != nil {
			t.Fatal(err)
		}
		if _, err := store.Commit(t.Context(), batch); err != nil {
			t.Fatal(err)
		}
	})
	t.Run("compaction refuses an interleaved destination write", func(t *testing.T) {
		root := t.TempDir()
		destination, err := Open(root)
		if err != nil {
			t.Fatal(err)
		}
		defer destination.Close()
		other, err := Open(root)
		if err != nil {
			t.Fatal(err)
		}
		defer other.Close()
		writer := compactionWriter{ctx: t.Context(), destination: destination}
		first := artifact.Batch{Artifacts: []artifact.Descriptor{fixtureDescriptor(t, artifact.KindOutput, "compaction first")}}
		if err := writer.commit([]artifact.Batch{first}); err != nil {
			t.Fatal(err)
		}
		if _, err := other.Commit(t.Context(), artifact.Batch{Key: "interleaved", Artifacts: []artifact.Descriptor{fixtureDescriptor(t, artifact.KindOutput, "interleaved")}}); err != nil {
			t.Fatal(err)
		}
		second := artifact.Batch{Artifacts: []artifact.Descriptor{fixtureDescriptor(t, artifact.KindOutput, "compaction second")}}
		if err := writer.commit([]artifact.Batch{second}); !errors.Is(err, ErrHeadConflict) {
			t.Fatalf("interleaved compaction = %v", err)
		}
		if _, found, err := destination.Artifact(t.Context(), second.Artifacts[0].ID); err != nil || found {
			t.Fatalf("refused compaction published: %v, %v", found, err)
		}
	})
	t.Run("idle handles refresh before publication", func(t *testing.T) {
		root := t.TempDir()
		first, err := Open(root)
		if err != nil {
			t.Fatal(err)
		}
		defer first.Close()
		second, err := Open(root)
		if err != nil {
			t.Fatalf("idle handle reserves writer: %v", err)
		}
		defer second.Close()
		batch := fixtureBatch(t)
		id, err := first.Commit(t.Context(), batch)
		if err != nil {
			t.Fatal(err)
		}
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
		if err := first.Refresh(t.Context()); err != nil {
			t.Fatal(err)
		}
		if _, found, err := first.Artifact(t.Context(), descriptor.ID); err != nil || !found {
			t.Fatalf("independent write lost: %v, %v", found, err)
		}
		if _, seq := first.Head(); seq != 2 {
			t.Fatalf("sequence = %d", seq)
		}
	})
}

type transactionRequest struct{ Action, Key string }
type transactionReply struct {
	Result   string
	Sequence uint64
}

func TestTransactionWriterProcess(t *testing.T) {
	root := os.Getenv("OVERGO_TRANSACTION_TEST_ROOT")
	if root == "" {
		return
	}
	store, err := Open(root)
	if err != nil {
		t.Fatal(err)
	}
	encoder, decoder := json.NewEncoder(os.Stdout), json.NewDecoder(os.Stdin)
	if err := encoder.Encode(transactionReply{Result: "ready"}); err != nil {
		t.Fatal(err)
	}
	for {
		var request transactionRequest
		if err := decoder.Decode(&request); errors.Is(err, io.EOF) {
			break
		} else if err != nil {
			t.Fatal(err)
		}
		if request.Action == "tear" {
			lock, err := processlock.AcquireContext(t.Context(), filepath.Join(root, lockFilename), storeFileMode)
			if err != nil {
				t.Fatal(err)
			}
			defer lock.Close()
			file, err := os.OpenFile(filepath.Join(root, storeFilename), os.O_WRONLY|os.O_APPEND, storeFileMode)
			if err != nil {
				t.Fatal(err)
			}
			if _, err := file.Write([]byte("torn")); err != nil {
				t.Fatal(err)
			}
			if err := file.Sync(); err != nil {
				t.Fatal(err)
			}
			if err := encoder.Encode(transactionReply{Result: "locked-torn-tail"}); err != nil {
				t.Fatal(err)
			}
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
		if err := encoder.Encode(transactionReply{Result: result, Sequence: sequence}); err != nil {
			t.Fatal(err)
		}
	}
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
	os.Exit(0)
}

func testTransactionProcesses(t *testing.T) {
	root := t.TempDir()
	store, err := Open(root)
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	executable, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	type peer struct {
		cmd    *exec.Cmd
		input  *json.Encoder
		output *json.Decoder
	}
	start := func() peer {
		cmd := exec.CommandContext(t.Context(), executable, "-test.run=^TestTransactionWriterProcess$", "-test.timeout=30s")
		cmd.Env = append(os.Environ(), "OVERGO_TRANSACTION_TEST_ROOT="+root)
		input, err := cmd.StdinPipe()
		if err != nil {
			t.Fatal(err)
		}
		output, err := cmd.StdoutPipe()
		if err != nil {
			t.Fatal(err)
		}
		cmd.Stderr = os.Stderr
		if err := cmd.Start(); err != nil {
			t.Fatal(err)
		}
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
		if err := peer.output.Decode(&reply); err != nil {
			t.Fatal(err)
		}
		return reply
	}
	send := func(peer peer, action, key string) {
		if err := peer.input.Encode(transactionRequest{action, key}); err != nil {
			t.Fatal(err)
		}
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
	if lock, err := processlock.Acquire(filepath.Join(root, lockFilename), storeFileMode); !errors.Is(err, processlock.ErrBusy) {
		if lock != nil {
			lock.Close()
		}
		t.Fatalf("live writer exclusion: %v", err)
	}
	if err := first.cmd.Process.Kill(); err != nil {
		t.Fatal(err)
	}
	_ = first.cmd.Wait()
	send(second, "write", "after-crash")
	if reply := read(second); reply.Result != "committed" || reply.Sequence != 4 {
		t.Fatalf("recovery = %+v", reply)
	}
	if err := store.Refresh(t.Context()); err != nil {
		t.Fatal(err)
	}
	for _, key := range []string{"independent/0", "independent/1", "after-crash"} {
		id := fixtureDescriptor(t, artifact.KindOutput, key).ID
		if _, found, err := store.Artifact(t.Context(), id); err != nil || !found {
			t.Fatalf("lost %s: %v, %v", key, found, err)
		}
	}
}
