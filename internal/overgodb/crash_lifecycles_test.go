//go:build crashpoints

package overgodb

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"maps"
	"math/rand/v2"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"sync"
	"testing"

	"overgo/internal/artifact"
)

// Generated lifecycles, run by TestGeneratedStoreLifecycles: seeded sequences of commits, seals, closes, opens, reads,
// releases, packs, repeats and stale heads across two writers and a reader, every handle held to a small reference of
// what the store holds, and a sample of the states a kill leaves in each sequence held to the store's promises.

// A lifecycle run's shape: its seeds, and the long run's, which a full run's checks take (TestLongStoreLifecycles),
// the steps of each, the kill states sampled from each, the pack's limit, a tail a closing writer may leave, and the
// steps a finding quotes. Of the commits, every crashBlobEvery-th at crashBlobTurn holds a blob under an alias, and at
// crashOutputTurn an output nothing reaches.
const (
	crashSeeds      = 40
	crashLongSeeds  = 1500
	crashSteps      = 60
	crashSamples    = 6
	crashPackLimit  = 1 << 20
	crashBlobEvery  = 5
	crashBlobTurn   = 1
	crashOutputTurn = 3
	crashFewTail    = 4
	crashTraceSteps = 14
)

// crashCloseTails are the tails a closing writer may leave: none, a few records, and any.
var crashCloseTails = []uint64{crashCloseEveryTail, crashFewTail, ^uint64(0)}

// crashReference is what every handle of a lifecycle must agree with: the batches acknowledged and their commits, the
// head, and each content committed, with those a release took.
type crashReference struct {
	batches  []artifact.Batch
	ids      []artifact.CommitID
	head     artifact.CommitID
	sequence uint64
	contents map[artifact.ID][]byte
	released map[artifact.ID]bool
}

// crashGeneratedBatch is a lifecycle's commit index: inline content under an alias and, in turn, a blob under an alias
// or an output nothing reaches, each of a size of its own, so an index that names another pack's places reads the
// wrong bytes or none.
func crashGeneratedBatch(t testing.TB, index int) artifact.Batch {
	alias := fmt.Sprintf("lifecycle/%d", index)
	inline := crashContent(t, artifact.KindModel, fmt.Appendf(nil, "lifecycle record %d", index))
	batch := artifact.Batch{
		Key:      fmt.Sprintf("lifecycle/commit/%d", index),
		Contents: []artifact.Content{inline},
		Aliases:  []artifact.AliasBinding{{Name: alias, Target: inline.Descriptor.ID}},
	}
	payload := bytes.Repeat(fmt.Appendf(nil, "%d;", index), inlineContentLimit)[:inlineContentLimit+index]
	switch index % crashBlobEvery {
	case crashBlobTurn:
		blob := crashContent(t, artifact.KindTensorSet, payload)
		batch.Contents = append(batch.Contents, blob)
		batch.Aliases = append(batch.Aliases, artifact.AliasBinding{Name: alias + "/blob", Target: blob.Descriptor.ID})
	case crashOutputTurn:
		batch.Contents = append(batch.Contents, crashContent(t, artifact.KindOutput, payload))
	}
	return batch
}

// crashSequence is one lifecycle: its handles, the reference they are held to, the ledger and recorder its kill states
// are checked by, the steps it took, and what it found.
type crashSequence struct {
	t        testing.TB
	ctx      context.Context
	root     string
	reader   *Store
	writers  [2]*Store
	ref      crashReference
	ledger   *crashLedger
	recorder *crashRecorder
	next     int
	trace    []string
	failures []string
}

func (s *crashSequence) fail(format string, arguments ...any) {
	s.failures = append(s.failures, fmt.Sprintf(format, arguments...))
}

// writer is one of the two writers, drawn at random, opened if it is closed.
func (s *crashSequence) writer(rng *rand.Rand) *Store {
	slot := rng.IntN(len(s.writers))
	if s.writers[slot] == nil {
		s.trace = append(s.trace, fmt.Sprintf("open writer %d", slot))
		store, err := Create(s.root)
		if err != nil {
			s.fail("open writer %d: %v", slot, err)
			return nil
		}
		s.writers[slot] = store
	}
	return s.writers[slot]
}

func (s *crashSequence) commit(rng *rand.Rand) {
	store := s.writer(rng)
	if store == nil {
		return
	}
	s.next++
	batch := crashGeneratedBatch(s.t, s.next)
	s.trace = append(s.trace, "commit "+batch.Key)
	id, err := store.Commit(s.ctx, batch)
	if err != nil {
		s.fail("commit %s: %v", batch.Key, err)
		return
	}
	head, sequence := store.Head()
	if sequence != s.ref.sequence+1 || head != id {
		s.fail("commit %s took sequence %d after %d", batch.Key, sequence, s.ref.sequence)
	}
	s.ref.batches, s.ref.ids = append(s.ref.batches, batch), append(s.ref.ids, id)
	s.ref.head, s.ref.sequence = head, sequence
	for _, content := range batch.Contents {
		s.ref.contents[content.Descriptor.ID] = content.Data
		delete(s.ref.released, content.Descriptor.ID)
	}
	s.ledger.mu.Lock()
	s.ledger.commits = append(s.ledger.commits, crashCommit{key: batch.Key, batch: batch, id: id, sequence: sequence})
	s.ledger.heads[sequence] = head
	s.ledger.mu.Unlock()
	s.recorder.ack()
}

// verify holds one handle to the reference: its head after a refresh, and every content's bytes, or its absence once
// released.
func (s *crashSequence) verify(store *Store, name string) {
	if err := store.Refresh(s.ctx); err != nil {
		s.fail("%s refresh: %v", name, err)
		return
	}
	if head, sequence := store.Head(); sequence != s.ref.sequence || head != s.ref.head {
		s.fail("%s at sequence %d, reference %d", name, sequence, s.ref.sequence)
		return
	}
	for _, id := range slices.SortedFunc(maps.Keys(s.ref.contents), artifact.CompareID) {
		_, source, found, err := store.OpenContent(s.ctx, id)
		if err != nil {
			s.fail("%s content %s: %v", name, id, err)
			continue
		}
		if s.ref.released[id] {
			if found {
				s.fail("%s reads released content %s", name, id)
			}
			continue
		}
		if !found {
			s.fail("%s lost content %s", name, id)
			continue
		}
		read, err := io.ReadAll(source)
		if closer, ok := source.(io.Closer); ok {
			_ = closer.Close()
		}
		if err != nil || !bytes.Equal(read, s.ref.contents[id]) {
			s.fail("%s content %s differs: %d bytes of %d, %v", name, id, len(read), len(s.ref.contents[id]), err)
		}
	}
}

func (s *crashSequence) snapshot(rng *rand.Rand) {
	if store := s.writer(rng); store != nil && s.ref.sequence != 0 {
		s.trace = append(s.trace, "snapshot")
		if _, err := store.Snapshot(s.ctx); err != nil {
			s.fail("snapshot: %v", err)
		}
	}
}

func (s *crashSequence) closeWriter(rng *rand.Rand) {
	slot := rng.IntN(len(s.writers))
	if s.writers[slot] == nil {
		return
	}
	tail := crashCloseTails[rng.IntN(len(crashCloseTails))]
	s.trace = append(s.trace, fmt.Sprintf("close writer %d tail %d", slot, tail))
	s.writers[slot].tail = tail
	if err := s.writers[slot].Close(); err != nil {
		s.fail("close: %v", err)
	}
	s.writers[slot] = nil
}

// toggleReader opens the reader once the store has a journal, or closes it.
func (s *crashSequence) toggleReader(*rand.Rand) {
	if s.reader != nil {
		s.trace = append(s.trace, "close reader")
		if err := s.reader.Close(); err != nil {
			s.fail("close reader: %v", err)
		}
		s.reader = nil
		return
	}
	if _, err := os.Stat(filepath.Join(s.root, storeFilename)); err != nil {
		return
	}
	s.trace = append(s.trace, "open reader")
	reader, err := OpenReadOnly(s.root)
	if err != nil {
		s.fail("open reader: %v", err)
		return
	}
	s.reader = reader
}

func (s *crashSequence) verifyReader(*rand.Rand) {
	if s.reader != nil {
		s.trace = append(s.trace, "verify reader")
		s.verify(s.reader, "reader")
	}
}

func (s *crashSequence) verifyWriter(rng *rand.Rand) {
	if store := s.writer(rng); store != nil {
		s.trace = append(s.trace, "verify writer")
		s.verify(store, "writer")
	}
}

func (s *crashSequence) release(rng *rand.Rand) {
	store := s.writer(rng)
	if store == nil || s.ref.sequence == 0 {
		return
	}
	s.trace = append(s.trace, "release")
	report, err := Release(s.ctx, store, nil, RetentionPolicy{}, crashReleasable)
	if err != nil {
		s.fail("release: %v", err)
		return
	}
	for id := range s.ref.contents {
		if crashReleasable(artifact.Descriptor{ID: id}) {
			s.ref.released[id] = true
		}
	}
	s.ref.head, s.ref.sequence = report.HeadAfter, report.SequenceAfter
	s.ledger.head(report.HeadAfter, report.SequenceAfter)
}

func (s *crashSequence) pack(rng *rand.Rand) {
	if store := s.writer(rng); store != nil {
		s.trace = append(s.trace, "pack")
		if _, err := packBlobsUnder(s.ctx, store, crashPackLimit); err != nil {
			s.fail("pack: %v", err)
		}
	}
}

// repeat commits an acknowledged batch again, which must give back its commit and move nothing.
func (s *crashSequence) repeat(rng *rand.Rand) {
	store := s.writer(rng)
	if store == nil || len(s.ref.batches) == 0 {
		return
	}
	index := rng.IntN(len(s.ref.batches))
	s.trace = append(s.trace, "repeat "+s.ref.batches[index].Key)
	if id, err := store.Commit(s.ctx, s.ref.batches[index]); err != nil || id != s.ref.ids[index] {
		s.fail("repeat of %s: %s, %v", s.ref.batches[index].Key, id, err)
	}
	if _, sequence := store.Head(); sequence != s.ref.sequence {
		s.fail("a repeat moved the sequence to %d", sequence)
	}
}

// staleHead commits a new batch that expects the head before last, which must be refused.
func (s *crashSequence) staleHead(rng *rand.Rand) {
	store := s.writer(rng)
	if store == nil || len(s.ref.ids) < 2 {
		return
	}
	stale := s.ref.ids[len(s.ref.ids)-2]
	if stale == s.ref.head {
		return
	}
	s.next++
	batch := crashGeneratedBatch(s.t, s.next)
	batch.ExpectedHead = &stale
	s.trace = append(s.trace, "stale head")
	if _, err := store.Commit(s.ctx, batch); !errors.Is(err, ErrHeadConflict) {
		s.fail("a stale head was not refused: %v", err)
	}
}

// crashActions are a lifecycle's steps, each drawn as often as it is listed.
var crashActions = []func(*crashSequence, *rand.Rand){
	(*crashSequence).commit, (*crashSequence).commit, (*crashSequence).commit,
	(*crashSequence).commit, (*crashSequence).commit, (*crashSequence).commit,
	(*crashSequence).snapshot, (*crashSequence).closeWriter, (*crashSequence).toggleReader,
	(*crashSequence).verifyReader, (*crashSequence).verifyReader, (*crashSequence).verifyWriter,
	(*crashSequence).release, (*crashSequence).pack, (*crashSequence).repeat, (*crashSequence).staleHead,
}

// crashRunSequence drives seed's lifecycle under directory, then checks a sample of the states a kill leaves in it,
// and says what it found.
func crashRunSequence(t testing.TB, ctx context.Context, directory string, seed uint64) []string {
	defer os.RemoveAll(directory)
	root := filepath.Join(directory, "store")
	recorder := newCrashRecorder(root)
	detach := crashAttach(recorder)
	sequence := &crashSequence{t: t, ctx: ctx, root: root, ledger: newCrashLedger(), recorder: recorder,
		ref: crashReference{contents: map[artifact.ID][]byte{}, released: map[artifact.ID]bool{}}}
	rng := rand.New(rand.NewPCG(seed, seed))
	func() {
		defer func() {
			if failure := recover(); failure != nil {
				sequence.fail("panic: %v", failure)
			}
		}()
		for range crashSteps {
			crashActions[rng.IntN(len(crashActions))](sequence, rng)
			if len(sequence.failures) != 0 {
				return
			}
		}
		for slot, writer := range sequence.writers {
			if writer != nil {
				sequence.verify(writer, fmt.Sprintf("writer %d at the end", slot))
			}
		}
		if sequence.reader != nil {
			sequence.verify(sequence.reader, "reader at the end")
		}
	}()
	for _, store := range append(sequence.writers[:], sequence.reader) {
		if store != nil {
			_ = store.Close()
		}
	}
	detach()
	if len(sequence.failures) != 0 {
		recent := sequence.trace[max(0, len(sequence.trace)-crashTraceSteps):]
		return []string{fmt.Sprintf("seed %d: %s after [%s]", seed, crashShort(sequence.failures[0]), strings.Join(recent, "; "))}
	}
	states := crashKillStates(recorder.ops, recorder.acked)
	rng.Shuffle(len(states), func(i, j int) { states[i], states[j] = states[j], states[i] })
	var lines []string
	for index, state := range states[:min(crashSamples, len(states))] {
		for _, finding := range crashCheck(t, ctx, filepath.Join(directory, "crash", strconv.Itoa(index)), state, sequence.ledger) {
			lines = append(lines, fmt.Sprintf("seed %d crash at %d (%s): %s: %s", seed, state.position, state.what, finding.invariant, crashShort(strings.SplitN(finding.detail, "\n", 2)[0])))
		}
	}
	return lines
}

// TestLifecyclesKeepTheStoresPromises drives crashSeeds generated lifecycles, which a commit's checks take.
func TestLifecyclesKeepTheStoresPromises(t *testing.T) {
	// Serial: its workers share the process's recorder registry, and it runs alone under the crashpoints tag.
	crashLifecycles(t, crashSeeds)
}

// TestLongLifecyclesKeepTheStoresPromises drives crashLongSeeds generated lifecycles, which a full run's checks take.
func TestLongLifecyclesKeepTheStoresPromises(t *testing.T) {
	// Serial: its workers share the process's recorder registry, and it runs alone under the crashpoints tag.
	crashLifecycles(t, crashLongSeeds)
}

// crashLifecycles drives seeds generated lifecycles on crashWorkers at once, and holds each to the store's promises.
func crashLifecycles(t *testing.T, seeds int) {
	ctx := t.Context()
	base := crashTemp(t)
	work := make(chan uint64)
	var mu sync.Mutex
	var lines []string
	var workers sync.WaitGroup
	for range crashWorkers {
		workers.Go(func() {
			for seed := range work {
				found := crashRunSequence(t, ctx, filepath.Join(base, strconv.FormatUint(seed, 10)), seed)
				mu.Lock()
				lines = append(lines, found...)
				mu.Unlock()
			}
		})
	}
	for seed := range uint64(seeds) {
		work <- seed + 1
	}
	close(work)
	workers.Wait()
	slices.Sort(lines)
	t.Logf("%d lifecycles of %d steps, %d kill states sampled from each, %d findings", seeds, crashSteps, crashSamples, len(lines))
	for _, line := range lines {
		t.Log(line)
	}
	if len(lines) != 0 {
		t.Errorf("%d lifecycles broke a promise", len(lines))
	}
}
