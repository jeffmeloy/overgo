package plan

import (
	"encoding/json"
	"math"
	"reflect"
	"strings"
	"testing"
	"time"
)

func flushFixture() BatchFlush {
	return BatchFlush{Key: "model", MaxSize: 3, MaxInterval: "200ms", MaxBytes: 4 << 20}
}

// TestBatchFlushDeclarations pins: declaration round trip inside a
// verification batch; refused declarations; size/interval/bytes bounds in
// order; key isolation; deterministic decisions.
func TestBatchFlushDeclarations(t *testing.T) {
	document := batchPlanFixture()
	flush := flushFixture()
	document.Items[0].Steps[0].VerificationBatch.Flush = &flush
	data, err := json.Marshal(document)
	if err != nil {
		t.Fatal(err)
	}
	parsed, err := Parse(data)
	if err != nil || !reflect.DeepEqual(parsed, document) {
		t.Fatalf("flush declaration round trip = %+v, %v", parsed, err)
	}
	for _, test := range []struct {
		name string
		edit func(*BatchFlush)
	}{
		{"empty key", func(f *BatchFlush) { f.Key = "" }},
		{"upper key", func(f *BatchFlush) { f.Key = "Model" }},
		{"zero size", func(f *BatchFlush) { f.MaxSize = 0 }},
		{"zero bytes", func(f *BatchFlush) { f.MaxBytes = 0 }},
		{"missing interval", func(f *BatchFlush) { f.MaxInterval = "" }},
		{"negative interval", func(f *BatchFlush) { f.MaxInterval = "-1s" }},
		{"bare number interval", func(f *BatchFlush) { f.MaxInterval = "200" }},
	} {
		t.Run(test.name, func(t *testing.T) {
			candidate := batchPlanFixture()
			invalid := flushFixture()
			test.edit(&invalid)
			candidate.Items[0].Steps[0].VerificationBatch.Flush = &invalid
			if err := Validate(candidate); err == nil || !strings.Contains(err.Error(), "batch flush") {
				t.Fatalf("validate %s = %v, want batch flush refusal", test.name, err)
			}
			if _, _, _, err := invalid.Advance(BatchState{}, nil, time.Now(), false); err == nil {
				t.Fatalf("accumulator accepted %s", test.name)
			}
		})
	}

	for _, test := range []struct {
		name   string
		state  BatchState
		flush  bool
		reason FlushReason
	}{
		{"below every bound", BatchState{Key: "model", Size: 2, Bytes: 10, Elapsed: 100 * time.Millisecond}, false, FlushNone},
		{"size bound", BatchState{Key: "model", Size: 3, Bytes: 10, Elapsed: 0}, true, FlushSize},
		{"interval bound", BatchState{Key: "model", Size: 1, Bytes: 10, Elapsed: 200 * time.Millisecond}, true, FlushInterval},
		{"bytes bound", BatchState{Key: "model", Size: 1, Bytes: 4 << 20, Elapsed: 0}, true, FlushBytes},
		{"size names reason before interval", BatchState{Key: "model", Size: 3, Bytes: 4 << 20, Elapsed: time.Second}, true, FlushSize},
		{"other key never flushes", BatchState{Key: "tenant", Size: 9, Bytes: 9 << 20, Elapsed: time.Hour}, false, FlushNone},
	} {
		t.Run(test.name, func(t *testing.T) {
			origin := time.Date(2026, 9, 7, 0, 0, 0, 0, time.UTC)
			test.state.StartedAt = origin
			for repeat := range 2 {
				_, got, reason, err := flush.Advance(test.state, nil, origin.Add(test.state.Elapsed), false)
				if err != nil && test.state.Key == flush.Key {
					t.Fatal(err)
				}
				if got != test.flush || reason != test.reason {
					t.Fatalf("decide repeat %d = %v %q, want %v %q", repeat, got, reason, test.flush, test.reason)
				}
			}
		})
	}

	start := time.Date(2026, 9, 6, 0, 0, 0, 0, time.UTC)
	bytes := int64(1)
	pending := BatchState{}
	for index, want := range []struct {
		flush  bool
		reason FlushReason
	}{{false, FlushNone}, {false, FlushNone}, {true, FlushSize}} {
		state, got, reason, err := flush.Advance(pending, &bytes, start.Add(time.Duration(index)*time.Millisecond), false)
		if err != nil || got != want.flush || reason != want.reason || state.Size != index+1 {
			t.Fatalf("add %d = %+v %v %q %v, want %v %q", index, state, got, reason, err, want.flush, want.reason)
		}
		pending = state
		if got {
			pending = BatchState{}
		}
	}
	if pending, ready, _, err := flush.Advance(pending, nil, start.Add(time.Second), true); err != nil || ready || pending.Size != 0 {
		t.Fatalf("pending after accepted size flush = %+v, %v, %v", pending, ready, err)
	}
	pending, ready, reason, err := flush.Advance(pending, &bytes, start.Add(time.Second), false)
	if err != nil || ready || reason != FlushNone {
		t.Fatalf("first member after flush = %v %q %v", ready, reason, err)
	}
	if _, ready, reason, err := flush.Advance(pending, &bytes, start.Add(time.Second+200*time.Millisecond), false); err != nil || !ready || reason != FlushInterval {
		t.Fatalf("interval member = %v %q %v", ready, reason, err)
	}
	bytes = 4 << 20
	if _, ready, reason, err := flush.Advance(BatchState{}, &bytes, start.Add(2*time.Second), false); err != nil || !ready || reason != FlushBytes {
		t.Fatalf("bytes member = %v %q %v", ready, reason, err)
	}
}

func TestBatchFlushWithoutNewMembers(t *testing.T) {
	flush := flushFixture()
	if _, ready, _, _ := flush.Advance(BatchState{Key: flush.Key, Elapsed: time.Hour}, nil, time.Now(), false); ready {
		t.Fatal("an empty batch became flushable merely by aging")
	}
	flush.MaxBytes = math.MaxInt64
	now := time.Date(2026, 9, 7, 0, 0, 0, 0, time.UTC)
	bytes := int64(math.MaxInt64 - 1)
	pending, ready, _, err := flush.Advance(BatchState{}, &bytes, now, false)
	if err != nil || ready {
		t.Fatalf("seed pending bytes: %v %v", ready, err)
	}
	bytes = 2
	if _, _, _, err := flush.Advance(pending, &bytes, now, false); err == nil {
		t.Fatal("byte overflow corrupted the pending batch")
	}
	if pending.Bytes != math.MaxInt64-1 {
		t.Fatal("overflow mutated the caller's durable state")
	}
}

func TestBatchFlushRestartAndBoundary(t *testing.T) {
	declaration := flushFixture()
	origin := time.Date(2026, 9, 7, 0, 0, 0, 0, time.UTC)
	bytes := int64(1)
	pending, ready, _, err := declaration.Advance(BatchState{}, &bytes, origin, false)
	if err != nil || ready {
		t.Fatalf("seed: %+v %v %v", pending, ready, err)
	}
	encoded, err := json.Marshal(pending)
	if err != nil {
		t.Fatal(err)
	}
	var restored BatchState
	if err := json.Unmarshal(encoded, &restored); err != nil {
		t.Fatal(err)
	}
	interval, err := declaration.Interval()
	if err != nil {
		t.Fatal(err)
	}
	state, ready, reason, err := declaration.Advance(restored, nil, origin.Add(interval), false)
	if err != nil || !ready || reason != FlushInterval || state.Size != 1 || state.Bytes != 1 || !state.StartedAt.Equal(origin) {
		t.Fatalf("elapsed-only restart: %+v %v %s %v", state, ready, reason, err)
	}
	// A failed publication leaves the original pending state available to retry.
	if restored != pending {
		t.Fatal("a flush decision consumed durable pending state")
	}
	if retry, ready, _, err := declaration.Advance(restored, nil, origin.Add(interval), true); err != nil || !ready || retry != state {
		t.Fatalf("unacknowledged flush was lost: %+v %v %v", retry, ready, err)
	}
	if state, ready, _, err := declaration.Advance(BatchState{}, nil, origin.Add(interval), true); err != nil || ready || state.Size != 0 {
		t.Fatalf("acknowledged flush replayed: %+v %v %v", state, ready, err)
	}
	for _, member := range []*int64{nil, &bytes} {
		if _, _, _, err := declaration.Advance(pending, member, origin.Add(-time.Nanosecond), true); err == nil {
			t.Fatal("accepted pre-origin event")
		}
	}
	state, ready, reason, err = declaration.Advance(pending, nil, origin, true)
	if err != nil || !ready || reason != flushBoundary || state.Size != 1 || state.Bytes != 1 {
		t.Fatalf("trailing boundary: %+v %v %s %v", state, ready, reason, err)
	}
	for _, invalid := range []BatchState{
		{Key: "other", Size: 1, StartedAt: origin},
		{Key: declaration.Key, Size: -1, StartedAt: origin},
		{Key: declaration.Key, Size: 1, Bytes: -1, StartedAt: origin},
		{Key: declaration.Key, Size: 1, Elapsed: -1, StartedAt: origin},
		{Key: declaration.Key, Size: 1},
		{Key: declaration.Key, Bytes: 1},
	} {
		if _, _, _, err := declaration.Advance(invalid, nil, origin, false); err == nil {
			t.Fatalf("restored invalid state: %+v", invalid)
		}
	}
	declaration.MaxSize = math.MaxInt
	pending = BatchState{Key: declaration.Key, Size: math.MaxInt, StartedAt: origin}
	bytes = 0
	if _, _, _, err := declaration.Advance(pending, &bytes, origin, false); err == nil {
		t.Fatal("member count overflow was accepted")
	}
	state, ready, reason, err = declaration.Advance(pending, nil, origin, false)
	if err != nil || !ready || reason != FlushSize || state.Size != math.MaxInt {
		t.Fatalf("overflow refusal lost pending state: %+v %v %s %v", state, ready, reason, err)
	}
}
