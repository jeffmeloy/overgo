package plan

import (
	"encoding/json"
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
			if _, err := NewBatchAccumulator(invalid); err == nil {
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
			for repeat := range 2 {
				got, reason := flush.Decide(test.state)
				if got != test.flush || reason != test.reason {
					t.Fatalf("decide repeat %d = %v %q, want %v %q", repeat, got, reason, test.flush, test.reason)
				}
			}
		})
	}

	accumulator, err := NewBatchAccumulator(flush)
	if err != nil {
		t.Fatal(err)
	}
	start := time.Date(2026, 9, 6, 0, 0, 0, 0, time.UTC)
	if _, _, _, err := accumulator.Add("tenant", 1, start); err == nil {
		t.Fatal("accumulator accepted a member of an undeclared key")
	}
	for index, want := range []struct {
		flush  bool
		reason FlushReason
	}{{false, FlushNone}, {false, FlushNone}, {true, FlushSize}} {
		state, got, reason, err := accumulator.Add("model", 1, start.Add(time.Duration(index)*time.Millisecond))
		if err != nil || got != want.flush || reason != want.reason || state.Size != index+1 {
			t.Fatalf("add %d = %+v %v %q %v, want %v %q", index, state, got, reason, err, want.flush, want.reason)
		}
	}
	if pending := accumulator.Pending("model"); pending != 0 {
		t.Fatalf("pending after size flush = %d, want 0", pending)
	}
	if _, got, reason, err := accumulator.Add("model", 1, start.Add(time.Second)); err != nil || got || reason != FlushNone {
		t.Fatalf("first member after flush = %v %q %v, want no flush", got, reason, err)
	}
	if _, got, reason, err := accumulator.Add("model", 1, start.Add(time.Second+200*time.Millisecond)); err != nil || !got || reason != FlushInterval {
		t.Fatalf("interval member = %v %q %v, want interval flush", got, reason, err)
	}
	if _, got, reason, err := accumulator.Add("model", 4<<20, start.Add(2*time.Second)); err != nil || !got || reason != FlushBytes {
		t.Fatalf("bytes member = %v %q %v, want bytes flush", got, reason, err)
	}
}
