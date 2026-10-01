package trainingworkflow

import (
	"bytes"
	"encoding/json"
	"fmt"
	"slices"
	"strings"
	"testing"
)

// TestUnpairedDatasetReadsLabelledCompletions reads twelve labelled
// completions, three of them desirable, as KTO microbatches: each holds the
// declared number of records, each prompt keeps its tokens outside the
// completion mask, and the class counts the run weighs by are the
// dataset's. A dataset of one record makes no microbatch; a record without
// its label, or whose prompt's tokens are no prefix of the whole, is
// refused.
func TestUnpairedDatasetReadsLabelledCompletions(t *testing.T) {
	// Each letter is its token, except that "pq" merges into one.
	encode := func(text string) ([]int, error) {
		var tokens []int
		for index := 0; index < len(text); index++ {
			if strings.HasPrefix(text[index:], "pq") {
				tokens, index = append(tokens, 0), index+1
				continue
			}
			tokens = append(tokens, int(text[index]))
		}
		return tokens, nil
	}
	lines := func(documents ...map[string]any) []byte {
		var raw bytes.Buffer
		for _, document := range documents {
			line, err := json.Marshal(document)
			if err != nil {
				t.Fatal(err)
			}
			raw.Write(append(line, '\n'))
		}
		return raw.Bytes()
	}
	const records, desirableEvery = 12, 4
	var documents []map[string]any
	for index := range records {
		documents = append(documents, map[string]any{
			"id": fmt.Sprintf("record-%02d", index), "group": "group", "prompt": strings.Repeat("a", index+1), "completion": "bc",
			"label": index%desirableEvery == 0,
		})
	}
	read, err := unpairedBatchesResume(t.Context(), lines(documents...), 0, encode, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if read.units != 2 || len(read.batches) != read.units || read.desirable != 3 || read.undesirable != 9 || read.stream.Position != 2*ktoMicrobatchExamples {
		t.Fatalf("read %d units, %d batches, %d desirable and %d undesirable to position %d", read.units, len(read.batches), read.desirable, read.undesirable, read.stream.Position)
	}
	labels := map[string]bool{}
	for _, document := range documents {
		labels[document["id"].(string)] = document["label"].(bool)
	}
	for _, batch := range read.batches {
		if len(batch.Examples) != ktoMicrobatchExamples {
			t.Fatalf("a microbatch holds %d completions, want %d", len(batch.Examples), ktoMicrobatchExamples)
		}
		for _, example := range batch.Examples {
			prompt := slices.Index(example.Sequence.Completion, true)
			if prompt <= 0 || !slices.Equal(example.Sequence.Tokens[:prompt], slices.Repeat([]int{'a'}, prompt)) ||
				!slices.Equal(example.Sequence.Tokens[prompt:], []int{'b', 'c'}) || example.Desirable != labels[example.ID] {
				t.Fatalf("example %s = %+v, want a prompt of a's before the completion bc, desirable %t", example.ID, example, labels[example.ID])
			}
		}
	}
	for name, raw := range map[string][]byte{
		"one record": lines(documents[0]),
		"no label":   lines(documents[0], map[string]any{"id": "x", "group": "group", "prompt": "a", "completion": "b"}),
		"a merged prompt boundary": lines(documents[0], map[string]any{
			"id": "y", "group": "group", "prompt": "ap", "completion": "q", "label": true,
		}),
	} {
		if _, err := unpairedBatchesResume(t.Context(), raw, 0, encode, nil, nil); err == nil {
			t.Fatalf("read %s", name)
		}
	}
}
