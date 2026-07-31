package inference

import "testing"

func TestStopSequenceValidationAndMatching(t *testing.T) {
	if err := validateStopSequences([]string{"END", "stop"}); err != nil {
		t.Fatal(err)
	}
	if err := validateStopSequences([]string{""}); err == nil {
		t.Fatal("empty stop sequence was accepted")
	}
	if !matchesStopSequence("prefix END suffix", []string{"END"}) {
		t.Fatal("stop sequence was not matched")
	}
	if matchesStopSequence("prefix EN", []string{"END"}) {
		t.Fatal("partial stop sequence matched")
	}
}
