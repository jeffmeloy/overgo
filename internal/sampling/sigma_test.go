package sampling

import "testing"

func TestValidateSigmaSchedule(t *testing.T) {
	if err := ValidateSigmaSchedule([]float32{0.9, 0.5, 0}); err != nil {
		t.Fatal(err)
	}
	if err := ValidateSigmaSchedule([]float32{0, 0}); err == nil {
		t.Fatal("accepted early zero sigma")
	}
}
