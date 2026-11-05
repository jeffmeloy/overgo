package sampling

import (
	"testing"

	"overgo/internal/testcheck"
)

func TestValidateSigmaSchedule(t *testing.T) {
	testcheck.NoError(t, ValidateSigmaSchedule([]float32{0.9, 0.5, 0}))
	if err := ValidateSigmaSchedule([]float32{0, 0}); err == nil {
		t.Fatal("accepted early zero sigma")
	}
}
