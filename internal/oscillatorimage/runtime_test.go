package oscillatorimage

import (
	"testing"
)

func TestValidateRequestRejectsNegativeClass(t *testing.T) {
	if err := ValidateRequest(Request{Class: -1}); err == nil {
		t.Fatal("negative class accepted")
	}
}
