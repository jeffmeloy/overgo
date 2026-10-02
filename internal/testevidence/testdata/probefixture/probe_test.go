package probefixture

import (
	"testing"

	"overgo/internal/testskip"
)

func TestExisting(t *testing.T) {}

// TestUnavailableCapability passes here and skips in skipfixture, so a run of
// both shows one package's outcome lending nothing to a same-named test of
// another.
func TestUnavailableCapability(t *testing.T) {}

// TestElsewhere declares itself inapplicable: visible in a run, never credited.
func TestElsewhere(t *testing.T) { testskip.NotApplicable(t, "a fixture of another lane") }
