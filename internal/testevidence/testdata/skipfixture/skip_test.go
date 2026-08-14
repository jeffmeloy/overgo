package skipfixture

import "testing"

func TestUnavailableCapability(t *testing.T) {
	t.Skip("fixture unavailable")
}
