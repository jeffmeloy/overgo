package gate

import (
	"errors"
	"fmt"
	"testing"

	"overgo/internal/closurescan"
)

// TestClosureRemediationReadsTypedFailures holds the closure rebind to the
// failure kinds closurescan declares: a stale binding and an uncatalogued
// site are remediable wherever they sit in the error chain, and a plain
// error that merely says the same words is not.
func TestClosureRemediationReadsTypedFailures(t *testing.T) {
	t.Parallel()
	for name, remediable := range map[string]error{
		"stale binding":       fmt.Errorf("magics: %w", fmt.Errorf("permanent authority: %w literal orphan.go:x", closurescan.ErrStaleBinding)),
		"uncatalogued site":   errors.Join(errors.New("other"), &closurescan.UncataloguedPolicyError{}),
		"words without kind":  errors.New("permanent authority: stale active binding literal orphan.go:x"),
		"unrelated authority": errors.New("magic scan: active document identity mismatch"),
	} {
		want := name == "stale binding" || name == "uncatalogued site"
		if got := staleClosureAuthorityFailure(remediable); got != want {
			t.Fatalf("%s: remediable = %t, want %t", name, got, want)
		}
	}
}
