// Package testskip classifies integration skips where they are taken: each
// records its kind as a test attribute, which the evidence parser reads in
// place of the skip's message. It imports nothing from the repository.
package testskip

import (
	"os"
	"strings"
	"testing"
)

const (
	// Key is the attribute carrying a skip's kind.
	Key = "overgo.skip"
	// KindShort is classified in a short run only.
	KindShort = "short"
	// KindInapplicable -- another lane, an opt-in not given -- is a declared
	// exclusion in every run, earning no credit.
	KindInapplicable = "inapplicable"
	// StoreAcceptanceEnv opts into the producer-store census and guard checks.
	StoreAcceptanceEnv = "OVERGO_STORE_ACCEPTANCE"
)

// Stored receipts and harness files frozen by an acquisition record cite the
// skip messages without a kind, so the evidence parser reads them where no
// kind was recorded.
const (
	// ShortIntegration is the message of a short-mode skip.
	ShortIntegration = "integration excluded by -short"
	// Inapplicable is the message prefix of a declared exclusion.
	Inapplicable = "integration inapplicable here"
)

// Short skips an integration test under -short; the guard is the call's own,
// so no environment gate can cite it.
func Short(t testing.TB, detail ...string) {
	if testing.Short() {
		t.Helper()
		t.Attr(Key, KindShort)
		t.Skip(strings.Join(append([]string{ShortIntegration}, detail...), ": "))
	}
}

// NotApplicable skips an integration test that cannot apply where it runs.
func NotApplicable(t testing.TB, reason string) {
	t.Helper()
	t.Attr(Key, KindInapplicable)
	t.Skip(Inapplicable + ": " + reason)
}

// StoreAcceptance holds the producer-store checks until the run opts in.
func StoreAcceptance(t testing.TB) {
	if os.Getenv(StoreAcceptanceEnv) == "" {
		t.Helper()
		NotApplicable(t, "store-lineage acceptance runs when "+StoreAcceptanceEnv+" is set")
	}
}
