// Package testskip holds the skip reasons integration tests share. It
// imports nothing from the repository, so a test that only names a reason
// inherits no reach from the evidence parser or the process launcher.
package testskip

// ShortIntegration prefixes the skip reason of an integration test excluded
// under -short; the evidence parser classifies such skips as exclusions.
const ShortIntegration = "integration excluded by -short"

// StoreAcceptanceEnv names the variable an operator sets to run the
// acceptances bound to the producer store's lineage: the capability census
// denominator, the long-form guard cohort and the speech alignment corpus.
// The gate's store carries mirrored lane activations, so those acceptances
// hold only against the recorded disposition; the plan row
// stale-store-acceptances re-records them.
const StoreAcceptanceEnv = "OVERGO_STORE_ACCEPTANCE"

// StoreAcceptance is the skip reason when StoreAcceptanceEnv is unset.
const StoreAcceptance = ShortIntegration + ": store-lineage acceptance runs when " + StoreAcceptanceEnv + " is set"
