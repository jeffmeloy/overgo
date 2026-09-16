// Package testskip holds the skip reasons integration tests share. It
// imports nothing from the repository, so a test that only names a reason
// inherits no reach from the evidence parser or the process launcher.
package testskip

// ShortIntegration prefixes the skip reason of an integration test excluded
// under -short; the evidence parser classifies such skips as exclusions.
const ShortIntegration = "integration excluded by -short"

// StoreAcceptanceEnv opts into the producer-store census and guard checks.
const StoreAcceptanceEnv = "OVERGO_STORE_ACCEPTANCE"

// StoreAcceptance identifies the producer-store checks awaiting that opt-in.
const StoreAcceptance = ShortIntegration + ": store-lineage acceptance runs when " + StoreAcceptanceEnv + " is set"
