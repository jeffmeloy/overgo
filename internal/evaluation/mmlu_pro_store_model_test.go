//go:build modeltest

package evaluation

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"slices"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/dataroot"
	"overgo/internal/overgodb"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

// TestTheStoresMMLUProCasesKeepTheirIdentities holds the store's MMLU-Pro cases to the identities the MMLU-Pro kind
// bound, as TestMMLUProIsAGroupedChoiceWithItsIdentitiesUnchanged holds the fixture's: the part a measurement run
// scores, the whole, forty taken evenly and each case alone, under every protocol, and the family derived for a
// likelihood reading. It reads the store, so it runs under modeltest, where the store is admitted before a model
// loads, and never in the host gate, where an import into the store refused landings that touched nothing of MMLU-Pro
// (identity-gate/do, SQA-368).
func TestTheStoresMMLUProCasesKeepTheirIdentities(t *testing.T) {
	t.Parallel()
	protocols := mmluProProtocols()
	roots, err := dataroot.Resolve(testutil.RepoRoot(t))
	if err != nil {
		t.Skipf("no store to read MMLU-Pro cases from: %v", err)
	}
	reader, err := overgodb.OpenReadOnly(roots.Store)
	if err != nil {
		t.Skipf("the store at %s is not readable: %v", roots.Store, err)
	}
	defer reader.Close()
	records, err := storeFamilies(t.Context(), reader, "mmlu-pro")
	families, datasets := records.families, records.datasets
	if err != nil || fmt.Sprint(datasets["mmlu-pro"]) != mmluProStoreDatasets {
		t.Skipf("the store's MMLU-Pro cases are records of %v, the digests were pinned on those of %s: %v", datasets["mmlu-pro"], mmluProStoreDatasets, err)
	}
	whole := families["mmlu-pro"]
	share, spaced := slices.Clone(whole), slices.Clone(whole)
	drawShare(share, suiteShares["mmlu-pro"])
	testcheck.NoError(t, drawSpaced(spaced, 40, true))
	var sharePlan artifact.ID
	for label, protocol := range protocols {
		for name, cases := range map[string][]storeCase{"share": share, "whole": whole, "spaced": spaced} {
			got, ids := mmluProIdentities(t, cases, protocol.authorities)
			if got != protocol.store[name] {
				t.Errorf("under %s the store's %s draw binds %v, digest %s, want digest %s", label, name, ids, got, protocol.store[name])
			}
			if name == "share" && label == PromptingRawCompletion.Label() {
				sharePlan = ids[4]
			}
		}
		each := sha256.New()
		for index := range whole {
			digest, _ := mmluProIdentities(t, whole[index:index+1], protocol.authorities)
			each.Write([]byte(digest + "\n"))
		}
		if got := hex.EncodeToString(each.Sum(nil)); got != protocol.store["each"] {
			t.Errorf("under %s the store's %d cases alone digest to %s, want %s", label, len(whole), got, protocol.store["each"])
		}
	}
	// The family derived for a likelihood reading is the share given the store's validation demonstrations, the
	// share alone where the store holds none (mmlu-pro-lm-eval/five-shot).
	if shots := records.shots["mmlu-pro"]; len(shots) > 0 {
		assembled, _, err := assembleMMLUProSuite(share, shots)
		testcheck.NoError(t, err)
		compiled := testcheck.Must(CompileGroupedChoice(assembled.(GroupedChoiceSuite)))(t)
		sharePlan = testcheck.Must(BindGroupedChoice(compiled, protocols[PromptingRawCompletion.Label()].authorities))(t).Identity()
	}
	suites, _, err := DeriveStoreSuiteFamily(t.Context(), reader, protocols[PromptingRawCompletion.Label()].authorities, "mmlu-pro")
	if err != nil || len(suites) != 1 || suites[0].Plan().Identity() != sharePlan || suites[0].Descriptor().Kind != GroupedChoiceKind {
		t.Fatalf("the derived MMLU-Pro family is %d suites, the first binding another plan than %s or of another kind: %v", len(suites), sharePlan, err)
	}
}
