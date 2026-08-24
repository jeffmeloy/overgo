package overgodbimport

import (
	"encoding/json"
	"strconv"
	"strings"
	"testing"

	"overgo/internal/model"
	"overgo/internal/modelrecipe"
	"overgo/internal/strictjson"
	"overgo/internal/tensor"
)

// renamedProfileFixture is the exact shape published under the v2 label before
// the rename: DenseStages still carries the boolean, and the policy is
// otherwise a current profile.
func renamedProfileFixture(t *testing.T, applied bool) []byte {
	t.Helper()
	// Any registered architecture exercises the upgrade: the rename is a
	// document-shape concern, not a model fact, so naming one architecture here
	// would copy a model identity into a test that does not depend on it.
	var profile model.ArchitectureProfile
	var found bool
	for _, name := range model.SupportedArchitectures() {
		if profile, found = model.LookupArchitecture(name); found {
			break
		}
	}
	if !found {
		t.Fatal("the embedded catalog registers no resolvable architecture")
	}
	encoded, err := json.Marshal(profile)
	if err != nil {
		t.Fatal(err)
	}
	var policy map[string]any
	if err := json.Unmarshal(encoded, &policy); err != nil {
		t.Fatal(err)
	}
	stages, ok := policy["DenseStages"].(map[string]any)
	if !ok {
		t.Fatalf("dense stages absent from %q", profile.Name)
	}
	delete(stages, "PostRotaryRMSExcludedExperts")
	stages["PostRotaryRMSNon128"] = applied
	// The version field is deliberately absent: the upgrade routes on the
	// document's schema label, not on a version carried inside the body, so
	// restating one here would only copy a production value into the fixture.
	document, err := json.Marshal(map[string]any{
		"architecture": profile.Name, "policy": policy,
	})
	if err != nil {
		t.Fatal(err)
	}
	return document
}

func upgradedExcludedExperts(t *testing.T, applied bool) uint32 {
	t.Helper()
	content, err := upgradeRenamedProfile(renamedProfileFixture(t, applied))
	if err != nil {
		t.Fatal(err)
	}
	var body struct {
		Version      uint16                    `json:"version"`
		Architecture string                    `json:"architecture"`
		Policy       model.ArchitectureProfile `json:"policy"`
		Provenance   []json.RawMessage         `json:"provenance,omitempty"`
	}
	if err := strictjson.DecodeBytes(content, &body); err != nil {
		t.Fatal(err)
	}
	if body.Version != modelrecipe.ProfileVersion {
		t.Fatalf("upgraded version = %d, want %d", body.Version, modelrecipe.ProfileVersion)
	}
	return body.Policy.DenseStages.PostRotaryRMSExcludedExperts
}

// TestProfileRenameUpgrade pins both directions of the boolean the v3 rename
// replaced. The pre-v3 predicate was "apply post-rotary RMS unless the expert
// count is 128", so a true boolean must upgrade to that exact count and a false
// boolean must upgrade to the disabled sentinel.
func TestProfileRenameUpgrade(t *testing.T) {
	if excluded := upgradedExcludedExperts(t, true); excluded != uint32(renamedProfileExcludedExperts) {
		t.Fatalf("applied upgrade = %d, want %d", excluded, uint32(renamedProfileExcludedExperts))
	}
	if excluded := upgradedExcludedExperts(t, false); excluded != uint32(tensor.FirstOffset) {
		t.Fatalf("unapplied upgrade = %d, want %d", excluded, uint32(tensor.FirstOffset))
	}
}

// TestProfileRenameUpgradeRefusesUnknownShape keeps the upgrade from silently
// accepting a document it cannot migrate exactly.
func TestProfileRenameUpgradeRefusesUnknownShape(t *testing.T) {
	for name, document := range map[string]string{
		"no policy":        `{"architecture":"absent"}`,
		"no dense stages":  `{"architecture":"absent","policy":{}}`,
		"non-boolean flag": `{"architecture":"absent","policy":{"DenseStages":{"PostRotaryRMSNon128":"not-a-bool"}}}`,
	} {
		if _, err := upgradeRenamedProfile([]byte(document)); err == nil {
			t.Fatalf("%s: upgrade accepted an unmigratable document", name)
		}
	}
}

// TestProfileSchemaVersion refuses a future in-place edit of the published
// schema. The rename that broke every stored profile shipped because the field
// moved while the version and the schema label stayed put, so what has to hold
// is a relationship, not a value: the label must carry the version, and every
// upgradable label must differ from the current one. Asserting the literal
// version here would only copy production policy into a test and would still
// have passed on the day the break shipped.
func TestProfileSchemaVersion(t *testing.T) {
	if !strings.HasSuffix(modelrecipe.ProfileSchema, "/v"+strconv.FormatUint(uint64(modelrecipe.ProfileVersion), 10)) {
		t.Fatalf("schema label %q does not carry version %d",
			modelrecipe.ProfileSchema, modelrecipe.ProfileVersion)
	}
	for _, upgradable := range []string{legacyProfileSchema, renamedProfileSchema} {
		if upgradable == modelrecipe.ProfileSchema {
			t.Fatalf("upgradable label %q equals the current schema; its documents cannot be routed to an upgrade",
				upgradable)
		}
	}
}

// TestProfileUpgradeBackfillsFieldsAddedAfterPublication covers the second way
// these documents are stale. The rename was not the only unversioned change:
// policy fields added to ArchitectureProfile after a document was published are
// simply absent from it, and decoding them as zero values fails the current
// validator. The registered catalog completes them, so a document that predates
// a field still upgrades instead of being refused.
func TestProfileUpgradeBackfillsFieldsAddedAfterPublication(t *testing.T) {
	document := renamedProfileFixture(t, false)
	var body map[string]any
	if err := json.Unmarshal(document, &body); err != nil {
		t.Fatal(err)
	}
	policy, ok := body["policy"].(map[string]any)
	if !ok {
		t.Fatal("fixture policy is absent")
	}
	// Drop every field the catalog declares, leaving only what identifies the
	// architecture. A document this sparse is the limit case of publication
	// predating the current profile shape.
	name := policy["Name"]
	stages := policy["DenseStages"]
	for key := range policy {
		delete(policy, key)
	}
	policy["Name"] = name
	policy["DenseStages"] = stages
	sparse, err := json.Marshal(body)
	if err != nil {
		t.Fatal(err)
	}
	upgraded, err := upgradeRenamedProfile(sparse)
	if err != nil {
		t.Fatalf("sparse published document refused: %v", err)
	}
	var decoded struct {
		Architecture string                    `json:"architecture"`
		Version      uint16                    `json:"version"`
		Policy       model.ArchitectureProfile `json:"policy"`
		Provenance   []json.RawMessage         `json:"provenance,omitempty"`
	}
	if err := strictjson.DecodeBytes(upgraded, &decoded); err != nil {
		t.Fatal(err)
	}
	registered, found := model.LookupArchitecture(decoded.Architecture)
	if !found {
		t.Fatalf("upgraded architecture %q is not registered", decoded.Architecture)
	}
	if decoded.Policy.DraftKind != registered.DraftKind ||
		decoded.Policy.DraftQueryCopies != registered.DraftQueryCopies {
		t.Fatalf("catalog fields were not backfilled: got kind=%v copies=%v want kind=%v copies=%v",
			decoded.Policy.DraftKind, decoded.Policy.DraftQueryCopies,
			registered.DraftKind, registered.DraftQueryCopies)
	}
}

// TestProfileUpgradeRefusesUnregisteredArchitecture keeps the backfill honest:
// an architecture the catalog does not declare has no authority to complete it.
func TestProfileUpgradeRefusesUnregisteredArchitecture(t *testing.T) {
	if _, err := upgradeRenamedProfile([]byte(
		`{"architecture":"absent","policy":{"DenseStages":{}}}`)); err == nil {
		t.Fatal("upgrade completed a document with no registered architecture")
	}
}
