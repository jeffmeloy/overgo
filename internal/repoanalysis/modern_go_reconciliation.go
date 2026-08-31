package repoanalysis

import "slices"

// ModernGoPriorDisposition is the bounded outcome of auditing one commit from
// the earlier modernization lane against the current tree and catalog.
type ModernGoPriorDisposition string

const (
	// ModernGoPriorSuperseded marks prior authority replaced by this campaign.
	ModernGoPriorSuperseded ModernGoPriorDisposition = "superseded"
	// ModernGoPriorReimplement marks a sound invariant needing a new implementation.
	ModernGoPriorReimplement ModernGoPriorDisposition = "reimplement"
	// ModernGoPriorReplayCandidate marks transforms eligible for current-HEAD review.
	ModernGoPriorReplayCandidate ModernGoPriorDisposition = "replay-candidate"
	// ModernGoPriorSupporting marks relevant source-analysis fixes outside the catalog.
	ModernGoPriorSupporting ModernGoPriorDisposition = "supporting"
	// ModernGoPriorExcluded marks topology or orchestration that must not be replayed.
	ModernGoPriorExcluded ModernGoPriorDisposition = "excluded"
)

// ModernGoPriorCommit records how prior work feeds this campaign. Commit and
// subject preserve provenance; disposition and rationale prevent an unsafe
// wholesale merge from standing in for reconciliation.
type ModernGoPriorCommit struct {
	Commit      string                   `json:"commit"`
	Subject     string                   `json:"subject"`
	Disposition ModernGoPriorDisposition `json:"disposition"`
	Rationale   string                   `json:"rationale"`
	Guidelines  []string                 `json:"guidelines,omitempty"`
}

var modernGoPriorCampaign = []ModernGoPriorCommit{
	{Commit: "97137f1ad3498a88fa89a02fa4fe2301b2695f61", Subject: "Plan modern Go conformance from a measured census", Disposition: ModernGoPriorSuperseded, Rationale: "replaced by the current authoritative campaign plan"},
	{Commit: "a1ffc6867783835b6d5c5029643d6d2608cfcf09", Subject: "Scope the lane to catalog alignment and widen the census to all 38", Disposition: ModernGoPriorSuperseded, Rationale: "the pinned catalog now contains renamed and additional rules"},
	{Commit: "313a14483e4bc43eee727215a2d99259aafc4fbe", Subject: "Compute the modern-Go conformance census from syntax", Disposition: ModernGoPriorReimplement, Rationale: "reuse the typed document shape but replace syntax-only favorable gaps with complete typed measurement"},
	{Commit: "b6cf8bd4533be32f24832644a9bbd56abd3cf713", Subject: "Ratchet modern-Go debt so the ceiling only falls", Disposition: ModernGoPriorReimplement, Rationale: "retain the monotonic invariant and bind it to complete catalog coverage and exact exceptions"},
	{Commit: "13dcf74b400fa91be197afaf96d000092a9ea3bd", Subject: "Close efaceany and lower the ceiling to 3214", Disposition: ModernGoPriorReplayCandidate, Rationale: "re-evaluate the mechanical edits on current HEAD", Guidelines: []string{"any"}},
	{Commit: "207bdca5dd42f99b740657beedaa8dc8f7d7387e", Subject: "Close reflecttypefor and narrow the diagnostic to what it means", Disposition: ModernGoPriorReplayCandidate, Rationale: "reuse the detector oracle and replay only still-applicable edits", Guidelines: []string{"reflect_type_for"}},
	{Commit: "27535a793a3ea6c68c310f7a1f2fb71736b926c0", Subject: "Carry closures across a transform instead of orphaning them", Disposition: ModernGoPriorSupporting, Rationale: "retain the generic closure-preservation fix independently of modernization policy"},
	{Commit: "24cbe40959bc746a2b81aba71df52e1bbf09d5ac", Subject: "Absorb master into the modern-Go lane", Disposition: ModernGoPriorExcluded, Rationale: "merge topology is not an implementation change to replay"},
	{Commit: "3e292b5817cea7379b1d49f8d0980beb9d588bf2", Subject: "Give the repeated pre-commit ritual a deterministic owner", Disposition: ModernGoPriorExcluded, Rationale: "lane orchestration is outside the catalog reconciliation step"},
	{Commit: "603f238104a9f870ba4d46f82245098ed7cdeea9", Subject: "Ship the lane command, and the untracked files it forgot", Disposition: ModernGoPriorExcluded, Rationale: "the current gate remains the sole campaign commit authority"},
	{Commit: "e2202611f818a61be3d65ddfeecf926044aac6b0", Subject: "Govern every worktree, and scope a stop to the lane that recorded it", Disposition: ModernGoPriorExcluded, Rationale: "worktree governance is independent of modern-Go conformance"},
	{Commit: "bedd42242dc63ad3342d8176e37973cd75f1b6ed", Subject: "Count a literal within its expression, not across its scope", Disposition: ModernGoPriorSupporting, Rationale: "retain as source-analysis correctness evidence, not as a catalog transform"},
	{Commit: "0c224d095050374431f3e4563a03d07c9beb49b7", Subject: "Close rangeint across the tree", Disposition: ModernGoPriorReplayCandidate, Rationale: "partition the prior bulk edit into control-plane and numerical parity slices", Guidelines: []string{"range_over_int"}},
	{Commit: "1dde63e011798806cf7f96feaaafc08ddb592357", Subject: "Close naturally ordered sorts through slices.Sort", Disposition: ModernGoPriorReplayCandidate, Rationale: "reuse naturally ordered conversions after deterministic-order review", Guidelines: []string{"slices_sort"}},
	{Commit: "2ff331bfbbba67aa377b295336df9c95389a71a3", Subject: "Empty a map with clear rather than a delete loop", Disposition: ModernGoPriorReplayCandidate, Rationale: "reuse exact delete-only transforms after aliasing review", Guidelines: []string{"clear"}},
	{Commit: "dd2eb0a95d3f9a398a325523b0f5ea4dc619c566", Subject: "Cut a prefix once instead of testing then trimming", Disposition: ModernGoPriorReplayCandidate, Rationale: "reuse parser-proven exact prefix transforms", Guidelines: []string{"strings_cut_prefix_suffix"}},
	{Commit: "738294245a0d9613a3213e03932548d3b97dc423", Subject: "Hold bytescut closed with its own oracle", Disposition: ModernGoPriorReplayCandidate, Rationale: "reuse the bytes separator oracle and re-evaluate edits on current HEAD", Guidelines: []string{"bytes_cut"}},
}

// ModernGoPriorCampaign returns a detached, ordered reconciliation record.
func ModernGoPriorCampaign() []ModernGoPriorCommit {
	commits := slices.Clone(modernGoPriorCampaign)
	for index := range commits {
		commits[index].Guidelines = slices.Clone(commits[index].Guidelines)
	}
	return commits
}
