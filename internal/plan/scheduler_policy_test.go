package plan

import (
	"testing"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/testutil"
)

// TestSchedulerRankedByRealizedDelta pins the scheduler-as-artifact contract:
// next-candidate selection ranks by the capability delta per wall-hour each
// track has actually realized, not by an applied formula over predictions --
// the shortest-wall candidate loses to the highest-realized-rate track, a
// demonstrated loss ranks below no history at all, budget exclusions stay in
// the output with reasons, identity is order-canonical, and the parsed
// artifact selects identically to the trained one.
func TestSchedulerPolicyRankedByRealizedDelta(t *testing.T) {
	evidence := func(seed string) artifact.ID {
		return testutil.ArtifactID(t, artifact.KindEvidence, "scheduler-measurement-"+seed)
	}
	hour := uint64(time.Hour.Nanoseconds())
	realizations := []SchedulerRealization{
		// composition: large delta over long wall -- solid but slow (0.15/h).
		{Track: "composition-bridge", CapabilityDelta: 0.20, WallNS: 90 * hour / 60, Evidence: evidence("comp-1")},
		{Track: "composition-bridge", CapabilityDelta: 0.10, WallNS: 30 * hour / 60, Evidence: evidence("comp-2")},
		// tier0 chaining: small deltas realized fast (0.20/h) -- best rate.
		{Track: "tier0-chain", CapabilityDelta: 0.03, WallNS: 9 * hour / 60, Evidence: evidence("tier0-1")},
		{Track: "tier0-chain", CapabilityDelta: 0.02, WallNS: 6 * hour / 60, Evidence: evidence("tier0-2")},
		// adapter: measured regression (-0.10/h) -- a demonstrated loss.
		{Track: "adapter", CapabilityDelta: -0.10, WallNS: hour, Evidence: evidence("adapter-1")},
	}
	policy, err := TrainSchedulerPolicy(realizations)
	if err != nil {
		t.Fatal(err)
	}
	reversed := make([]SchedulerRealization, len(realizations))
	for i, realization := range realizations {
		reversed[len(realizations)-1-i] = realization
	}
	replay, err := TrainSchedulerPolicy(reversed)
	if err != nil || replay.ID != policy.ID {
		t.Fatalf("policy identity not order-canonical: (%v, %v)", replay.ID, err)
	}
	rateOf := func(track string) float64 {
		for _, value := range policy.Tracks {
			if value.Track == track {
				return value.DeltaPerWallHour()
			}
		}
		t.Fatalf("track %s not learned", track)
		return 0
	}
	if rate := rateOf("tier0-chain"); rate < 0.199 || rate > 0.201 {
		t.Fatalf("tier0 realized rate = %f, want 0.20 per wall-hour", rate)
	}
	if rate := rateOf("composition-bridge"); rate < 0.149 || rate > 0.151 {
		t.Fatalf("composition realized rate = %f, want 0.15 per wall-hour", rate)
	}

	candidates := []SchedulerCandidate{
		// The shortest-wall candidate belongs to the losing track: any
		// shortest-job-first formula would pick it.
		{Task: "adapter-retry", Track: "adapter", PredictedWallNS: 10 * hour / 60, VRAMGiB: 8},
		{Task: "compose-next-bridge", Track: "composition-bridge", PredictedWallNS: hour, VRAMGiB: 16},
		{Task: "chain-next-pair", Track: "tier0-chain", PredictedWallNS: 30 * hour / 60, VRAMGiB: 12},
		{Task: "index-refresh", Track: "hdc-index", PredictedWallNS: 20 * hour / 60, VRAMGiB: 4},
		{Task: "chain-giant-pair", Track: "tier0-chain", PredictedWallNS: 30 * hour / 60, VRAMGiB: 48},
		{Task: "compose-marathon", Track: "composition-bridge", PredictedWallNS: 100 * hour, VRAMGiB: 16},
	}
	budget := SchedulerPolicyBudget{WallNS: 2 * hour, VRAMGiB: 24}
	ranks, err := policy.SelectNext(candidates, budget)
	if err != nil {
		t.Fatal(err)
	}
	order := make([]string, 0, len(ranks))
	for _, rank := range ranks {
		if rank.Admissible {
			order = append(order, rank.Candidate.Task)
		}
	}
	want := []string{"chain-next-pair", "compose-next-bridge", "index-refresh", "adapter-retry"}
	if len(order) != len(want) {
		t.Fatalf("admissible = %v, want %v", order, want)
	}
	for i := range want {
		if order[i] != want[i] {
			t.Fatalf("admissible order = %v, want %v (realized rate must outrank shortest wall)", order, want)
		}
	}
	if ranks[0].Candidate.Task == "adapter-retry" {
		t.Fatal("shortest-wall candidate won: selection is a formula, not learned")
	}
	for _, rank := range ranks {
		switch rank.Candidate.Task {
		case "chain-giant-pair":
			if rank.Admissible || rank.Reason == "" {
				t.Fatalf("VRAM-exceeding candidate = %+v, want a reasoned exclusion", rank)
			}
		case "compose-marathon":
			if rank.Admissible || rank.Reason == "" {
				t.Fatalf("wall-exceeding candidate = %+v, want a reasoned exclusion", rank)
			}
		case "index-refresh":
			if rank.RatePerWallHour != 0 {
				t.Fatalf("unmeasured track rate = %f, want zero learned value", rank.RatePerWallHour)
			}
		case "adapter-retry":
			if rank.RatePerWallHour >= 0 {
				t.Fatalf("losing track rate = %f, want the measured regression", rank.RatePerWallHour)
			}
		}
	}

	content, err := policy.Content()
	if err != nil {
		t.Fatal(err)
	}
	parsed, err := ParseSchedulerPolicy(content.Data)
	if err != nil || parsed.ID != policy.ID || len(parsed.Sources) != len(realizations) {
		t.Fatalf("roundtrip = (%v, %v), want %d sources", parsed.ID, err, len(realizations))
	}
	replayed, err := parsed.SelectNext(candidates, budget)
	if err != nil || replayed[0].Candidate.Task != ranks[0].Candidate.Task {
		t.Fatalf("parsed policy selects differently: (%v, %v)", replayed[0].Candidate.Task, err)
	}

	if _, err := TrainSchedulerPolicy(nil); err == nil {
		t.Fatal("empty history accepted")
	}
	if _, err := TrainSchedulerPolicy([]SchedulerRealization{
		{Track: "composition-bridge", CapabilityDelta: 0.1, Evidence: evidence("no-wall")},
	}); err == nil {
		t.Fatal("realization without wall-clock accepted")
	}
	if _, err := TrainSchedulerPolicy([]SchedulerRealization{
		{Track: "composition-bridge", CapabilityDelta: 0.1, WallNS: hour},
	}); err == nil {
		t.Fatal("realization without measurement evidence accepted")
	}
	if _, err := policy.SelectNext(candidates, SchedulerPolicyBudget{VRAMGiB: 24}); err == nil {
		t.Fatal("selection without a wall-clock budget accepted")
	}
}
