package plan

import (
	"errors"
	"fmt"
	"slices"
	"sort"

	"overgo/internal/artifact"
	"overgo/internal/textcheck"
)

const (
	schedulerPolicyVersion   uint16 = 1
	schedulerPolicyMediaType        = "application/vnd.overgo.scheduler-policy+json"
	schedulerPolicySchema           = "overgo/scheduler-policy/v1"

	nsPerHour = uint64(3_600_000_000_000)
)

// SchedulerRealization is one measured unit of scheduling history: a
// candidate track that ran to completion, the capability delta its evaluator
// realized, and the wall-clock it consumed doing so. The evidence artifact is
// the committed measurement the delta derives from -- history without a
// measurement cannot train the policy.
type SchedulerRealization struct {
	Track           string      `json:"track"`
	CapabilityDelta float64     `json:"capability_delta"`
	WallNS          uint64      `json:"wall_ns"`
	Evidence        artifact.ID `json:"evidence"`
}

// TrackValue is the learned per-track record: realized capability delta and
// wall-clock totals. The rate is derived at query time, never stored, so the
// artifact stays a faithful tally of what actually happened.
type TrackValue struct {
	Track        string  `json:"track"`
	Realizations uint32  `json:"realizations"`
	TotalDelta   float64 `json:"total_delta"`
	TotalWallNS  uint64  `json:"total_wall_ns"`
}

// DeltaPerWallHour is the realized capability delta per wall-hour: the
// scheduling value this track has actually demonstrated.
func (v TrackValue) DeltaPerWallHour() float64 {
	if v.TotalWallNS == 0 {
		return 0
	}
	return v.TotalDelta * float64(nsPerHour) / float64(v.TotalWallNS)
}

// SchedulerPolicy is next-candidate selection as a learned artifact: ranking
// derives from the realized capability delta per wall-hour each track has
// measured, not from an applied formula over predictions. Tracks the ledger
// has never measured carry zero learned value -- they rank between
// demonstrated wins and demonstrated losses, which is exactly what ignorance
// should buy.
type SchedulerPolicy struct {
	Version uint16       `json:"version"`
	Tracks  []TrackValue `json:"tracks"`
	// Sources are the measurement artifacts the tallies derive from.
	Sources []artifact.ID `json:"sources"`
	ID      artifact.ID   `json:"-"`
}

var schedulerPolicyCodec = artifact.JSONDocumentCodec(
	"scheduler policy", artifact.KindEvidence, schedulerPolicyMediaType, schedulerPolicySchema,
	canonicalizeSchedulerPolicy,
	func(value SchedulerPolicy) artifact.ID { return value.ID },
	func(value *SchedulerPolicy, id artifact.ID) { value.ID = id },
	func(value SchedulerPolicy) SchedulerPolicy {
		value.Tracks = slices.Clone(value.Tracks)
		value.Sources = slices.Clone(value.Sources)
		return value
	},
)

// TrainSchedulerPolicy tallies realized history into the learned policy.
// Every realization needs a track, positive wall-clock, and its committed
// measurement evidence; deltas may be negative -- demonstrated losses are as
// instructive as demonstrated wins.
func TrainSchedulerPolicy(realizations []SchedulerRealization) (SchedulerPolicy, error) {
	if len(realizations) == 0 {
		return SchedulerPolicy{}, errors.New("plan: scheduler policy requires realized history")
	}
	tallies := map[string]*TrackValue{}
	sources := map[artifact.ID]bool{}
	for _, realization := range realizations {
		if !textcheck.Bounded(realization.Track, 2048, "\x00\r\n") || realization.Track == "" {
			return SchedulerPolicy{}, errors.New("plan: scheduler realization requires a track")
		}
		if realization.WallNS == 0 {
			return SchedulerPolicy{}, errors.New("plan: scheduler realization requires measured wall-clock")
		}
		if !realization.Evidence.Valid() {
			return SchedulerPolicy{}, errors.New("plan: scheduler realization requires measurement evidence")
		}
		tally := tallies[realization.Track]
		if tally == nil {
			tally = &TrackValue{Track: realization.Track}
			tallies[realization.Track] = tally
		}
		tally.Realizations++
		tally.TotalDelta += realization.CapabilityDelta
		tally.TotalWallNS += realization.WallNS
		sources[realization.Evidence] = true
	}
	policy := SchedulerPolicy{Version: schedulerPolicyVersion}
	for _, tally := range tallies {
		policy.Tracks = append(policy.Tracks, *tally)
	}
	for source := range sources {
		policy.Sources = append(policy.Sources, source)
	}
	return schedulerPolicyCodec.New(policy)
}

func ParseSchedulerPolicy(content []byte) (SchedulerPolicy, error) {
	return schedulerPolicyCodec.Parse(content)
}

func (p SchedulerPolicy) Content() (artifact.Content, error) {
	return schedulerPolicyCodec.Content(p)
}

// Batch commits the policy with lineage to every measurement it learned from.
func (p SchedulerPolicy) Batch(key string) (artifact.Batch, error) {
	return schedulerPolicyCodec.Batch(key, p, artifact.DependencyLineage(p.ID, p.Sources...), nil)
}

// SchedulerCandidate is one admissible unit of next work: the task it would
// advance, the track whose realized history prices it, and its predicted
// wall-clock and VRAM demands.
type SchedulerCandidate struct {
	Task            string `json:"task"`
	Track           string `json:"track"`
	PredictedWallNS uint64 `json:"predicted_wall_ns"`
	VRAMGiB         int    `json:"vram_gib"`
}

// SchedulerPolicyBudget bounds one policy selection round.
type SchedulerPolicyBudget struct {
	WallNS  uint64 `json:"wall_ns"`
	VRAMGiB int    `json:"vram_gib"`
}

// CandidateRank is one ranked or excluded candidate. Selection remains
// advisory by contract: the rank names a reason, never starts work.
type CandidateRank struct {
	Candidate SchedulerCandidate `json:"candidate"`
	// RatePerWallHour is the track's learned realized delta per wall-hour;
	// zero when the ledger has never measured the track.
	RatePerWallHour float64 `json:"rate_per_wall_hour"`
	Admissible      bool    `json:"admissible"`
	Reason          string  `json:"reason"`
}

// SelectNext ranks candidates by the learned realized rate of their tracks
// under the wall-clock and VRAM budget. Admissible candidates sort by rate,
// then by shorter predicted wall, then by task; candidates the budget cannot
// admit are kept in the output with their exclusion reason so the selection
// is auditable, not silent.
func (p SchedulerPolicy) SelectNext(candidates []SchedulerCandidate, budget SchedulerPolicyBudget) ([]CandidateRank, error) {
	if budget.WallNS == 0 {
		return nil, errors.New("plan: scheduler selection requires a wall-clock budget")
	}
	rates := make(map[string]float64, len(p.Tracks))
	for _, track := range p.Tracks {
		rates[track.Track] = track.DeltaPerWallHour()
	}
	ranks := make([]CandidateRank, 0, len(candidates))
	for _, candidate := range candidates {
		rank := CandidateRank{Candidate: candidate, RatePerWallHour: rates[candidate.Track]}
		switch {
		case candidate.Task == "" || candidate.Track == "":
			rank.Reason = "candidate requires a task and a track"
		case candidate.PredictedWallNS == 0:
			rank.Reason = "candidate requires a wall-clock prediction"
		case candidate.PredictedWallNS > budget.WallNS:
			rank.Reason = fmt.Sprintf("predicted wall %d ns exceeds budget %d ns", candidate.PredictedWallNS, budget.WallNS)
		case budget.VRAMGiB > 0 && candidate.VRAMGiB > budget.VRAMGiB:
			rank.Reason = fmt.Sprintf("VRAM demand %d GiB exceeds budget %d GiB", candidate.VRAMGiB, budget.VRAMGiB)
		default:
			rank.Admissible = true
			if _, measured := rates[candidate.Track]; measured {
				rank.Reason = fmt.Sprintf("track realized %+.6f capability delta per wall-hour", rank.RatePerWallHour)
			} else {
				rank.Reason = "track has no realized history; ranked at zero learned value"
			}
		}
		ranks = append(ranks, rank)
	}
	sort.SliceStable(ranks, func(i, j int) bool {
		left, right := ranks[i], ranks[j]
		if left.Admissible != right.Admissible {
			return left.Admissible
		}
		if left.RatePerWallHour != right.RatePerWallHour {
			return left.RatePerWallHour > right.RatePerWallHour
		}
		if left.Candidate.PredictedWallNS != right.Candidate.PredictedWallNS {
			return left.Candidate.PredictedWallNS < right.Candidate.PredictedWallNS
		}
		return left.Candidate.Task < right.Candidate.Task
	})
	return ranks, nil
}

func canonicalizeSchedulerPolicy(value *SchedulerPolicy) error {
	if value == nil || value.Version != schedulerPolicyVersion {
		return errors.New("plan: invalid scheduler policy version")
	}
	if len(value.Tracks) == 0 || len(value.Sources) == 0 {
		return errors.New("plan: scheduler policy requires tracks and measurement sources")
	}
	for _, track := range value.Tracks {
		if track.Track == "" || track.Realizations == 0 || track.TotalWallNS == 0 {
			return errors.New("plan: scheduler policy track requires a name, realizations and wall-clock")
		}
	}
	sort.Slice(value.Tracks, func(i, j int) bool { return value.Tracks[i].Track < value.Tracks[j].Track })
	for i := 1; i < len(value.Tracks); i++ {
		if value.Tracks[i].Track == value.Tracks[i-1].Track {
			return errors.New("plan: scheduler policy tracks must be unique")
		}
	}
	for _, source := range value.Sources {
		if !source.Valid() {
			return errors.New("plan: scheduler policy source must be a valid artifact")
		}
	}
	sort.Slice(value.Sources, func(i, j int) bool {
		return value.Sources[i].String() < value.Sources[j].String()
	})
	value.Sources = slices.Compact(value.Sources)
	return nil
}
