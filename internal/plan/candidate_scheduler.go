package plan

import (
	"errors"
	"math"
	"slices"
	"sort"
	"strings"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/checked"
	"overgo/internal/runrecord"
)

const (
	CandidateSchedulerVersion   uint16 = 1
	CandidateSchedulerMediaType        = "application/vnd.overgo.candidate-scheduler+json"
	CandidateSchedulerSchema           = "overgo/candidate-scheduler/v1"
)

type CapabilityChange struct {
	Name      string              `json:"name"`
	Before    float64             `json:"before"`
	After     float64             `json:"after"`
	Direction runrecord.Direction `json:"direction"`
}

type CandidateOutcome struct {
	Evidence        artifact.ID        `json:"evidence"`
	Candidate       artifact.ID        `json:"candidate"`
	Strategy        artifact.ID        `json:"strategy"`
	WallNS          uint64             `json:"wall_ns"`
	PeakDeviceBytes uint64             `json:"peak_device_bytes"`
	Changes         []CapabilityChange `json:"changes"`
}

type ScheduledCandidate struct {
	Candidate       artifact.ID `json:"candidate"`
	Strategy        artifact.ID `json:"strategy"`
	PredictedWallNS uint64      `json:"predicted_wall_ns"`
	PredictedVRAM   uint64      `json:"predicted_vram_bytes"`
}

type SchedulerBudget struct {
	WallNS    uint64 `json:"wall_ns"`
	VRAMBytes uint64 `json:"vram_bytes"`
}

type CapabilityRate struct {
	Name         string  `json:"name"`
	DeltaPerHour float64 `json:"delta_per_hour"`
}

type SchedulerEvidenceRow struct {
	Strategy        artifact.ID      `json:"strategy"`
	Trials          uint64           `json:"trials"`
	PeakWallNS      uint64           `json:"peak_wall_ns"`
	PeakDeviceBytes uint64           `json:"peak_device_bytes"`
	Rates           []CapabilityRate `json:"rates"`
}

// CandidateScheduler: learned resource-bounded candidate order.
type CandidateScheduler struct {
	Version      uint16                 `json:"version"`
	Budget       SchedulerBudget        `json:"budget"`
	Outcomes     []CandidateOutcome     `json:"outcomes"`
	Candidates   []ScheduledCandidate   `json:"candidates"`
	EvidenceRows []SchedulerEvidenceRow `json:"evidence_rows"`
	Ranking      []artifact.ID          `json:"ranking"`
	Selected     artifact.ID            `json:"selected,omitzero"`
	ID           artifact.ID            `json:"-"`
}

var candidateSchedulerCodec = artifact.JSONDocumentCodec(
	"candidate scheduler", artifact.KindProfile, CandidateSchedulerMediaType, CandidateSchedulerSchema,
	canonicalizeCandidateScheduler,
	func(value CandidateScheduler) artifact.ID { return value.ID },
	func(value *CandidateScheduler, id artifact.ID) { value.ID = id },
	cloneCandidateScheduler,
)

func CompileCandidateScheduler(outcomes []CandidateOutcome, candidates []ScheduledCandidate, budget SchedulerBudget) (CandidateScheduler, error) {
	return candidateSchedulerCodec.New(CandidateScheduler{
		Version: CandidateSchedulerVersion, Budget: budget, Outcomes: cloneCandidateOutcomes(outcomes),
		Candidates: slices.Clone(candidates),
	})
}

func (s CandidateScheduler) Content() (artifact.Content, error) {
	return candidateSchedulerCodec.Content(s)
}

func (s CandidateScheduler) Lineage() []artifact.Lineage {
	parents := make([]artifact.ID, 0, len(s.Outcomes)+len(s.Candidates)*2)
	for _, outcome := range s.Outcomes {
		parents = append(parents, outcome.Evidence, outcome.Candidate, outcome.Strategy)
	}
	for _, candidate := range s.Candidates {
		parents = append(parents, candidate.Candidate, candidate.Strategy)
	}
	return artifact.DependencyLineage(s.ID, parents...)
}

func compileSchedulerEvidence(outcomes []CandidateOutcome) ([]SchedulerEvidenceRow, error) {
	type aggregate struct {
		wall, trials, peakWall, peakDevice uint64
		deltas                             map[string]float64
		metrics                            []string
	}
	byStrategy := map[artifact.ID]*aggregate{}
	for _, outcome := range outcomes {
		if outcome.Evidence.Kind() != artifact.KindEvidence || !outcome.Candidate.Valid() ||
			outcome.Strategy.Kind() != artifact.KindProfile || outcome.WallNS == 0 || outcome.PeakDeviceBytes == 0 || len(outcome.Changes) == 0 {
			return nil, errors.New("plan: invalid candidate outcome")
		}
		row := byStrategy[outcome.Strategy]
		if row == nil {
			row = &aggregate{deltas: map[string]float64{}}
			byStrategy[outcome.Strategy] = row
		}
		if math.MaxUint64-row.wall < outcome.WallNS {
			return nil, errors.New("plan: candidate outcome wall time overflows")
		}
		row.wall += outcome.WallNS
		row.trials++
		row.peakWall = max(row.peakWall, outcome.WallNS)
		row.peakDevice = max(row.peakDevice, outcome.PeakDeviceBytes)
		seen := map[string]bool{}
		metrics := make([]string, 0, len(outcome.Changes))
		for _, change := range outcome.Changes {
			if strings.TrimSpace(change.Name) == "" || change.Name != strings.TrimSpace(change.Name) || strings.ContainsAny(change.Name, "\x00\r\n") ||
				seen[change.Name] || !finite(change.Before) || !finite(change.After) ||
				change.Direction != runrecord.DirectionMinimize && change.Direction != runrecord.DirectionMaximize {
				return nil, errors.New("plan: invalid capability change")
			}
			seen[change.Name] = true
			metrics = append(metrics, change.Name)
			delta := change.After - change.Before
			if change.Direction == runrecord.DirectionMinimize {
				delta = -delta
			}
			row.deltas[change.Name] += delta
			if !finite(row.deltas[change.Name]) {
				return nil, errors.New("plan: capability delta overflows")
			}
		}
		slices.Sort(metrics)
		if row.metrics == nil {
			row.metrics = metrics
		} else if !slices.Equal(row.metrics, metrics) {
			return nil, errors.New("plan: strategy capability metric set differs")
		}
	}
	rows := make([]SchedulerEvidenceRow, 0, len(byStrategy))
	for strategy, aggregate := range byStrategy {
		rates := make([]CapabilityRate, 0, len(aggregate.deltas))
		for name, delta := range aggregate.deltas {
			rates = append(rates, CapabilityRate{Name: name, DeltaPerHour: delta * float64(time.Hour) / float64(aggregate.wall)})
		}
		sort.Slice(rates, func(left, right int) bool { return rates[left].Name < rates[right].Name })
		rows = append(rows, SchedulerEvidenceRow{
			Strategy: strategy, Trials: aggregate.trials, PeakWallNS: aggregate.peakWall,
			PeakDeviceBytes: aggregate.peakDevice, Rates: rates,
		})
	}
	sort.Slice(rows, func(left, right int) bool { return rows[left].Strategy.String() < rows[right].Strategy.String() })
	return rows, nil
}

func validateScheduledCandidates(candidates []ScheduledCandidate, budget SchedulerBudget) error {
	if len(candidates) == 0 || budget.WallNS == 0 || budget.VRAMBytes == 0 {
		return errors.New("plan: candidate scheduler requires candidates and resource budget")
	}
	seen := map[artifact.ID]bool{}
	for _, candidate := range candidates {
		if !candidate.Candidate.Valid() || seen[candidate.Candidate] || candidate.Strategy.Kind() != artifact.KindProfile ||
			candidate.PredictedWallNS == 0 || candidate.PredictedVRAM == 0 {
			return errors.New("plan: invalid scheduled candidate")
		}
		seen[candidate.Candidate] = true
	}
	return nil
}

func rankScheduledCandidates(candidates []ScheduledCandidate, rows []SchedulerEvidenceRow, budget SchedulerBudget) []artifact.ID {
	rowFor := func(strategy artifact.ID) SchedulerEvidenceRow {
		index, found := slices.BinarySearchFunc(rows, strategy, func(row SchedulerEvidenceRow, target artifact.ID) int {
			return strings.Compare(row.Strategy.String(), target.String())
		})
		if found {
			return rows[index]
		}
		return SchedulerEvidenceRow{}
	}
	eligible := make([]ScheduledCandidate, 0, len(candidates))
	for _, candidate := range candidates {
		row := rowFor(candidate.Strategy)
		wallFits := max(candidate.PredictedWallNS, row.PeakWallNS) <= budget.WallNS
		vramFits := max(candidate.PredictedVRAM, row.PeakDeviceBytes) <= budget.VRAMBytes
		if wallFits && vramFits {
			eligible = append(eligible, candidate)
		}
	}
	sort.SliceStable(eligible, func(left, right int) bool {
		leftRow, rightRow := rowFor(eligible[left].Strategy), rowFor(eligible[right].Strategy)
		if order := compareCapabilityRates(leftRow.Rates, rightRow.Rates); order != 0 {
			return order > 0
		}
		leftWall := max(eligible[left].PredictedWallNS, leftRow.PeakWallNS)
		rightWall := max(eligible[right].PredictedWallNS, rightRow.PeakWallNS)
		if leftWall != rightWall {
			return leftWall < rightWall
		}
		leftVRAM := max(eligible[left].PredictedVRAM, leftRow.PeakDeviceBytes)
		rightVRAM := max(eligible[right].PredictedVRAM, rightRow.PeakDeviceBytes)
		if leftVRAM != rightVRAM {
			return leftVRAM < rightVRAM
		}
		return eligible[left].Candidate.String() < eligible[right].Candidate.String()
	})
	result := make([]artifact.ID, len(eligible))
	for index := range eligible {
		result[index] = eligible[index].Candidate
	}
	return result
}

func compareCapabilityRates(left, right []CapabilityRate) int {
	if len(left) == 0 || len(right) == 0 || len(left) != len(right) {
		return compareRateTier(left) - compareRateTier(right)
	}
	leftBetter, rightBetter := false, false
	for index := range left {
		if left[index].Name != right[index].Name {
			return 0
		}
		leftBetter = leftBetter || left[index].DeltaPerHour > right[index].DeltaPerHour
		rightBetter = rightBetter || right[index].DeltaPerHour > left[index].DeltaPerHour
	}
	if leftBetter == rightBetter {
		return 0
	}
	if leftBetter {
		return 1
	}
	return -1
}

func compareRateTier(rates []CapabilityRate) int {
	if len(rates) == 0 {
		return 0
	}
	positive, negative := false, false
	for _, rate := range rates {
		positive = positive || rate.DeltaPerHour > 0
		negative = negative || rate.DeltaPerHour < 0
	}
	if positive && !negative {
		return 1
	}
	if negative && !positive {
		return -1
	}
	return 0
}

func canonicalizeCandidateScheduler(value *CandidateScheduler) error {
	if value.Version != CandidateSchedulerVersion {
		return errors.New("plan: invalid candidate scheduler")
	}
	sort.Slice(value.Outcomes, func(left, right int) bool {
		return value.Outcomes[left].Evidence.String() < value.Outcomes[right].Evidence.String()
	})
	for index := range value.Outcomes {
		sort.Slice(value.Outcomes[index].Changes, func(left, right int) bool {
			return value.Outcomes[index].Changes[left].Name < value.Outcomes[index].Changes[right].Name
		})
	}
	for index := 1; index < len(value.Outcomes); index++ {
		if value.Outcomes[index-1].Evidence == value.Outcomes[index].Evidence {
			return errors.New("plan: duplicate candidate outcome")
		}
	}
	sort.Slice(value.Candidates, func(left, right int) bool {
		return value.Candidates[left].Candidate.String() < value.Candidates[right].Candidate.String()
	})
	if err := validateScheduledCandidates(value.Candidates, value.Budget); err != nil {
		return err
	}
	rows, err := compileSchedulerEvidence(value.Outcomes)
	if err != nil {
		return err
	}
	value.EvidenceRows = rows
	value.Ranking = rankScheduledCandidates(value.Candidates, rows, value.Budget)
	if len(value.Ranking) == 0 {
		return errors.New("plan: no candidate fits scheduler budget")
	}
	value.Selected = value.Ranking[0]
	return nil
}

func cloneCandidateScheduler(value CandidateScheduler) CandidateScheduler {
	value.Outcomes = cloneCandidateOutcomes(value.Outcomes)
	value.Candidates = slices.Clone(value.Candidates)
	value.EvidenceRows = slices.Clone(value.EvidenceRows)
	for index := range value.EvidenceRows {
		value.EvidenceRows[index].Rates = slices.Clone(value.EvidenceRows[index].Rates)
	}
	value.Ranking = slices.Clone(value.Ranking)
	return value
}

func cloneCandidateOutcomes(source []CandidateOutcome) []CandidateOutcome {
	result := slices.Clone(source)
	for index := range result {
		result[index].Changes = slices.Clone(result[index].Changes)
	}
	return result
}

func finite(value float64) bool { return checked.Finite64(value) }
