package evaluation

import (
	"context"
	"errors"
	"fmt"
	"math"
	"slices"

	"overgo/internal/artifact"
	"overgo/internal/runrecord"
)

const (
	// LiveSafetyWindowMediaType identifies safety-window derivation evidence.
	LiveSafetyWindowMediaType = "application/vnd.overgo.live-safety-window+json"
	// LiveSafetyWindowSchema identifies the safety-window contract.
	LiveSafetyWindowSchema = "overgo/live-safety-window/v1"
	// LiveSafetyModeEnvelope marks a small-sample window: the boundaries are
	// the exact history extremes, and a decision is permitted only when the
	// live observations separate entirely from that envelope.
	LiveSafetyModeEnvelope = "envelope-separation"
	// LiveSafetyModeQuantile marks a large-history window: the boundaries
	// are symmetric order statistics, a distribution-free tolerance band.
	LiveSafetyModeQuantile = "quantile-band"
	// liveSafetyTailDenominator sets the trimmed tail per side to one
	// fortieth of the history — a 2.5% order-statistic tail, giving the
	// band at least 95% distribution-free content. Histories too small to
	// trim even one observation stay in envelope mode.
	liveSafetyTailDenominator = 40
	// liveSafetyFalseAlarmDenominator bounds the chance that one full
	// comparison window falls outside the boundaries by luck at one in a
	// thousand; the required observation count derives from it and the
	// history size, never from a hand-picked constant.
	liveSafetyFalseAlarmDenominator = 1000
)

// LiveSafetyWindow is the derived comparison contract for one live metric:
// alarm boundaries read from the metric's own history by order statistics
// — no Gaussian, stationary, or IID assumption — with the explicit number
// of comparable observations a decision requires. The record cites the
// history evidence it derives from, so every boundary is auditable.
type LiveSafetyWindow struct {
	Metric               string              `json:"metric"`
	Direction            runrecord.Direction `json:"direction"`
	History              artifact.ID         `json:"history"`
	Samples              uint64              `json:"samples"`
	Mode                 string              `json:"mode"`
	Lower                float64             `json:"lower"`
	Upper                float64             `json:"upper"`
	RequiredObservations uint64              `json:"required_observations"`
}

// DeriveLiveSafetyWindow derives one comparison window from a metric
// history. A history of fewer than two finite observations cannot bound
// anything and refuses. Small histories keep the exact [min, max] envelope
// and permit decisions only through complete separation; histories large
// enough to trim a 2.5% order-statistic tail per side use the symmetric
// quantile band. The required observation count is the smallest window
// whose chance of falling outside the boundaries by luck stays under the
// registered false-alarm bound given the history size.
func DeriveLiveSafetyWindow(
	metric string,
	direction runrecord.Direction,
	history []float64,
	evidence artifact.ID,
) (LiveSafetyWindow, error) {
	if metric == "" {
		return LiveSafetyWindow{}, errors.New("evaluation: safety window requires its metric name")
	}
	if direction != runrecord.DirectionMaximize && direction != runrecord.DirectionMinimize {
		return LiveSafetyWindow{}, fmt.Errorf("evaluation: safety window direction %q is not a comparison direction", direction)
	}
	if evidence.Kind() != artifact.KindEvidence {
		return LiveSafetyWindow{}, errors.New("evaluation: safety window requires the history evidence it derives from")
	}
	if len(history) < 2 {
		return LiveSafetyWindow{}, errors.New("evaluation: a history of fewer than two observations cannot bound a comparison")
	}
	sorted := slices.Clone(history)
	for _, value := range sorted {
		if !finite(value) {
			return LiveSafetyWindow{}, errors.New("evaluation: safety window history holds a non-finite observation")
		}
	}
	slices.Sort(sorted)
	samples := uint64(len(sorted))
	trim := int(samples+1) / liveSafetyTailDenominator
	window := LiveSafetyWindow{
		Metric: metric, Direction: direction, History: evidence, Samples: samples,
	}
	if trim == 0 {
		window.Mode = LiveSafetyModeEnvelope
		window.Lower, window.Upper = sorted[0], sorted[len(sorted)-1]
	} else {
		window.Mode = LiveSafetyModeQuantile
		window.Lower, window.Upper = sorted[trim], sorted[len(sorted)-1-trim]
	}
	// P(one draw outside the envelope) <= 2/(samples+1); the window length
	// makes the joint luck probability at most the registered bound.
	outside := 2.0 / float64(samples+1)
	required := math.Ceil(math.Log(1.0/liveSafetyFalseAlarmDenominator) / math.Log(outside))
	if required < 1 || math.IsNaN(required) || math.IsInf(required, 0) {
		required = 1
	}
	window.RequiredObservations = uint64(required)
	return window, nil
}

// PublishLiveSafetyWindow commits one derivation as immutable evidence
// citing the history it was read from.
func PublishLiveSafetyWindow(
	ctx context.Context,
	store artifact.Repository,
	window LiveSafetyWindow,
) (artifact.ID, error) {
	if window.Samples == 0 || window.Mode == "" {
		return artifact.ID{}, errors.New("evaluation: only a derived safety window can publish")
	}
	contract := artifact.DocumentContract{
		Kind: artifact.KindEvidence, MediaType: LiveSafetyWindowMediaType, Schema: LiveSafetyWindowSchema,
	}
	content, err := artifact.JSONContent(contract, window)
	if err != nil {
		return artifact.ID{}, err
	}
	batch, err := artifact.NewDocumentBatch(
		"live-safety/window/"+content.Descriptor.ID.String(),
		[]artifact.Content{content},
		artifact.DependencyLineage(content.Descriptor.ID, window.History),
		nil,
	)
	if err != nil {
		return artifact.ID{}, err
	}
	if _, err := artifact.CommitBatch(ctx, store, batch); err != nil {
		return artifact.ID{}, err
	}
	return content.Descriptor.ID, nil
}
