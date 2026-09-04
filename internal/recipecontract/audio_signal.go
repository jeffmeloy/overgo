package recipecontract

import (
	"errors"
	"math"
	"slices"
	"time"

	"overgo/internal/checked"
)

// AudioDecodeStatus describes whether bytes became a complete decoded signal.
type AudioDecodeStatus string

const (
	// AudioDecodeNotAttempted records that schema admission stopped decoding.
	AudioDecodeNotAttempted AudioDecodeStatus = "not-attempted"
	// AudioDecodeComplete records a complete decoded sample stream.
	AudioDecodeComplete AudioDecodeStatus = "complete"
	// AudioDecodeUnsupported records a recognized but unsupported encoding.
	AudioDecodeUnsupported AudioDecodeStatus = "unsupported"
	// AudioDecodeCorrupt records malformed encoded audio.
	AudioDecodeCorrupt AudioDecodeStatus = "corrupt"
	// AudioDecodeTruncated records an incomplete encoded stream.
	AudioDecodeTruncated AudioDecodeStatus = "truncated"
	// AudioDecodeResourceLimit records a bounded decoder refusing allocation.
	AudioDecodeResourceLimit AudioDecodeStatus = "resource-limit"
)

// DecodedAudioSignalProfile is policy-neutral evidence measured from decoded
// channel-interleaved samples. FrameCount counts one sample position across
// all channels; SampleCount counts the interleaved scalar values.
type DecodedAudioSignalProfile struct {
	Source               AudioReference    `json:"source"`
	SchemaValid          bool              `json:"schema_valid"`
	DecodeStatus         AudioDecodeStatus `json:"decode_status"`
	Format               AudioFormat       `json:"format,omitzero"`
	FrameCount           uint64            `json:"frame_count,omitzero"`
	SampleCount          uint64            `json:"sample_count,omitzero"`
	FiniteSampleCount    uint64            `json:"finite_sample_count,omitzero"`
	NonFiniteSampleCount uint64            `json:"non_finite_sample_count,omitzero"`
	ClipThreshold        float64           `json:"clip_threshold,omitzero"`
	ClippedSampleCount   uint64            `json:"clipped_sample_count,omitzero"`
	PeakAbsolute         float64           `json:"peak_absolute,omitzero"`
	RootMeanSquare       float64           `json:"root_mean_square,omitzero"`
	DCOffset             float64           `json:"dc_offset,omitzero"`
}

// Validate rejects internally inconsistent decode and measurement evidence.
func (profile DecodedAudioSignalProfile) Validate() error {
	if err := profile.Source.Validate(); err != nil {
		return err
	}
	switch profile.DecodeStatus {
	case AudioDecodeNotAttempted, AudioDecodeComplete, AudioDecodeUnsupported, AudioDecodeCorrupt, AudioDecodeTruncated, AudioDecodeResourceLimit:
	default:
		return errors.New("audio signal: invalid decode state")
	}
	if profile.DecodeStatus == AudioDecodeComplete && !profile.SchemaValid {
		return errors.New("audio signal: invalid decode state")
	}
	if profile.DecodeStatus != AudioDecodeComplete {
		if profile.hasDecodedMeasurements() {
			return errors.New("audio signal: failed decode carries measurements")
		}
		return nil
	}
	if err := profile.Format.Validate(); err != nil {
		return err
	}
	samples, ok := checked.Mul64(profile.FrameCount, uint64(profile.Format.Channels))
	if !ok || samples != profile.SampleCount {
		return errors.New("audio signal: sample geometry differs")
	}
	classified, ok := checked.Add64(profile.FiniteSampleCount, profile.NonFiniteSampleCount)
	if !ok || classified != profile.SampleCount || profile.ClippedSampleCount > profile.FiniteSampleCount {
		return errors.New("audio signal: sample classification differs")
	}
	if !checked.PositiveFinite64(profile.ClipThreshold) ||
		!checked.NonNegativeFinite64(profile.PeakAbsolute) ||
		!checked.NonNegativeFinite64(profile.RootMeanSquare) || !checked.Finite64(profile.DCOffset) ||
		profile.RootMeanSquare > profile.PeakAbsolute || math.Abs(profile.DCOffset) > profile.RootMeanSquare {
		return errors.New("audio signal: invalid measurements")
	}
	clipped := profile.ClippedSampleCount != 0
	if clipped != (profile.PeakAbsolute >= profile.ClipThreshold) {
		return errors.New("audio signal: clipping evidence differs from peak")
	}
	if _, ok := profile.DurationNanoseconds(); !ok {
		return errors.New("audio signal: duration cannot be represented")
	}
	return nil
}

// DurationNanoseconds derives duration from frame count and sample rate.
func (profile DecodedAudioSignalProfile) DurationNanoseconds() (uint64, bool) {
	if profile.DecodeStatus != AudioDecodeComplete || profile.Format.SampleRate == 0 {
		return 0, false
	}
	perSecond := uint64(time.Second)
	seconds := profile.FrameCount / profile.Format.SampleRate
	remainder := profile.FrameCount % profile.Format.SampleRate
	whole, ok := checked.Mul64(seconds, perSecond)
	if !ok {
		return 0, false
	}
	partialProduct, ok := checked.Mul64(remainder, perSecond)
	if !ok {
		return 0, false
	}
	return checked.Add64(whole, partialProduct/profile.Format.SampleRate)
}

func (profile DecodedAudioSignalProfile) hasDecodedMeasurements() bool {
	return profile.Format != (AudioFormat{}) || profile.FrameCount != 0 || profile.SampleCount != 0 ||
		profile.FiniteSampleCount != 0 || profile.NonFiniteSampleCount != 0 || profile.ClipThreshold != 0 ||
		profile.ClippedSampleCount != 0 || profile.PeakAbsolute != 0 || profile.RootMeanSquare != 0 ||
		profile.DCOffset != 0
}

// AudioAdmissionPolicy holds the artifact-owned thresholds applied to a
// decoded signal profile. A zero maximum duration means no upper bound.
type AudioAdmissionPolicy struct {
	MinimumDurationNanoseconds uint64  `json:"minimum_duration_nanoseconds"`
	MaximumDurationNanoseconds uint64  `json:"maximum_duration_nanoseconds,omitzero"`
	MinimumChannels            uint32  `json:"minimum_channels"`
	MaximumChannels            uint32  `json:"maximum_channels"`
	MaximumNonFiniteSamples    uint64  `json:"maximum_non_finite_samples"`
	SilenceRMSThreshold        float64 `json:"silence_rms_threshold"`
	MaximumClippedFraction     float64 `json:"maximum_clipped_fraction"`
	MaximumAbsoluteDCOffset    float64 `json:"maximum_absolute_dc_offset"`
}

// Validate rejects incomplete, inverted, or non-finite policy bounds.
func (policy AudioAdmissionPolicy) Validate() error {
	if policy.MinimumChannels == 0 || policy.MaximumChannels < policy.MinimumChannels ||
		policy.MaximumDurationNanoseconds != 0 && policy.MaximumDurationNanoseconds < policy.MinimumDurationNanoseconds ||
		!checked.NonNegativeFinite64(policy.SilenceRMSThreshold) ||
		!checked.UnitInterval64(policy.MaximumClippedFraction) ||
		!checked.NonNegativeFinite64(policy.MaximumAbsoluteDCOffset) {
		return errors.New("audio admission: invalid policy")
	}
	return nil
}

// AudioSignalViolation is one closed reason that a signal is quarantined.
type AudioSignalViolation string

const (
	// AudioViolationSchemaInvalid rejects an invalid source schema.
	AudioViolationSchemaInvalid AudioSignalViolation = "schema-invalid"
	// AudioViolationDecodeIncomplete rejects a source that did not decode completely.
	AudioViolationDecodeIncomplete AudioSignalViolation = "decode-incomplete"
	// AudioViolationDurationBelowMinimum rejects a signal shorter than policy.
	AudioViolationDurationBelowMinimum AudioSignalViolation = "duration-below-minimum"
	// AudioViolationDurationAboveMaximum rejects a signal longer than policy.
	AudioViolationDurationAboveMaximum AudioSignalViolation = "duration-above-maximum"
	// AudioViolationChannelsBelowMinimum rejects insufficient channel geometry.
	AudioViolationChannelsBelowMinimum AudioSignalViolation = "channels-below-minimum"
	// AudioViolationChannelsAboveMaximum rejects excessive channel geometry.
	AudioViolationChannelsAboveMaximum AudioSignalViolation = "channels-above-maximum"
	// AudioViolationNonFiniteSamples rejects excessive NaN or infinite samples.
	AudioViolationNonFiniteSamples AudioSignalViolation = "non-finite-samples"
	// AudioViolationSilent rejects RMS at or below the silence threshold.
	AudioViolationSilent AudioSignalViolation = "silent"
	// AudioViolationClipped rejects excessive saturated samples.
	AudioViolationClipped AudioSignalViolation = "clipped"
	// AudioViolationDCOffset rejects excessive absolute mean amplitude.
	AudioViolationDCOffset AudioSignalViolation = "dc-offset"
)

// AudioAdmissionOutcome states whether a decoded signal may enter downstream
// evaluation, training, or serving materialization.
type AudioAdmissionOutcome string

const (
	// AudioAdmissionAccepted admits the source for downstream use.
	AudioAdmissionAccepted AudioAdmissionOutcome = "accepted"
	// AudioAdmissionQuarantined preserves the source but excludes it from use.
	AudioAdmissionQuarantined AudioAdmissionOutcome = "quarantined"
)

// AudioAdmissionDecision is the deterministic result of applying one policy.
type AudioAdmissionDecision struct {
	Outcome    AudioAdmissionOutcome  `json:"outcome"`
	Violations []AudioSignalViolation `json:"violations,omitzero"`
}

// Validate requires a canonical violation set consistent with the outcome.
func (decision AudioAdmissionDecision) Validate() error {
	if decision.Outcome != AudioAdmissionAccepted && decision.Outcome != AudioAdmissionQuarantined {
		return errors.New("audio admission: invalid outcome")
	}
	if !slices.IsSorted(decision.Violations) || hasDuplicateAudioViolations(decision.Violations) {
		return errors.New("audio admission: violations are not canonical")
	}
	for _, violation := range decision.Violations {
		switch violation {
		case AudioViolationSchemaInvalid, AudioViolationDecodeIncomplete,
			AudioViolationDurationBelowMinimum, AudioViolationDurationAboveMaximum,
			AudioViolationChannelsBelowMinimum, AudioViolationChannelsAboveMaximum,
			AudioViolationNonFiniteSamples, AudioViolationSilent, AudioViolationClipped, AudioViolationDCOffset:
		default:
			return errors.New("audio admission: invalid violation")
		}
	}
	if decision.Outcome == AudioAdmissionAccepted && len(decision.Violations) != 0 ||
		decision.Outcome == AudioAdmissionQuarantined && len(decision.Violations) == 0 {
		return errors.New("audio admission: outcome differs from violations")
	}
	return nil
}

// Evaluate applies this policy to exact decoded-signal evidence.
func (policy AudioAdmissionPolicy) Evaluate(profile DecodedAudioSignalProfile) (AudioAdmissionDecision, error) {
	if err := policy.Validate(); err != nil {
		return AudioAdmissionDecision{}, err
	}
	if err := profile.Validate(); err != nil {
		return AudioAdmissionDecision{}, err
	}
	var violations []AudioSignalViolation
	if !profile.SchemaValid {
		violations = append(violations, AudioViolationSchemaInvalid)
	}
	if profile.DecodeStatus != AudioDecodeComplete {
		violations = append(violations, AudioViolationDecodeIncomplete)
	} else {
		duration, _ := profile.DurationNanoseconds()
		if duration < policy.MinimumDurationNanoseconds {
			violations = append(violations, AudioViolationDurationBelowMinimum)
		}
		if policy.MaximumDurationNanoseconds != 0 && duration > policy.MaximumDurationNanoseconds {
			violations = append(violations, AudioViolationDurationAboveMaximum)
		}
		if profile.Format.Channels < policy.MinimumChannels {
			violations = append(violations, AudioViolationChannelsBelowMinimum)
		}
		if profile.Format.Channels > policy.MaximumChannels {
			violations = append(violations, AudioViolationChannelsAboveMaximum)
		}
		if profile.NonFiniteSampleCount > policy.MaximumNonFiniteSamples {
			violations = append(violations, AudioViolationNonFiniteSamples)
		}
		if profile.RootMeanSquare <= policy.SilenceRMSThreshold {
			violations = append(violations, AudioViolationSilent)
		}
		if profile.FiniteSampleCount == 0 ||
			float64(profile.ClippedSampleCount)/float64(profile.FiniteSampleCount) > policy.MaximumClippedFraction {
			violations = append(violations, AudioViolationClipped)
		}
		if math.Abs(profile.DCOffset) > policy.MaximumAbsoluteDCOffset {
			violations = append(violations, AudioViolationDCOffset)
		}
	}
	slices.Sort(violations)
	outcome := AudioAdmissionAccepted
	if len(violations) != 0 {
		outcome = AudioAdmissionQuarantined
	}
	return AudioAdmissionDecision{Outcome: outcome, Violations: violations}, nil
}

func hasDuplicateAudioViolations(violations []AudioSignalViolation) bool {
	for index := 1; index < len(violations); index++ {
		if violations[index] == violations[index-1] {
			return true
		}
	}
	return false
}
