package longform

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"path/filepath"
	"strings"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/cuda/driver"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/runrecord"
)

// Capability names the verification claim a long-form run commits.
const Capability = "long-form"

// Measure is what one run measured: the prompt and output lengths, the
// two rates, and the degeneration measures over the output.
type Measure struct {
	PromptTokens          int     `json:"prompt_tokens"`
	OutputTokens          int     `json:"output_tokens"`
	StoppedEarly          bool    `json:"stopped_early"`
	PromptMilliseconds    float64 `json:"prompt_ms"`
	DecodeMilliseconds    float64 `json:"decode_ms"`
	PromptTokensPerSecond float64 `json:"prompt_tokens_per_second"`
	DecodeTokensPerSecond float64 `json:"decode_tokens_per_second"`
	DistinctFourGramRatio float64 `json:"distinct_four_gram_ratio"`
	LongestRepeatedSpan   int     `json:"longest_repeated_span"`
	// Score is the teacher-forced read of the true continuation; zero
	// ScoreTokens means the run did not score.
	Score ContextScore `json:"score"`
	// Execution counts the device work the generation cost per output
	// token; a rung that collapses shows which resource exploded.
	Execution Execution `json:"execution"`
	// Memory is library-owned allocation accounting after generation and scoring.
	// PeakBytes is cumulative since model open, not total physical device usage.
	Memory driver.MemoryStats `json:"memory,omitzero"`
}

// Execution is the device work one generation cost, per output token:
// kernel launches, stream synchronizations, host-to-device bytes, and
// graph instantiations and launches.
type Execution struct {
	KernelLaunchesPerToken       float64 `json:"kernel_launches_per_token"`
	SynchronizationsPerToken     float64 `json:"synchronizations_per_token"`
	HostToDeviceBytesPerToken    float64 `json:"host_to_device_bytes_per_token"`
	GraphInstantiationsPerToken  float64 `json:"graph_instantiations_per_token"`
	GraphLaunchesPerToken        float64 `json:"graph_launches_per_token"`
	DeviceToHostBytesPerToken    float64 `json:"device_to_host_bytes_per_token"`
	PromptKernelLaunches         uint64  `json:"prompt_kernel_launches"`
	PromptHostToDeviceBytes      uint64  `json:"prompt_host_to_device_bytes"`
	PromptGraphInstantiations    uint64  `json:"prompt_graph_instantiations"`
	PromptStreamSynchronizations uint64  `json:"prompt_stream_synchronizations"`
}

// Result is the evidence document one run commits: the model, the code
// it ran on, the prompt source, the measure, the short-prompt rates it
// was bounded against, the floors, the verdict, and the output text so
// a failed verdict can be read.
type Result struct {
	Program       modelrecipe.ProgramIdentity `json:"program,omitzero"`
	Device        driver.DeviceInfo           `json:"device,omitzero"`
	ContextLength uint32                      `json:"context_length,omitzero"`
	Inputs        Inputs                      `json:"inputs,omitzero"`
	BudgetNS      int64                       `json:"budget_ns,omitzero"`
	ModelBudgetNS int64                       `json:"model_budget_ns,omitzero"`
	WallNS        int64                       `json:"wall_ns,omitzero"`
	ModelPath     string                      `json:"model_path"`
	ModelName     string                      `json:"model_name"`
	Architecture  string                      `json:"architecture"`
	FileType      string                      `json:"file_type"`
	Commit        string                      `json:"commit"`
	// Surface is the inference code surface digest the run measured
	// (see Surface); the admission keys on it, the commit is provenance.
	Surface      string `json:"surface"`
	PromptSource string `json:"prompt_source"`
	// Measure is the judged rung of the ladder, the one at the floors'
	// prompt length.
	Measure Measure    `json:"measure"`
	Short   ShortRates `json:"short_prompt_rates"`
	Floors  Floors     `json:"floors"`
	Verdict Verdict    `json:"verdict"`
	// Shape is the SHORT fingerprint, Rungs the LONG ladder the run
	// climbed, LadderStop why it stopped before the last planned rung.
	Shape      ShortShape `json:"shape"`
	Rungs      []Rung     `json:"rungs,omitempty"`
	LadderStop string     `json:"ladder_stop,omitzero"`
	// PromptTail is the text of the prompt's last tokens, so the output
	// is read against what it continues.
	PromptTail string `json:"prompt_tail"`
	Output     string `json:"output"`
}

// Summary is the latest committed long-form record for one weights
// location, as the report and the suite admission read it.
type Summary struct {
	Record artifact.ID
	Result Result
}

// ReadBaseline resolves an explicit passing record and verifies its checkpoint binding.
// Legacy admission records remain readable but cannot establish input-safe regression evidence.
func ReadBaseline(ctx context.Context, store *overgodb.Store, id artifact.ID) (Summary, error) {
	content, found, err := artifact.ReadContent(ctx, store, id)
	if err != nil {
		return Summary{}, err
	}
	if !found {
		return Summary{}, fmt.Errorf("longform: baseline %s is absent", id)
	}
	record, err := runrecord.ParseModelVerification(content.Data)
	if err != nil {
		return Summary{}, err
	}
	if record.ID != id {
		return Summary{}, errors.New("longform: baseline record identity differs")
	}
	for _, claim := range record.Claims {
		if claim.Capability != Capability || len(claim.Evidence) != 1 {
			continue
		}
		content, found, err := artifact.ReadContent(ctx, store, claim.Evidence[0])
		if err != nil {
			return Summary{}, err
		}
		if !found {
			return Summary{}, errors.New("longform: baseline measurement is absent")
		}
		var result Result
		if err := json.Unmarshal(content.Data, &result); err != nil {
			return Summary{}, err
		}
		if !result.Inputs.valid() || result.Inputs.Model != record.Model || result.Commit != claim.Commit || result.Surface == "" {
			return Summary{}, errors.New("longform: baseline lacks matching model, input or provenance identity")
		}
		if !result.Verdict.Passed || len(result.Verdict.Reasons) != 0 || len(result.Rungs) == 0 || len(result.Shape.OutputIDs) == 0 {
			return Summary{}, errors.New("longform: baseline is failed or incomplete")
		}
		return Summary{Record: id, Result: result}, nil
	}
	return Summary{}, errors.New("longform: baseline has no long-form measurement")
}

// Claim shapes the result into one capability-measured claim over its
// own committed evidence bytes: the context is the prompt length and the
// wall covers the complete model measurement; legacy records use judged-rung time.
func Claim(result Result) (runrecord.CapabilityClaim, []byte, artifact.ID, error) {
	evidenceData, err := json.Marshal(result)
	if err != nil {
		return runrecord.CapabilityClaim{}, nil, artifact.ID{}, err
	}
	evidence, _, err := artifact.Identify(artifact.KindEvidence, bytes.NewReader(evidenceData))
	if err != nil {
		return runrecord.CapabilityClaim{}, nil, artifact.ID{}, err
	}
	wall := time.Duration((result.Measure.PromptMilliseconds + result.Measure.DecodeMilliseconds) * float64(time.Millisecond))
	if result.WallNS > 0 {
		wall = time.Duration(result.WallNS)
	}
	return runrecord.CapabilityClaim{
		Capability: Capability, Tier: runrecord.TierCapabilityMeasured, Commit: result.Commit,
		ContextTokens: uint64(result.Measure.PromptTokens),
		WallNS:        uint64(wall.Nanoseconds()),
		Evidence:      []artifact.ID{evidence},
	}, evidenceData, evidence, nil
}

// Publish commits the result as a verification record over the model
// identity, through the same landing every claim path shares.
func Publish(ctx context.Context, store artifact.Repository, model artifact.ID, result Result) (artifact.ID, error) {
	claim, evidenceData, evidence, err := Claim(result)
	if err != nil {
		return artifact.ID{}, err
	}
	record, err := runrecord.NewModelVerification(model, result.ModelName, []runrecord.CapabilityClaim{claim})
	if err != nil {
		return artifact.ID{}, err
	}
	if err := runrecord.CommitVerificationClaim(ctx, store, record, evidenceData, evidence, result.ModelPath); err != nil {
		return artifact.ID{}, err
	}
	return record.ID, nil
}

// LatestByLocation reads the newest committed long-form record per
// weights location. A limit of zero visits the whole history.
func LatestByLocation(ctx context.Context, store *overgodb.Store, limit int) map[string]Summary {
	latest := map[string]Summary{}
	if store == nil {
		return latest
	}
	_, _ = overgodb.VisitDecodedDocuments(ctx, store, overgodb.DocumentQuery{
		Contracts: []artifact.DocumentContract{{
			Kind: artifact.KindEvidence, MediaType: runrecord.ModelVerificationMediaType, Schema: runrecord.ModelVerificationSchema,
		}},
		Order: overgodb.DocumentNewestFirst, MaxResults: limit,
	}, runrecord.ParseModelVerification, func(_ overgodb.DocumentView, record runrecord.ModelVerification) error {
		for _, claim := range record.Claims {
			if claim.Capability != Capability || len(claim.Evidence) == 0 {
				continue
			}
			content, found, err := artifact.ReadContent(ctx, store, claim.Evidence[0])
			if err != nil || !found {
				return nil
			}
			var result Result
			if json.Unmarshal(content.Data, &result) != nil {
				return nil
			}
			// The record is keyed by the weights file the run opened, which
			// the evidence names, and by every file location the store
			// registers for the model: a claim bound to a catalog manifest
			// identity carries a directory location, and the passes look
			// models up by their weights file.
			keys := []string{}
			if result.ModelPath != "" {
				keys = append(keys, locationKey(result.ModelPath))
			}
			if locations, err := store.Locations(ctx, record.Model); err == nil {
				for _, location := range locations {
					if location.Kind == artifact.LocationFile {
						keys = append(keys, locationKey(location.Value))
					}
				}
			}
			for _, key := range keys {
				if _, seen := latest[key]; !seen {
					latest[key] = Summary{Record: record.ID, Result: result}
				}
			}
			return nil
		}
		return nil
	})
	return latest
}

// Key normalizes a weights path so a record registered with one
// spelling of the path meets a pass launched with another; it is the
// key LatestByLocation indexes by.
func Key(path string) string {
	return strings.ToLower(filepath.ToSlash(filepath.Clean(path)))
}

func locationKey(path string) string { return Key(path) }

// Latest reads the newest committed long-form record for one location.
func Latest(ctx context.Context, store *overgodb.Store, location string) (Summary, bool) {
	summary, found := LatestByLocation(ctx, store, 0)[Key(location)]
	return summary, found
}

// ErrRefused marks a suite admission the long-form evidence refuses.
var ErrRefused = errors.New("long-form verification refuses the suite pass")

// Admit decides whether a suite pass may run the model on this
// inference code surface (owner rule 2026-09-04): the latest long-form
// record must exist, must have run on the same surface, and must have
// passed its floors. The message names the run that lifts the refusal.
func Admit(ctx context.Context, store *overgodb.Store, location, surface string) error {
	rerun := fmt.Sprintf("run: go run ./cmd/longform -publish -budget <measured-duration> %q", location)
	summary, found := Latest(ctx, store, location)
	switch {
	case !found:
		return fmt.Errorf("%w: no long-form record for %s; %s", ErrRefused, location, rerun)
	case summary.Result.Surface != surface:
		return fmt.Errorf("%w: the long-form record for %s measured inference surface %.12s (commit %.12s), the code is surface %.12s; %s",
			ErrRefused, location, summary.Result.Surface, summary.Result.Commit, surface, rerun)
	case !summary.Result.Verdict.Passed:
		return fmt.Errorf("%w: %s failed its floors on this surface (%s); fix the runtime, then %s",
			ErrRefused, location, strings.Join(summary.Result.Verdict.Reasons, "; "), rerun)
	}
	return nil
}
