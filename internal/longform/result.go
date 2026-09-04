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
}

// Result is the evidence document one run commits: the model, the code
// it ran on, the prompt source, the measure, the short-prompt rates it
// was bounded against, the floors, the verdict, and the output text so
// a failed verdict can be read.
type Result struct {
	ModelPath    string `json:"model_path"`
	ModelName    string `json:"model_name"`
	Architecture string `json:"architecture"`
	FileType     string `json:"file_type"`
	Commit       string `json:"commit"`
	// Surface is the inference code surface digest the run measured
	// (see Surface); the admission keys on it, the commit is provenance.
	Surface      string     `json:"surface"`
	PromptSource string     `json:"prompt_source"`
	Measure      Measure    `json:"measure"`
	Short        ShortRates `json:"short_prompt_rates"`
	Floors       Floors     `json:"floors"`
	Verdict      Verdict    `json:"verdict"`
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

// Claim shapes the result into one capability-measured claim over its
// own committed evidence bytes: the context is the prompt length and the
// wall is the prompt and decode time the run measured.
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
			locations, err := store.Locations(ctx, record.Model)
			if err != nil {
				return nil
			}
			for _, location := range locations {
				if location.Kind != artifact.LocationFile {
					continue
				}
				key := locationKey(location.Value)
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
	rerun := fmt.Sprintf("run: go run ./cmd/longform -publish %q", location)
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
