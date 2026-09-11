//overgo:runtime-inputs caller

// reverify re-establishes one model's activation after a document schema
// migration: it locates the model's stored exact-golden inference suite from
// its claims manifest, replays it through `recipe verify` (real inference,
// fresh gate and run evidence under the current schema), and activates the
// re-derived recipe with that evidence.
//
// The claims manifest names evidence artifacts; the suite is the one whose
// content carries cases. Verification and activation stay owned by cmd/recipe
// -- this command only sequences them, so the evidence chain is identical to
// running the steps by hand.
package main

import (
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"os"
	"os/exec"
	"strings"

	"overgo/internal/clioptions"
	"overgo/internal/recipe"
)

type claimsManifest struct {
	Claims []struct {
		Capability string   `json:"capability"`
		Tier       string   `json:"tier"`
		Evidence   []string `json:"evidence"`
	} `json:"claims"`
}

type verifyResult struct {
	GateID   string `json:"gate_id"`
	RecipeID string `json:"recipe_id"`
	ReportID string `json:"report_id"`
	RunID    string `json:"run_id"`
}

func main() {
	clioptions.MainNamed("reverify", run)
}

func run() error {
	repository := flag.String("repo", "overgodb-store", "OvergoDB store directory")
	claimsPath := flag.String("claims", "", "claims manifest naming the exact-golden inference evidence")
	reason := flag.String("reason", "", "activation reason recorded in the decision event")
	flag.Parse()
	if flag.NArg() != 1 || strings.TrimSpace(*claimsPath) == "" || strings.TrimSpace(*reason) == "" {
		return errors.New("usage: reverify -claims <manifest.json> -reason <text> [-repo <path>] <model>")
	}
	suite, evidenceID, err := goldenSuite(*repository, *claimsPath)
	if err != nil {
		return err
	}
	fmt.Printf("golden %s\n", evidenceID)
	for _, modelPath := range flag.Args() {
		verified, err := verify(*repository, modelPath, suite)
		if err != nil {
			return err
		}
		fmt.Printf("verified gate=%s run=%s report=%s\n", verified.GateID, verified.RunID, verified.ReportID)
		if err := activate(*repository, modelPath, *reason, verified); err != nil {
			return err
		}
	}
	return nil
}

// goldenSuite returns the exact-golden suite content and the evidence artifact
// that carries it: the first inference-claim evidence whose content decodes as
// a suite with cases.
func goldenSuite(repository, claimsPath string) ([]byte, string, error) {
	raw, err := os.ReadFile(claimsPath)
	if err != nil {
		return nil, "", err
	}
	var manifest claimsManifest
	if err := json.Unmarshal(raw, &manifest); err != nil {
		return nil, "", fmt.Errorf("reverify: decode claims manifest: %w", err)
	}
	var lastErr error
	for _, claim := range manifest.Claims {
		if claim.Capability != string(recipe.TaskInference) {
			continue
		}
		for _, evidence := range claim.Evidence {
			content, err := command("go", "run", "./cmd/overgodb-query", "-repo", repository, "-id", evidence, "-content")
			if err != nil {
				lastErr = err
				continue
			}
			normalized, ok := normalizeSuite(content)
			if ok {
				return normalized, evidence, nil
			}
		}
	}
	if lastErr != nil {
		return nil, "", fmt.Errorf("reverify: no readable exact-golden suite in claims evidence: %w", lastErr)
	}
	return nil, "", errors.New("reverify: claims manifest names no exact-golden inference suite")
}

// normalizeSuite accepts the stored golden dialects. Some goldens record
// explicit token counts; older ones record the exact prompt and generated id
// sequences instead. The counts are the lengths of those sequences, so the
// derivation is lossless and stays inside the same evidence -- no new golden
// is invented here.
func normalizeSuite(content []byte) ([]byte, bool) {
	var suite map[string]any
	if json.Unmarshal(content, &suite) != nil {
		return nil, false
	}
	rawCases, _ := suite["cases"].([]any)
	if len(rawCases) == 0 {
		return nil, false
	}
	for _, entry := range rawCases {
		testCase, ok := entry.(map[string]any)
		if !ok {
			return nil, false
		}
		promptIDs, _ := testCase["prompt_ids"].([]any)
		generatedIDs, _ := testCase["generated_ids"].([]any)
		if _, counted := testCase["prompt_tokens"]; !counted && len(promptIDs) > 0 && len(generatedIDs) > 0 {
			testCase["prompt_tokens"] = len(promptIDs)
			testCase["generated_tokens"] = len(generatedIDs)
			testCase["max_tokens"] = len(generatedIDs)
			// This dialect's text carries prompt plus generation; the evaluator
			// compares generation alone, so the prompt prefix is removed. Both
			// halves come from the same stored case.
			prompt, _ := testCase["prompt"].(string)
			if text, ok := testCase["text"].(string); ok && prompt != "" && strings.HasPrefix(text, prompt) {
				testCase["text"] = strings.TrimPrefix(text, prompt)
			}
			delete(testCase, "prompt_ids")
			delete(testCase, "generated_ids")
		}
	}
	normalized, err := json.Marshal(suite)
	if err != nil {
		return nil, false
	}
	return normalized, true
}

func verify(repository, modelPath string, suite []byte) (verifyResult, error) {
	execution := exec.Command("go", "run", "./cmd/recipe", "verify", "-repo", repository, "-input", "-", modelPath)
	execution.Stdin = strings.NewReader(string(suite))
	output, err := execution.CombinedOutput()
	if err != nil {
		return verifyResult{}, fmt.Errorf("reverify: verify: %w\n%s", err, strings.TrimSpace(string(output)))
	}
	lines := strings.Split(strings.TrimSpace(string(output)), "\n")
	var result verifyResult
	if err := json.Unmarshal([]byte(lines[len(lines)-1]), &result); err != nil ||
		result.GateID == "" || result.RunID == "" {
		return verifyResult{}, fmt.Errorf("reverify: verify emitted no evidence identifiers:\n%s", strings.TrimSpace(string(output)))
	}
	return result, nil
}

func activate(repository, modelPath, reason string, verified verifyResult) error {
	output, err := command("go", "run", "./cmd/recipe", "activate", "-repo", repository,
		"-reason", reason, "-gate", verified.GateID, "-run-id", verified.RunID, modelPath)
	if err != nil {
		return fmt.Errorf("reverify: activate: %w", err)
	}
	fmt.Print(string(output))
	return nil
}

func command(name string, arguments ...string) ([]byte, error) {
	execution := exec.Command(name, arguments...)
	output, err := execution.CombinedOutput()
	if err != nil {
		return nil, fmt.Errorf("%s %s: %w\n%s", name, strings.Join(arguments, " "), err, strings.TrimSpace(string(output)))
	}
	return output, nil
}
