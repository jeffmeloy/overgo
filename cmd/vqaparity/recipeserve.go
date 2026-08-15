package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/cuda/device"
	"overgo/internal/cuda/executor"
	"overgo/internal/dataroot"
	"overgo/internal/hfbpe"
	"overgo/internal/patchtower"
	"overgo/internal/recipe"
	"overgo/internal/repodb"
	"overgo/internal/routedlm"
)

// decodeChain: token ids -> text via the checkpoint tokenizer.
func decodeChain(modelDir string, ids []int) (string, error) {
	tok, err := hfbpe.Load(modelDir)
	if err != nil {
		return "", err
	}
	return tok.Decode(ids), nil
}

// readEOSTokenIDs: scalar or list generation stop tokens.
func readEOSTokenIDs(modelDir string) ([]int, error) {
	raw, err := os.ReadFile(filepath.Join(modelDir, "generation_config.json"))
	if err != nil {
		return nil, err
	}
	var parsed struct {
		EOS json.RawMessage `json:"eos_token_id"`
	}
	if err := json.Unmarshal(raw, &parsed); err != nil {
		return nil, fmt.Errorf("generation_config: %w", err)
	}
	if len(parsed.EOS) == 0 {
		return nil, fmt.Errorf("generation_config: no eos_token_id")
	}
	var list []int
	if err := json.Unmarshal(parsed.EOS, &list); err == nil {
		return list, nil
	}
	var scalar int
	if err := json.Unmarshal(parsed.EOS, &scalar); err != nil {
		return nil, fmt.Errorf("generation_config: eos_token_id shape: %w", err)
	}
	return []int{scalar}, nil
}

// openRecipeStore: writable workflow lineage store.
func openRecipeStore(repo string) (*repodb.Store, error) {
	repository := repo
	if repository == "" {
		roots, err := dataroot.ResolveCurrent()
		if err != nil {
			return nil, err
		}
		repository = roots.Store
	}
	return repodb.Open(repository)
}

type preparedVQA struct {
	pixels              []float32
	inputIDs, positions []int
	eosIDs              []int
	gridT, gridH, gridW int
}

func prepareVQA(modelDir string, image []byte, question string) (preparedVQA, error) {
	pre, err := patchtower.LoadPreprocessConfig(modelDir)
	if err != nil {
		return preparedVQA{}, err
	}
	tok, err := hfbpe.Load(modelDir)
	if err != nil {
		return preparedVQA{}, err
	}
	specials, err := routedlm.LoadPromptSpecials(modelDir, promptRoles)
	if err != nil {
		return preparedVQA{}, err
	}
	rgb, height, width, err := patchtower.DecodeImageBytesRGB(image)
	if err != nil {
		return preparedVQA{}, err
	}
	pixels, gridT, gridH, gridW, err := patchtower.PreprocessImage(pre, rgb, height, width)
	if err != nil {
		return preparedVQA{}, err
	}
	inputIDs, err := routedlm.RenderVisionQAPrompt(tok, specials, question, gridH, gridW, pre.MergeSize)
	if err != nil {
		return preparedVQA{}, err
	}
	eosIDs, err := readEOSTokenIDs(modelDir)
	if err != nil {
		return preparedVQA{}, err
	}
	return preparedVQA{
		pixels: pixels, inputIDs: inputIDs, positions: routedlm.ImageMaskPositions(inputIDs, specials),
		eosIDs: eosIDs, gridT: gridT, gridH: gridH, gridW: gridW,
	}, nil
}

type vqaExecution struct {
	chain  []int
	text   string
	result fullResult
}

func executeVQAProgram(
	l *ladder,
	ctx context.Context,
	store artifact.Repository,
	modelID artifact.ID,
	program recipe.Program,
	imagePath, question, tag string,
) (vqaExecution, error) {
	if _, statErr := os.Stat(imagePath); statErr != nil {
		return vqaExecution{}, fmt.Errorf("serve image not found: %s", imagePath)
	}
	rawImg, err := os.ReadFile(imagePath)
	if err != nil {
		return vqaExecution{}, err
	}
	const maxSteps = 64

	var execution vqaExecution
	var pipelineErr error
	answer, err := executeVQA(
		ctx, store, modelID, program, "recipe/vqa/"+strings.ToLower(tag), rawImg, question,
		func(image []byte, question string) (preparedVQA, error) {
			prepared, prepareErr := prepareVQA(l.modelDir, image, question)
			if prepareErr == nil {
				l.log(fmt.Sprintf("RECIPE serve %s PROCESSOR image=%s grid=[%d,%d,%d] promptLen=%d imageTokens=%d question=%q",
					tag, filepath.Base(imagePath), prepared.gridT, prepared.gridH, prepared.gridW,
					len(prepared.inputIDs), len(prepared.positions), question))
			}
			return prepared, prepareErr
		},
		func(
			executeContext context.Context,
			prepared preparedVQA,
		) (string, error) {
			pc, openErr := newPrefillContext(
				l.modelDir, prepared.inputIDs, prepared.positions,
				prepared.gridT, prepared.gridH, prepared.gridW,
			)
			if openErr != nil {
				return "", openErr
			}
			defer pc.src.Close()
			worker, openErr := device.New(0)
			if openErr != nil {
				return "", fmt.Errorf("recipe serve worker: %w", openErr)
			}
			defer worker.Close()
			exe, openErr := executor.NewWithWorker(worker)
			if openErr != nil {
				return "", fmt.Errorf("recipe serve executor: %w", openErr)
			}
			defer exe.Close()
			execution.result, pipelineErr = runFullPipeline(
				l, executeContext, worker, exe, pc, prepared.pixels,
				fullOpts{maxSteps: maxSteps, eosIDs: prepared.eosIDs},
			)
			if pipelineErr != nil {
				return "", pipelineErr
			}
			execution.chain = execution.result.generated
			execution.text, pipelineErr = decodeChain(l.modelDir, execution.chain)
			return execution.text, pipelineErr
		},
	)
	if err != nil {
		return vqaExecution{}, err
	}
	execution.text = answer
	l.log(fmt.Sprintf("RECIPE serve %s chain=%v", tag, execution.chain))
	l.log(fmt.Sprintf("RECIPE serve %s TEXT %q", tag, execution.text))
	l.log(fmt.Sprintf("RECIPE serve %s MEASURE e2e=%s decode=%s (%d steps, %.3f ms/token)",
		tag, execution.result.e2eWall.Round(time.Millisecond), execution.result.decodeWall.Round(time.Millisecond),
		execution.result.steps, float64(execution.result.decodeWall.Microseconds())/1000.0/float64(max1(execution.result.steps))))
	return execution, nil
}

func validateCanonicalVQA(l *ladder, execution vqaExecution, tag string) error {
	dg, err := loadGoldenJSON[decodeStepsGolden](l.fixturesDir, "rxbrain_vqa_decode_steps_golden.json")
	if err != nil {
		return err
	}
	if len(execution.chain) < len(dg.GeneratedTokens) {
		return fmt.Errorf("recipe serve chain len %d < golden prefix %d", len(execution.chain), len(dg.GeneratedTokens))
	}
	if err := requireIntSliceEqual("recipe serve golden-prefix chain", execution.chain[:len(dg.GeneratedTokens)], dg.GeneratedTokens); err != nil {
		return err
	}
	const wantPrefix = "The stovetop holds a metal pot on the left burner"
	if !strings.HasPrefix(execution.text, wantPrefix) {
		return fmt.Errorf("recipe serve answer %q does not start with %q", execution.text, wantPrefix)
	}
	l.log(fmt.Sprintf("RECIPE serve %s EXACT golden-prefix (%d tokens) + phrase; full answer %d tokens, first=%d e2e=%s",
		tag, len(dg.GeneratedTokens), len(execution.chain), execution.chain[0], execution.result.e2eWall.Round(time.Millisecond)))
	return nil
}

// runRecipeServe: canonical case through the active recipe.
func runRecipeServe(l *ladder, repo, imagePath, question string) error {
	ctx := context.Background()
	store, err := openRecipeStore(repo)
	if err != nil {
		return err
	}
	defer store.Close()
	modelID, program, err := resolveActiveVQA(ctx, store, l.modelDir)
	if err != nil {
		return err
	}
	recipeID := program.Definition().ID.String()
	l.log(fmt.Sprintf("RECIPE serve active vqa recipe resolved model=%s recipe=%s", modelID, recipeID))

	first, err := executeVQAProgram(l, ctx, store, modelID, program, imagePath, question, "RUN1")
	if err != nil {
		return err
	}
	if err := validateCanonicalVQA(l, first, "RUN1"); err != nil {
		return err
	}

	// Second pass: bit-exact determinism.
	second, err := executeVQAProgram(l, ctx, store, modelID, program, imagePath, question, "RUN2")
	if err != nil {
		return err
	}
	if err := requireIntSliceEqual("recipe serve determinism chain", second.chain, first.chain); err != nil {
		return err
	}
	if second.text != first.text {
		return fmt.Errorf("recipe serve RUN2 answer %q != RUN1 %q", second.text, first.text)
	}
	l.log(fmt.Sprintf("RECIPE serve DETERMINISTIC RUN1==RUN2 chain (%d tokens) e2e1=%s e2e2=%s",
		len(first.chain), first.result.e2eWall.Round(time.Millisecond), second.result.e2eWall.Round(time.Millisecond)))
	l.log(fmt.Sprintf("RECIPE serve ANSWER %q", first.text))
	l.log("RECIPE serve LANE GREEN (served THROUGH activated recipe " + recipeID + ")")
	return nil
}

func max1(n int) int {
	if n < 1 {
		return 1
	}
	return n
}
