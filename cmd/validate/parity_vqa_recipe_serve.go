package main

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/dataroot"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/vqaserve"
	"overgo/internal/workflowruntime"
)

// openRecipeStore: writable workflow lineage store.
func openRecipeStore(repo string) (*overgodb.Store, error) {
	repository := repo
	if repository == "" {
		roots, err := dataroot.ResolveCurrent()
		if err != nil {
			return nil, err
		}
		repository = roots.Store
	}
	return overgodb.Open(repository)
}

type vqaExecution struct {
	chain  []int
	text   string
	result vqaserve.Result
	walls  []workflowruntime.NodeWall
}

func executeVQAProgram(
	l *campaignContext, ctx context.Context,
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
	image, err := vqaserve.ImageContract.ContentBytes(rawImg)
	if err != nil {
		return vqaExecution{}, err
	}
	var execution vqaExecution
	answer, walls, err := vqaserve.RunProgram(
		ctx, store, modelID, program, "recipe/vqa/"+strings.ToLower(tag), image, question,
		func(image []byte, question string) (vqaserve.Prepared, error) {
			prepared, prepareErr := vqaserve.Prepare(l.modelDir, image, question)
			if prepareErr == nil {
				l.Log(fmt.Sprintf("RECIPE serve %s PROCESSOR image=%s grid=[%d,%d,%d] promptLen=%d imageTokens=%d question=%q",
					tag, filepath.Base(imagePath), prepared.GridT, prepared.GridH, prepared.GridW,
					len(prepared.InputIDs), len(prepared.Positions), question))
			}
			return prepared, prepareErr
		},
		func(executeContext context.Context, prepared vqaserve.Prepared) (string, error) {
			result, text, answerErr := vqaserve.Answer(executeContext, l.modelDir, prepared, vqaserve.Policy.DecodeMaxSteps, l.Log)
			if answerErr != nil {
				return "", answerErr
			}
			execution.result, execution.chain = result, result.Generated
			return text, nil
		},
	)
	if err != nil {
		return vqaExecution{}, err
	}
	execution.text, execution.walls = answer, walls
	l.Log(fmt.Sprintf("RECIPE serve %s chain=%v", tag, execution.chain))
	l.Log(fmt.Sprintf("RECIPE serve %s TEXT %q", tag, execution.text))
	l.Log(fmt.Sprintf("RECIPE serve %s MEASURE e2e=%s decode=%s (%d steps, %.3f ms/token)",
		tag, execution.result.E2EWall.Round(time.Millisecond), execution.result.DecodeWall.Round(time.Millisecond),
		execution.result.Steps, float64(execution.result.DecodeWall.Microseconds())/1000.0/float64(max(execution.result.Steps, 1))))
	return execution, nil
}

func validateCanonicalVQA(l *campaignContext, execution vqaExecution, tag string) error {
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
	l.Log(fmt.Sprintf("RECIPE serve %s EXACT golden-prefix (%d tokens) + phrase; full answer %d tokens, first=%d e2e=%s",
		tag, len(dg.GeneratedTokens), len(execution.chain), execution.chain[0], execution.result.E2EWall.Round(time.Millisecond)))
	return nil
}

// runRecipeServe: canonical case through the active recipe.
func runRecipeServe(l *campaignContext, repo, imagePath, question string) error {
	ctx := context.Background()
	store, err := openRecipeStore(repo)
	if err != nil {
		return err
	}
	defer store.Close()
	modelID, program, err := vqaserve.ResolveActive(ctx, store, l.modelDir)
	if err != nil {
		return err
	}
	recipeID := program.Definition().ID.String()
	l.Log(fmt.Sprintf("RECIPE serve active vqa recipe resolved model=%s recipe=%s", modelID, recipeID))

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
	l.Log(fmt.Sprintf("RECIPE serve DETERMINISTIC RUN1==RUN2 chain (%d tokens) e2e1=%s e2e2=%s",
		len(first.chain), first.result.E2EWall.Round(time.Millisecond), second.result.E2EWall.Round(time.Millisecond)))
	l.Log(fmt.Sprintf("RECIPE serve ANSWER %q", first.text))
	l.Log("RECIPE serve LANE GREEN (served THROUGH activated recipe " + recipeID + ")")
	return nil
}
