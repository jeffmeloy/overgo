package main

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"

	"overgo/internal/evaluation"
)

// suiteFileMode is the permission of a suite file written for a run.
const suiteFileMode = 0o600

// The files of the leaderboard's simple category a directory of it holds.
const (
	bfclQuestionsFile = "BFCL_v3_simple.json"
	bfclAnswersFile   = "possible_answer/BFCL_v3_simple.json"
)

// writeSuite writes a built suite to out for a run to evaluate.
func writeSuite(out string, suite any, cases int, what string) error {
	data, err := json.Marshal(suite)
	if err != nil {
		return err
	}
	if err := os.WriteFile(out, data, suiteFileMode); err != nil {
		return err
	}
	fmt.Printf("%s suite %s: %d cases\n", what, out, cases)
	return nil
}

// writeFunctionCalls builds the function-call suite of the leaderboard's
// simple category held in directory and writes it to out.
func writeFunctionCalls(directory, out string) error {
	suite, err := evaluation.BuildBFCLFunctionCalls(evaluation.FunctionCallSpec{
		Questions: filepath.Join(directory, bfclQuestionsFile), Answers: filepath.Join(directory, filepath.FromSlash(bfclAnswersFile)),
	})
	if err != nil {
		return err
	}
	return writeSuite(out, suite, len(suite.Cases), "function-call")
}
