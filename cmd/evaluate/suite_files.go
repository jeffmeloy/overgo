package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"

	"overgo/internal/dataset"
	"overgo/internal/evaluation"
)

// suiteFileMode is the permission of a suite file written for a run.
const suiteFileMode = 0o600

// defaultSeriesContext and defaultSeriesHorizon are the readings a series
// window shows and asks for unless a run names others: a context of sixteen
// patches of thirty-two and a horizon every forecast model offers.
const (
	defaultSeriesContext = 512
	defaultSeriesHorizon = 64
)

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

// writeSeriesForecast builds the series-forecast suite of an ETT file and
// writes it to out.
func writeSeriesForecast(csv, out string, context, horizon int) error {
	suite, err := evaluation.BuildETTSeriesForecast(evaluation.SeriesForecastSpec{CSV: csv, Context: context, Horizon: horizon})
	if err != nil {
		return err
	}
	return writeSuite(out, suite, len(suite.Cases), "series-forecast")
}

// passageLines bounds the lines read from a split's text column, far above
// the few thousand a WikiText-2 split holds.
const passageLines = 1 << 20

// writeTextSuite builds a suite of a WikiText-2 test parquet file's text
// column, its passages or the starts of its articles, and writes it to out.
func writeTextSuite(ctx context.Context, parquet, out, measure string, build func(lines []string) (any, int, error)) error {
	var lines []string
	if err := dataset.ReadParquetTextRows(ctx, parquet, "text", passageLines, func(_ uint64, line string) error {
		lines = append(lines, line)
		return nil
	}); err != nil {
		return err
	}
	suite, cases, err := build(lines)
	if err != nil {
		return err
	}
	return writeSuite(out, suite, cases, measure)
}

// tableBytes bounds what reading a table's rows may decode at once, far above
// the few megabytes a tabular benchmark's table holds, and
// requestDirectoryMode is the permission of the directory its requests are
// written to.
const (
	tableBytes           = 1 << 30
	requestDirectoryMode = 0o700
)

// writeTabularRequests builds the classification requests of the credit-g
// parquet table and writes each to directory, numbered in order.
func writeTabularRequests(ctx context.Context, parquet, directory string) error {
	leaves, err := dataset.ParquetLeaves(parquet)
	if err != nil {
		return err
	}
	names := make([]string, len(leaves))
	columns := make([]evaluation.TableColumn, len(leaves))
	for index, leaf := range leaves {
		names[index], columns[index] = leaf.Name, evaluation.TableColumn{Name: leaf.Name, Integer: leaf.Integer}
	}
	rows, err := dataset.OpenParquetRows(ctx, parquet, names, tableBytes)
	if err != nil {
		return err
	}
	defer rows.Close()
	for row := range rows.Rows() {
		cells, err := rows.Read(ctx, row)
		if err != nil {
			return err
		}
		for index, name := range names {
			if cells[name] == nil {
				return fmt.Errorf("evaluate: row %d of %s has no %s", row, parquet, name)
			}
			columns[index].Cells = append(columns[index].Cells, *cells[name])
		}
	}
	requests, err := evaluation.BuildCreditGRequests(columns)
	if err != nil {
		return err
	}
	if err := os.MkdirAll(directory, requestDirectoryMode); err != nil {
		return err
	}
	for index, request := range requests {
		if err := os.WriteFile(filepath.Join(directory, fmt.Sprintf("request-%02d.json", index)), request, suiteFileMode); err != nil {
			return err
		}
	}
	fmt.Printf("tabular requests %s: %d requests from %d rows\n", directory, len(requests), rows.Rows())
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
