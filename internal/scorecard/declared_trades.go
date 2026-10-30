package scorecard

import (
	_ "embed"
	"fmt"
	"math"
	"slices"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/recipe"
	"overgo/internal/scorecard/metric"
	"overgo/internal/strictjson"
)

// declaredTradesJSON holds each cell a change made worse on purpose for what the
// change bought, at the level it was accepted: a declaration the scorecard
// reads, so the next run does not file the decline again (Gaps). A declaration
// in a commit message alone was filed again by the next run: d184ff653 declared
// Qwen3.8-27B's drafting memory and three-token throughput, and the run of
// 2026-10-30 filed both again. cmd/validate -scorecard -declare writes it.
//
//go:embed declared_trades.json
var declaredTradesJSON []byte

// DeclaredTradesFile is where the trades are kept, beneath the repository root.
const DeclaredTradesFile = "internal/scorecard/declared_trades.json"

// Trade is one declared worsening: the cell (its model, task, measure and
// contract, the identity a decline's key is made of), the level it was
// accepted at in its unit, and what the worsening bought, with the change that
// declared it. Name is the model's, for a reader.
type Trade struct {
	Model    artifact.ID `json:"model"`
	Name     string      `json:"name"`
	Task     recipe.Task `json:"task"`
	Column   metric.Name `json:"column"`
	Contract Contract    `json:"contract"`
	Level    float64     `json:"level"`
	Unit     string      `json:"unit"`
	Bought   string      `json:"bought"`
}

// key is the key of the decline the trade accepts.
func (trade Trade) key() string {
	return regressionKey(Row{Model: trade.Model, Task: trade.Task}, trade.Column, Cell{Contract: trade.Contract})
}

// DeclaredTrades reads the trades; each names one cell once and says all it holds.
func DeclaredTrades() ([]Trade, error) {
	return parseTrades(declaredTradesJSON)
}

// TradesDocument is the file the trades are kept in.
type TradesDocument struct {
	Trades []Trade `json:"trades"`
}

func parseTrades(data []byte) ([]Trade, error) {
	var document TradesDocument
	if err := strictjson.DecodeBytes(data, &document); err != nil {
		return nil, fmt.Errorf("scorecard: declared trades: %w", err)
	}
	seen := map[string]bool{}
	for _, trade := range document.Trades {
		said := func(text string) bool { return strings.TrimSpace(text) != "" }
		if !trade.Model.Valid() || !trade.Task.Valid() || !slices.Contains(metric.Columns, trade.Column) || seen[trade.key()] ||
			math.IsNaN(trade.Level) || math.IsInf(trade.Level, 0) || !said(trade.Name) || !said(trade.Unit) || !said(trade.Bought) {
			return nil, fmt.Errorf("scorecard: the declared trade of %q %s %s is repeated or incomplete", trade.Name, trade.Task, trade.Column)
		}
		seen[trade.key()] = true
	}
	return document.Trades, nil
}

// declaredTrades are the trades Gaps reads; a malformed file fails every
// package that reads the scorecard, as a test of it does.
var declaredTrades = func() []Trade {
	trades, err := DeclaredTrades()
	if err != nil {
		panic(err)
	}
	return trades
}()

// accepted is the trade that accepts a decline found by keys, if one does.
func accepted(trades []Trade, keys []string) (Trade, bool) {
	at := slices.IndexFunc(trades, func(trade Trade) bool { return slices.Contains(keys, trade.key()) })
	if at < 0 {
		return Trade{}, false
	}
	return trades[at], true
}

// Declare is the trade that accepts the cell key names, read from rows, at the
// value it holds now.
func Declare(rows []Row, key, bought string) (Trade, error) {
	row, column, cell, found := cellByKey(rows, key)
	if !found {
		return Trade{}, fmt.Errorf("scorecard: no measured cell carries the key %s", key)
	}
	trade := Trade{Model: row.Model, Name: row.Name, Task: row.Task, Column: column, Contract: cell.Contract, Level: cell.Value, Unit: cell.Unit, Bought: bought}
	if cell.Contract.Origin == originServing {
		trade.Contract.Scope = artifact.ID{}
	}
	return trade, nil
}
