package main

import (
	"context"
	"fmt"
	"io"
	"slices"
	"strconv"
	"strings"
	"time"

	"overgo/internal/gitauthority"
	"overgo/internal/landing"
)

// interactiveWorker names the landings whose cost names no worker: an interactive session's work, which the driver
// recorded no attempt of.
const interactiveWorker = "an interactive session"

// workerCost is what one worker's landings landed and cost: the steps, the landings, the attempts at them, the
// workers' wall and the gate's.
type workerCost struct {
	worker           string
	landings, steps  int
	attempts         int
	workerWall, gate time.Duration
}

// landingCosts reads the cost and benefit of each landing since the date from its trailers (landing's Cost line), by
// the worker that ran its last attempt, and counts the landings from before landings said their cost.
func landingCosts(ctx context.Context, root, since string) (costs []workerCost, unsaid int, err error) {
	fields := []string{landedRowTrailer, landing.WorkerAttemptsTrailer, landing.WorkerWallTrailer, landing.GateWallTrailer, landing.WorkerTrailer}
	var format strings.Builder
	format.WriteString("--format=%H")
	for _, field := range fields {
		format.WriteString("%x00%(trailers:key=" + field + ",valueonly,separator=%x2C)")
	}
	log, err := gitauthority.Query(ctx, root, "log", "--since="+since, "--grep="+landedRowTrailer+":", format.String()+"%x1e")
	if err != nil {
		return nil, 0, err
	}
	by := map[string]*workerCost{}
	var order []string
	for record := range strings.SplitSeq(string(log), "\x1e") {
		// The commit, then each field's trailers in the order fields names them.
		_, trailers, _ := strings.Cut(strings.TrimSpace(record), "\x00")
		values := strings.Split(trailers, "\x00")
		if len(values) != len(fields) {
			continue
		}
		read := func(field string) string { return values[slices.Index(fields, field)] }
		steps, attempts, workerWall := read(landedRowTrailer), read(landing.WorkerAttemptsTrailer), read(landing.WorkerWallTrailer)
		gate, workers := read(landing.GateWallTrailer), read(landing.WorkerTrailer)
		if strings.TrimSpace(steps) == "" {
			continue
		}
		if strings.TrimSpace(attempts) == "" {
			unsaid++
			continue
		}
		worker := interactiveWorker
		if named := strings.Split(strings.TrimSpace(workers), ","); named[0] != "" {
			// The last worker named ran the attempt that landed.
			worker = strings.TrimSpace(named[len(named)-1])
		}
		if by[worker] == nil {
			by[worker] = &workerCost{worker: worker}
			order = append(order, worker)
		}
		cost := by[worker]
		cost.landings++
		cost.steps += len(strings.Split(steps, ","))
		cost.attempts += atoi(attempts)
		cost.workerWall += time.Duration(atoi(workerWall)) * time.Second
		cost.gate += time.Duration(atoi(gate)) * time.Second
	}
	for _, worker := range order {
		costs = append(costs, *by[worker])
	}
	slices.SortStableFunc(costs, func(a, b workerCost) int { return b.steps - a.steps })
	return costs, unsaid, nil
}

// atoi is a trailer's whole number, zero where it holds none.
func atoi(value string) int {
	number, _ := strconv.Atoi(strings.TrimSpace(value))
	return number
}

// printCosts prints, for each worker, the steps its landings landed against what they cost, and how many landings
// before landings said their cost are not counted.
func printCosts(out io.Writer, costs []workerCost, unsaid int) {
	for _, cost := range costs {
		fmt.Fprintf(out, "%s: landed %d steps in %d landings, %d worker attempts in %s of their wall, the gate %s\n",
			cost.worker, cost.steps, cost.landings, cost.attempts, cost.workerWall, cost.gate)
	}
	if unsaid > 0 {
		fmt.Fprintf(out, "%d landings from before landings said their cost are not counted\n", unsaid)
	}
}
