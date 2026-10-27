package inference

import (
	"errors"
	"fmt"
	"math"
	"slices"
	"time"

	"overgo/internal/cuda/executor"
	"overgo/internal/hostexpert"
	"overgo/internal/modeldevice"
)

// strataRoundPool is Strata's decode round's host pool (src/core/expert_source.cpp expert_pool_dispatch_multi, served
// through src/program/generate.cpp drive_pool_multi; Strata at 296b59d2, MIT) at the bench's configuration: the route
// tail skip off (STRATA_ROUTE_TAIL_SKIP=0), no peer or helper GPU, the PCIe share by the graph's copy kernel (pcie_mode
// 2). For each layer's doorbell it plans the window's distinct experts in routing order - the card's (VRAM groups),
// the last pcieNum/256 of the misses (PCIe groups, their blobs read over the link), the rest computed here - publishes
// the plan before the CPU works, then computes the CPU's rows; the GPU zeroes the rows it computes itself.
type strataRoundPool struct {
	layers []modeldevice.ExpertBlobLayer
	cpu    *hostexpert.Pool
	// pcieNum is the PCIe share of the misses in 256ths (drive.d.pcie_num).
	pcieNum int
	// lexical gathers a window's n-gram embedding rows (ss.ple.table->gather_batch).
	lexical func(tokens []int32, rows []float32) error
	// Each layer's CPU formats, and the scratch a layer's CPU share reuses.
	formats     []hostexpert.Formats
	values      [][]float32
	activations [][]byte
	work        []hostexpert.Expert
	jobExperts  []int32
	// first and last are each expert's first and last entry of the layer being planned, -1 between layers.
	first, last []int32
	parts       [][]byte
	// usage counts each expert's routes, [layer][expert], for the adaptive tier (drive.d.usage), nil where it is off;
	// runTime and misses are the CPU's run time and the experts it computed (ms_run, multi_misses), for PcieAuto.
	usage   []float32
	runTime time.Duration
	misses  int64
}

// The pool's widths: Flash-Next's hidden and expert widths, its routed experts a token, and the window's entries the
// tables hold (kMaxWindowEntries) and the PCIe share's most fetches a layer (dma_src[64]).
const (
	strataPoolHidden       = 2560
	strataPoolIntermediate = 640
	strataPoolRouted       = 10
	strataPoolMaxEntries   = 128
	strataPoolMaxFetches   = 64
	// strataPoolShareShift is the PCIe share's 256ths.
	strataPoolShareShift = 8
)

// The plan's counts (GpuPlanSink::counts): the VRAM groups, every GPU entry, the PCIe groups; a blob's parts; and the
// kinds of a window's entry (expert_pool_dispatch_multi's kind): the CPU's, the card's, the PCIe share's.
const (
	strataPlanGroups  = 0
	strataPlanEntries = 1
	strataPlanFetches = 2
	strataBlobGate    = 0
	strataBlobUp      = 1
	strataBlobDown    = 2
	strataKindCPU     = -1
	strataKindVRAM    = 0
	strataKindPCIe    = 1
)

func newStrataRoundPool(layers []modeldevice.ExpertBlobLayer, cpu *hostexpert.Pool, pcieNum int,
	lexical func(tokens []int32, rows []float32) error) (*strataRoundPool, error) {
	pool := &strataRoundPool{layers: layers, cpu: cpu, pcieNum: pcieNum, lexical: lexical, formats: make([]hostexpert.Formats, len(layers))}
	for index, layer := range layers {
		parts := layer.Parts()
		if len(parts) != len(expertBlobPartNames) {
			return nil, fmt.Errorf("strata round: layer %d's expert blob holds %d banks, want gate, up and down", index, len(parts))
		}
		formats := hostexpert.Formats{Gate: parts[strataBlobGate].Type, Up: parts[strataBlobUp].Type, Down: parts[strataBlobDown].Type}
		if !hostexpert.Supported(formats, strataPoolHidden, strataPoolIntermediate) {
			return nil, fmt.Errorf("strata round: layer %d's experts are in formats the host pool does not compute", index)
		}
		pool.formats[index] = formats
	}
	return pool, nil
}

// expertBlobPartNames are the blob's parts the pool reads, in order.
var expertBlobPartNames = []string{"gate", "up", "down"}

// ServeLayer is expert_pool_dispatch_multi for one layer of a window.
func (p *strataRoundPool) ServeLayer(layer int, x []float32, ids []int32, tokens int, plan *executor.StrataRoundPlan, out []float32) error {
	n := tokens * strataPoolRouted
	if n > strataPoolMaxEntries || int64(n) > plan.Cap || layer >= len(p.layers) {
		return errors.New("strata round: a verify window routes more entries than the expert pool's window tables hold")
	}
	blobs := p.layers[layer]
	experts := int32(blobs.Experts())
	if len(p.first) < int(experts) {
		p.first, p.last = make([]int32, experts), make([]int32, experts)
		for e := range p.first {
			p.first[e] = -1
		}
	}
	// The distinct experts in routing order, each with its blob, and each entry's next entry of its expert (chain): one
	// pass over the entries, a table of the layer's experts finding an expert's first.
	var kind, chain, distinct [strataPoolMaxEntries]int
	var blobOf [strataPoolMaxEntries]uint64
	var cardOf [strataPoolMaxEntries]bool
	nd, nmiss := 0, 0
	for i := range n {
		chain[i] = -1
		e := ids[i]
		if e >= 0 && e < experts && p.first[e] >= 0 {
			chain[p.last[e]] = i
			p.last[e] = int32(i)
			continue
		}
		// an id out of range stands alone, which the CPU share refuses
		blobOf[nd], cardOf[nd] = 0, false
		if e >= 0 && e < experts {
			p.first[e], p.last[e] = int32(i), int32(i)
			blob, onCard := blobs.Blob(int(e))
			blobOf[nd], cardOf[nd] = uint64(blob), onCard
			if !onCard {
				nmiss++
			}
		}
		distinct[nd] = i
		nd++
	}
	for _, i0 := range distinct[:nd] {
		if e := ids[i0]; e >= 0 && e < experts {
			p.first[e] = -1
		}
	}
	if p.usage != nil {
		for _, e := range ids[:n] {
			if e >= 0 && e < experts {
				p.usage[layer*int(experts)+int(e)]++
			}
		}
	}
	m := 0
	if p.pcieNum > 0 {
		m = (nmiss * p.pcieNum) >> strataPoolShareShift
	}
	missRank, groups, entries, fetches := 0, 0, 0, 0
	var pcieFirst [strataPoolMaxFetches]int
	enter := func(i0 int) {
		for i := i0; i >= 0; i = chain[i] {
			plan.Dst[entries], plan.Tok[entries] = int32(i), int32(i/strataPoolRouted)
			entries++
		}
	}
	for d, i0 := range distinct[:nd] {
		e := ids[i0]
		kd := strataKindCPU
		var pointer uint64
		if e >= 0 && e < experts {
			if cardOf[d] {
				kd, pointer = strataKindVRAM, blobOf[d]
			} else {
				// every host expert of the residency is pinned and mapped: its blob's device alias
				if missRank >= nmiss-m && int64(fetches) < plan.StagingCap && fetches < strataPoolMaxFetches {
					kd, pcieFirst[fetches] = strataKindPCIe, i0
					plan.PCIe[fetches] = blobOf[d]
					fetches++
				}
				missRank++
			}
		}
		for i := i0; i >= 0; i = chain[i] {
			kind[i] = kd
		}
		if kd != strataKindVRAM {
			continue
		}
		plan.Pointers[groups], plan.Start[groups] = pointer, int32(entries)
		enter(i0)
		groups++
	}
	plan.Start[groups] = int32(entries)
	for q := range fetches {
		plan.PCIeStart[q] = int32(entries)
		enter(pcieFirst[q])
	}
	plan.PCIeStart[fetches] = int32(entries)
	plan.Counts[strataPlanGroups], plan.Counts[strataPlanEntries], plan.Counts[strataPlanFetches] = int32(groups), int32(entries), int32(fetches)
	plan.Publish()
	// pcie_mode 2: no DMA, the graph's copy kernel reads the blobs; flag B rises now
	plan.Fetch(0)
	return p.cpuShare(layer, x, ids, tokens, kind[:n], out)
}

// cpuShare computes the rows the plan left to the CPU (kind -1), each distinct expert once over its tokens: the tokens'
// activations quantized, then the expert's gate, up, SwiGLU and down products into its entries' rows of out.
func (p *strataRoundPool) cpuShare(layer int, x []float32, ids []int32, tokens int, kind []int, out []float32) error {
	cpu := false
	for _, k := range kind {
		cpu = cpu || k == strataKindCPU
	}
	if !cpu {
		return nil
	}
	if len(p.values) < tokens {
		p.values, p.activations = make([][]float32, tokens), make([][]byte, tokens)
		for t := range tokens {
			p.activations[t] = make([]byte, hostexpert.Q8KBytes(strataPoolHidden))
		}
	}
	for t := range tokens {
		p.values[t] = x[t*strataPoolHidden : (t+1)*strataPoolHidden]
	}
	if err := p.cpu.QuantizeQ8K(p.values[:tokens], p.activations[:tokens]); err != nil {
		return err
	}
	blobs := p.layers[layer]
	// A job a distinct expert, its rows kept from layer to layer: a window's layer allocates nothing.
	p.work, p.jobExperts = p.work[:0], p.jobExperts[:0]
	for i, k := range kind {
		if k != strataKindCPU {
			continue
		}
		e := ids[i]
		job := slices.Index(p.jobExperts, e)
		if job < 0 {
			p.parts = blobs.Host(int(e), p.parts[:0])
			if len(p.parts) != len(expertBlobPartNames) {
				return fmt.Errorf("strata round: layer %d's expert %d has no host copy", layer, e)
			}
			job = len(p.work)
			if job == cap(p.work) {
				p.work = append(p.work, hostexpert.Expert{})
			}
			p.work = p.work[:job+1]
			p.work[job].Gate, p.work[job].Up, p.work[job].Down = p.parts[strataBlobGate], p.parts[strataBlobUp], p.parts[strataBlobDown]
			p.work[job].Activations, p.work[job].Out = p.work[job].Activations[:0], p.work[job].Out[:0]
			p.jobExperts = append(p.jobExperts, e)
		}
		t := i / strataPoolRouted
		p.work[job].Activations = append(p.work[job].Activations, p.activations[t])
		p.work[job].Out = append(p.work[job].Out, out[i*strataPoolHidden:(i+1)*strataPoolHidden])
	}
	started := time.Now()
	err := p.cpu.Layer(p.work, p.formats[layer], strataPoolHidden, strataPoolIntermediate, strataSiLUProduct)
	p.runTime += time.Since(started)
	p.misses += int64(len(p.work))
	return err
}

// strataSiLUProduct is the experts' SwiGLU: gate's SiLU times up.
func strataSiLUProduct(gate, up float32) float32 {
	return float32(gate/float32(1+math.Exp(-float64(gate)))) * up
}

// Lexical gathers the window's n-gram embedding rows.
func (p *strataRoundPool) Lexical(tokens []int32, rows []float32) error {
	if p.lexical == nil {
		return errors.New("strata round: no n-gram embedding table")
	}
	return p.lexical(tokens, rows)
}
