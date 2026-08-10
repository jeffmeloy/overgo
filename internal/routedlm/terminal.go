package routedlm

import (
	"encoding/binary"
	"fmt"
	"math"

	"overgo/internal/hostmath"
)

// terminalScratchBytes: streamed head chunk size (row-multiple enforced).
const terminalScratchBytes = 8 << 20

// TerminalFinalHidden: rmsnorm(layerOutput) * finalNorm — NOT bf16-rounded
// (reference terminal contract).
func TerminalFinalHidden(layerOutput []float32, cfg Config, terminal TerminalWeights) ([]float32, int, error) {
	d := cfg.HiddenSize
	if len(layerOutput) == 0 || len(layerOutput)%d != 0 {
		return nil, 0, fmt.Errorf("routed lm terminal: layer output len=%d not divisible by hidden %d", len(layerOutput), d)
	}
	if len(terminal.FinalNorm) != d {
		return nil, 0, fmt.Errorf("routed lm terminal: final norm len=%d, want %d", len(terminal.FinalNorm), d)
	}
	rows := len(layerOutput) / d
	finalHidden := make([]float32, len(layerOutput))
	hostmath.RMSNormInto(finalHidden, layerOutput, terminal.FinalNorm, rows, d, cfg.RMSNormEps)
	return finalHidden, rows, nil
}

// TerminalTopToken: streamed argmax over the head rows against the LAST
// final-hidden row; the 0.5GB head is never materialized. Ties keep the
// lowest id.
func TerminalTopToken(layerOutput []float32, cfg Config, terminal TerminalWeights) (int, float32, error) {
	finalHidden, rows, err := TerminalFinalHidden(layerOutput, cfg, terminal)
	if err != nil {
		return 0, 0, err
	}
	last := finalHidden[(rows-1)*cfg.HiddenSize:]
	return streamedArgmaxDot(terminal, cfg, last)
}

func streamedArgmaxDot(terminal TerminalWeights, cfg Config, vector []float32) (int, float32, error) {
	d := cfg.HiddenSize
	elemBytes := 0
	switch terminal.Head.DType {
	case "BF16":
		elemBytes = 2
	case "F32":
		elemBytes = 4
	default:
		return 0, 0, fmt.Errorf("routed lm terminal: head dtype %s unsupported", terminal.Head.DType)
	}
	rowBytes := d * elemBytes
	chunkRows := terminalScratchBytes / rowBytes
	if chunkRows < 1 {
		chunkRows = 1
	}
	scratch := make([]byte, chunkRows*rowBytes)
	bestID, bestLogit := -1, float32(math.Inf(-1))
	for start := 0; start < cfg.VocabSize; start += chunkRows {
		count := min(chunkRows, cfg.VocabSize-start)
		chunk := scratch[:count*rowBytes]
		if _, err := terminal.Head.ReadAt(chunk, int64(start)*int64(rowBytes)); err != nil {
			return 0, 0, fmt.Errorf("routed lm terminal head read: %w", err)
		}
		for row := 0; row < count; row++ {
			rowRaw := chunk[row*rowBytes : (row+1)*rowBytes]
			var sum float64
			if elemBytes == 2 {
				for i, v := range vector[:d] {
					bits := binary.LittleEndian.Uint16(rowRaw[i*2:])
					sum += float64(v) * float64(math.Float32frombits(uint32(bits)<<16))
				}
			} else {
				for i, v := range vector[:d] {
					sum += float64(v) * float64(math.Float32frombits(binary.LittleEndian.Uint32(rowRaw[i*4:])))
				}
			}
			logit := float32(sum)
			if id := start + row; bestID < 0 || logit > bestLogit {
				bestID, bestLogit = id, logit
			}
		}
	}
	return bestID, bestLogit, nil
}

// TerminalProbeValues: final-hidden probes + logits for selected head rows.
func TerminalProbeValues(layerOutput []float32, cfg Config, terminal TerminalWeights, hiddenIndices, headRows []int) ([]float32, []float32, error) {
	finalHidden, rows, err := TerminalFinalHidden(layerOutput, cfg, terminal)
	if err != nil {
		return nil, nil, err
	}
	hidden := make([]float32, len(hiddenIndices))
	for i, index := range hiddenIndices {
		if index < 0 || index >= len(finalHidden) {
			return nil, nil, fmt.Errorf("routed lm terminal probe index %d outside %d values", index, len(finalHidden))
		}
		hidden[i] = finalHidden[index]
	}
	weightRows, err := ReadTensorRowsF32(terminal.Head, cfg.HiddenSize, headRows)
	if err != nil {
		return nil, nil, err
	}
	d := cfg.HiddenSize
	logits := make([]float32, len(headRows))
	hostmath.LinearF64(logits, finalHidden[(rows-1)*d:rows*d], weightRows, nil, 1, d, len(headRows))
	return hidden, logits, nil
}
