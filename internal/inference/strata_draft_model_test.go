//go:build models || modeltest

package inference

import (
	"bufio"
	"bytes"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	"overgo/internal/dataroot"
	"overgo/internal/gguf"
	"overgo/internal/testutil"
)

// strataDraftModel is Flash-Next's MTP file under a model root, the file Strata's tools/mtp_rt.py reads.
var strataDraftModel = filepath.Join("Qwen3.8-Flash-Next", "unsloth-UD-IQ4_XS", "Qwen3.8-Flash-Next-MTP-q2_0.gguf")

// TestStrataDraftPackIsStratas holds the round's draft pack to Strata's own: made from Flash-Next's MTP file, its dense
// tensors (dense.txt's names, kinds, shapes and offsets, and dense.bin's bytes) and its experts' blobs (experts.bin's
// bytes, the first, a middle and the last chunk of experts) are the rt pack's that mtp_rt.py wrote from the same file.
func TestStrataDraftPackIsStratas(t *testing.T) {
	roots, err := dataroot.Resolve(testutil.RepoRoot(t))
	if err != nil {
		t.Fatal(err)
	}
	rt := filepath.Join(roots.References(), "strata", "mtp", "rt")
	path := roots.ResolveModelPath(strataDraftModel)
	for _, need := range []string{path, filepath.Join(rt, "dense.txt")} {
		if _, err := os.Stat(need); err != nil {
			t.Skipf("the pack's check needs %s: %v", need, err)
		}
	}
	file, err := gguf.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { file.Close() })
	dense, tensors, err := strataDraftDense(file)
	if err != nil {
		t.Fatal(err)
	}
	reference, err := os.ReadFile(filepath.Join(rt, "dense.bin"))
	if err != nil {
		t.Fatal(err)
	}
	index, err := os.Open(filepath.Join(rt, "dense.txt"))
	if err != nil {
		t.Fatal(err)
	}
	defer index.Close()
	lines := 0
	scanner := bufio.NewScanner(index)
	for scanner.Scan() {
		fields := strings.Fields(scanner.Text())
		if len(fields) != 6 {
			continue
		}
		lines++
		number := func(i int) uint64 {
			value, err := strconv.ParseUint(fields[i], 10, 64)
			if err != nil {
				t.Fatalf("dense.txt: %q", scanner.Text())
			}
			return value
		}
		name, kind, rows, cols, offset, size := fields[0], fields[1], number(2), number(3), number(4), number(5)
		ours, found := tensors[name]
		if !found {
			t.Errorf("%s is not in the pack", name)
			continue
		}
		if ours.kind != kind || ours.rows != rows || ours.cols != cols || ours.offset != offset || ours.bytes != size {
			t.Errorf("%s: %s %d %d at %d, %d bytes, where Strata's is %s %d %d at %d, %d bytes", name, ours.kind, ours.rows,
				ours.cols, ours.offset, ours.bytes, kind, rows, cols, offset, size)
			continue
		}
		if !bytes.Equal(dense[offset:offset+size], reference[offset:offset+size]) {
			t.Errorf("%s's bytes differ from dense.bin's", name)
		}
	}
	if lines != len(tensors) {
		t.Errorf("the pack holds %d dense tensors, dense.txt %d", len(tensors), lines)
	}
	experts, err := os.Open(filepath.Join(rt, "experts.bin"))
	if err != nil {
		t.Fatal(err)
	}
	defer experts.Close()
	gateUp, _ := file.Tensor("mtp.layers.0.mlp.experts.gate_up_proj")
	down, _ := file.Tensor("mtp.layers.0.mlp.experts.down_proj")
	gateUpExpert, downExpert := gateUp.Size/strataDraftExperts, down.Size/strataDraftExperts
	blob := strataDraftBlobBytes()
	for _, first := range []int{0, strataDraftExperts / 2, strataDraftExperts - strataDraftExpertChunk} {
		count := strataDraftExpertChunk
		rows, downs := make([]byte, uint64(count)*gateUpExpert), make([]byte, uint64(count)*downExpert)
		if err := file.ReadTensorRange(gateUp, uint64(first)*gateUpExpert, rows); err != nil {
			t.Fatal(err)
		}
		if err := file.ReadTensorRange(down, uint64(first)*downExpert, downs); err != nil {
			t.Fatal(err)
		}
		want := make([]byte, uint64(count)*blob)
		if _, err := experts.ReadAt(want, int64(uint64(first)*blob)); err != nil {
			t.Fatal(err)
		}
		if !bytes.Equal(strataDraftBlobs(rows, downs, count), want) {
			t.Errorf("experts %d to %d's blobs differ from experts.bin's", first, first+count-1)
		}
	}
}
