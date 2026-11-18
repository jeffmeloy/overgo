package hostmath

import (
	"math"
	"testing"
)

func TestLayerNormAndAdaptiveAffineF64(t *testing.T) {
	normed := LayerNormF64([]float64{1, 3}, 1, 2, 0)
	if math.Abs(normed[0]+1) > 1e-12 || math.Abs(normed[1]-1) > 1e-12 {
		t.Fatalf("layer norm=%v", normed)
	}
	got := AdaptiveAffineF64(normed, []float64{0, 0}, []float32{0, 0, 1, 2}, 1, 2)
	if math.Abs(got[0]) > 1e-12 || math.Abs(got[1]-3) > 1e-12 {
		t.Fatalf("adaptive affine=%v", got)
	}
}

// TestBatchedAttentionGivesOneThreadsBits holds the batched attention (moss-speed/depth) to one thread's: its query rows
// split across the host's workers, every output equals bit for bit the value the rows give computed in order on one
// thread, each row's scores, softmax and values summed in one order, at a size the host splits (a DINOv2 encoder's 2442
// tokens are many more).
func TestBatchedAttentionGivesOneThreadsBits(t *testing.T) {
	const batches, queryRows, keyRows, hidden, heads = 2, 96, 80, 48, 3
	fill := func(count, seed int) []float32 {
		values := make([]float32, count)
		for index := range values {
			values[index] = float32(math.Sin(float64(index*seed+7))) / 2
		}
		return values
	}
	q, k, v := fill(batches*queryRows*hidden, 3), fill(batches*keyRows*hidden, 5), fill(batches*keyRows*hidden, 11)
	got := BatchedAttentionF32(q, k, v, batches, queryRows, keyRows, hidden, heads)
	headWidth := hidden / heads
	scale := 1 / math.Sqrt(float64(headWidth))
	want := make([]float32, batches*queryRows*hidden)
	scores := make([]float64, keyRows)
	for batch := range batches {
		for query := range queryRows {
			for head := range heads {
				maximum := math.Inf(-1)
				for key := range keyRows {
					var score float64
					for channel := range headWidth {
						score += float64(q[(batch*queryRows+query)*hidden+head*headWidth+channel] * k[(batch*keyRows+key)*hidden+head*headWidth+channel])
					}
					scores[key] = score * scale
					maximum = max(maximum, scores[key])
				}
				var total float64
				for key := range scores {
					scores[key] = math.Exp(scores[key] - maximum)
					total += scores[key]
				}
				for key := range keyRows {
					factor := float32(scores[key] / total)
					for channel := range headWidth {
						want[(batch*queryRows+query)*hidden+head*headWidth+channel] += factor * v[(batch*keyRows+key)*hidden+head*headWidth+channel]
					}
				}
			}
		}
	}
	for index := range want {
		if math.Float32bits(got[index]) != math.Float32bits(want[index]) {
			t.Fatalf("output %d is %v; one thread gives %v", index, got[index], want[index])
		}
	}
}
