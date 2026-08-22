package thoughtbank

import "testing"

func TestRouteMoETopKUnderflowRemainsConvex(t *testing.T) {
	routings := RouteMoETopK([]float32{1}, []float32{-1000, -1000}, 1, 1, 2, 2)
	if len(routings) != 1 || len(routings[0].Weights) != 2 ||
		routings[0].Weights[0]+routings[0].Weights[1] != 1 {
		t.Fatalf("underflow routing is not a convex combination: %+v", routings)
	}
}
