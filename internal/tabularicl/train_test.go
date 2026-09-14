package tabularicl

import (
	"math"
	"runtime"
	"slices"
	"testing"

	"overgo/internal/optimizer"
	"overgo/internal/testskip"
	"overgo/internal/trainingprogram"
)

func TestDecoderTrainerRealArtifacts(t *testing.T) {
	if testing.Short() {
		t.Skip(testskip.ShortIntegration + ": loads both TabFM heads")
	}
	golden := readGolden(t)
	for _, task := range Tasks() {
		t.Run(task, func(t *testing.T) {
			var request Request
			for _, testCase := range golden.Cases {
				if testCase.Task == task {
					request = Request{
						Task: task, X: testCase.X, Y: testCase.Y, Rows: testCase.Rows,
						Cols: testCase.Cols, TrainRows: testCase.TrainRows,
					}
					break
				}
			}
			if request.Rows == 0 {
				t.Fatalf("%s fixture absent", task)
			}
			head, err := LoadHead(headDir(t, task))
			if err != nil {
				t.Fatal(err)
			}
			before := append([]float32(nil), head.decL1W...)
			trainer, err := NewTrainer(head, optimizer.Config{
				BaseLearningRate: 1e-5, Momentum: 0.9, Schedule: optimizer.ScheduleConstant,
			})
			if err != nil {
				t.Fatal(err)
			}
			result, stepErr := trainer.Step(request)
			closeErr := trainer.Close()
			if stepErr != nil || closeErr != nil {
				t.Fatalf("step=%v close=%v", stepErr, closeErr)
			}
			if trainer.program.Objective() != trainingprogram.ObjectiveTablePrediction ||
				trainer.ParameterCount() != len(head.decL0W)+len(head.decL0B)+len(head.decL1W)+len(head.decL1B) ||
				result.Step != 1 || result.LearningRate <= 0 || result.GradientL2 <= 0 ||
				math.IsNaN(result.Loss) || math.IsInf(result.Loss, 0) || slices.Equal(before, head.decL1W) {
				t.Fatalf("result=%+v parameters=%d unchanged=%t", result, trainer.ParameterCount(), slices.Equal(before, head.decL1W))
			}
			t.Logf("%s decoder update: loss=%.6f grad_l2=%.6g parameters=%d", task, result.Loss, result.GradientL2, trainer.ParameterCount())
			head, trainer = nil, nil
			runtime.GC()
		})
	}
}
