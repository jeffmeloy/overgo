package tokenizer

// mergeQueue shares container/heap mechanics while leaving each tokenizer's
// candidate representation and priority rule independent.
type mergeQueue[T any] struct {
	values []T
	less   func(T, T) bool
}

// Len reports the pending candidate count.
func (q mergeQueue[T]) Len() int { return len(q.values) }

// Less applies the priority rule supplied by the caller.
func (q mergeQueue[T]) Less(i, j int) bool { return q.less(q.values[i], q.values[j]) }

// Swap exchanges candidate positions in the heap.
func (q mergeQueue[T]) Swap(i, j int) { q.values[i], q.values[j] = q.values[j], q.values[i] }

// Push appends one candidate for container/heap.
func (q *mergeQueue[T]) Push(value any) { q.values = append(q.values, value.(T)) }

// Pop removes the last candidate for container/heap.
func (q *mergeQueue[T]) Pop() any {
	old := q.values
	last := old[len(old)-1]
	q.values = old[:len(old)-1]
	return last
}
