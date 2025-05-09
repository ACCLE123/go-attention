package attention

import (
	"fmt"
	"math"
	"math/rand"
)

// MultiHeadConfig holds configuration for multi-head attention
type MultiHeadConfig struct {
	NumHeads    int     // Number of attention heads
	DModel      int     // Model dimension
	DKey        int     // Key dimension per head
	DValue      int     // Value dimension per head
	DropoutRate float64 // Dropout rate (not implemented in this version for simplicity)
}

// MultiHeadAttention implements multi-head attention mechanism
type MultiHeadAttention struct {
	config MultiHeadConfig

	// Linear projections for each head
	queryProj []Matrix // [num_heads][d_model][d_k]
	keyProj   []Matrix // [num_heads][d_model][d_k]
	valueProj []Matrix // [num_heads][d_model][d_v]

	// Output projection
	outputProj Matrix // [num_heads * d_v][d_model]
}

// NewMultiHeadAttention creates a new multi-head attention module
func NewMultiHeadAttention(config MultiHeadConfig) (*MultiHeadAttention, error) {
	if config.NumHeads <= 0 {
		return nil, fmt.Errorf("number of heads must be positive, got %d", config.NumHeads)
	}
	if config.DModel <= 0 {
		return nil, fmt.Errorf("model dimension must be positive, got %d", config.DModel)
	}
	if config.DModel%config.NumHeads != 0 {
		return nil, fmt.Errorf("model dimension (%d) must be divisible by number of heads (%d)", config.DModel, config.NumHeads)
	}

	mha := &MultiHeadAttention{
		config:     config,
		queryProj:  make([]Matrix, config.NumHeads),
		keyProj:    make([]Matrix, config.NumHeads),
		valueProj:  make([]Matrix, config.NumHeads),
		outputProj: make(Matrix, config.DModel),
	}

	// Initialize projections with random weights by Xavier initialization
	for h := 0; h < config.NumHeads; h++ {
		mha.queryProj[h] = randomMatrix(config.DModel, config.DKey)
		mha.keyProj[h] = randomMatrix(config.DModel, config.DKey)
		mha.valueProj[h] = randomMatrix(config.DModel, config.DValue)
	}
	mha.outputProj = randomMatrix(config.NumHeads*config.DValue, config.DModel)

	return mha, nil
}

// Forward computes multi-head attention
// query, key, value: [batch_size, seq_len, d_model]
func (mha *MultiHeadAttention) Forward(query, key, value Matrix) (Matrix, error) {
	batchSize := len(query)
	if batchSize != len(key) || batchSize != len(value) {
		return nil, fmt.Errorf("batch size mismatch: query(%d), key(%d), value(%d)", batchSize, len(key), len(value))
	}

	// Compute attention for each head
	headOutputs := make([]Matrix, mha.config.NumHeads)
	for h := 0; h < mha.config.NumHeads; h++ {
		// Project inputs
		projQuery, err := projectBatch(query, mha.queryProj[h])
		if err != nil {
			return nil, fmt.Errorf("projecting query for head %d: %w", h, err)
		}

		projKey, err := projectBatch(key, mha.keyProj[h])
		if err != nil {
			return nil, fmt.Errorf("projecting key for head %d: %w", h, err)
		}

		projValue, err := projectBatch(value, mha.valueProj[h])
		if err != nil {
			return nil, fmt.Errorf("projecting value for head %d: %w", h, err)
		}

		// Initialize head output
		headOutputs[h] = make(Matrix, batchSize)

		// Compute attention for each item in the batch
		for b := 0; b < batchSize; b++ {
			attended, _, err := DotProductAttention(projQuery[b], projKey, projValue)
			if err != nil {
				return nil, fmt.Errorf("computing attention for batch %d, head %d: %w", b, h, err)
			}
			headOutputs[h][b] = attended
		}
	}

	// Concatenate and project heads
	output := make(Matrix, batchSize)
	for b := 0; b < batchSize; b++ {
		// Concatenate all head outputs
		concat := make(Vector, 0, mha.config.NumHeads*mha.config.DValue)
		for h := 0; h < mha.config.NumHeads; h++ {
			concat = append(concat, headOutputs[h][b]...)
		}
		output[b] = concat
	}
	output, err := projectBatch(output, mha.outputProj)
	if err != nil {
		return nil, fmt.Errorf("projecting output for head %d: %w", mha.config.NumHeads, err)
	}

	return output, nil
}

// Helper functions

func randomMatrix(rows, cols int) Matrix {
	mat := make(Matrix, rows)
	scale := math.Sqrt(6.0 / float64(rows+cols)) // Xavier initialization
	for i := range mat {
		mat[i] = make(Vector, cols)
		for j := range mat[i] {
			mat[i][j] = (rand.Float64() - 0.5) * scale
		}
	}
	return mat
}

func projectBatch(input Matrix, weights Matrix) (Matrix, error) {
	output := make(Matrix, len(input))
	for i, vec := range input {
		projected, err := projectVector(vec, weights)
		if err != nil {
			return nil, err
		}
		output[i] = projected
	}
	return output, nil
}

func projectVector(input Vector, weights Matrix) (Vector, error) {
	if len(weights) == 0 || len(weights[0]) == 0 {
		return nil, fmt.Errorf("empty weight matrix")
	}
	if len(input) != len(weights) {
		return nil, fmt.Errorf("input dimension (%d) does not match weights (%d)", len(input), len(weights))
	}

	output := make(Vector, len(weights[0]))
	for i := range weights[0] {
		for j, w := range weights {
			output[i] += input[j] * w[i]
		}
	}
	return output, nil
}
