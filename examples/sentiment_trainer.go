package main

import (
	"encoding/json"
	"fmt"
	"github.com/ACCLE123/go-attention/attention"
	"log"
	"math"
	"math/rand"
	"os"
	"strings"
	"time"
)

type IMDBReview struct {
	Text  string  `json:"text"`
	Label float64 `json:"label"`
}

type TrainingExample struct {
	Text      string  `json:"text"`
	Sentiment float64 `json:"sentiment"`
}

type Model struct {
	WordEmbedding []attention.Vector `json:"word_embedding"` // Simple word embedding lookup
	QueryWeights  attention.Matrix   `json:"query_weights"`  // Query projection weights
	KeyWeights    attention.Matrix   `json:"key_weights"`    // Key projection weights
	ValueWeights  attention.Matrix   `json:"value_weights"`  // Value projection weights
	OutputWeights attention.Vector   `json:"output_weights"` // Output layer weights
	VocabSize     int                `json:"vocab_size"`
	EmbedDim      int                `json:"embed_dim"`
	AttnDim       int                `json:"attn_dim"`
}

type EvaluationMetrics struct {
	Accuracy  float64
	Precision float64
	Recall    float64
	F1        float64
	TruePos   int
	TrueNeg   int
	FalsePos  int
	FalseNeg  int
}

func NewModel(vocabSize, embedDim, attnDim int) *Model {
	// Initialize word embeddings with Xavier initialization
	scale := math.Sqrt(6.0 / float64(embedDim))
	wordEmbedding := make([]attention.Vector, vocabSize)
	for i := range wordEmbedding {
		wordEmbedding[i] = make(attention.Vector, embedDim)
		for j := range wordEmbedding[i] {
			wordEmbedding[i][j] = rand.NormFloat64() * scale
		}
	}

	// Initialize attention weights
	queryWeights := make(attention.Matrix, embedDim)
	keyWeights := make(attention.Matrix, embedDim)
	valueWeights := make(attention.Matrix, embedDim)
	for i := 0; i < embedDim; i++ {
		queryWeights[i] = make(attention.Vector, attnDim)
		keyWeights[i] = make(attention.Vector, attnDim)
		valueWeights[i] = make(attention.Vector, attnDim)
		for j := 0; j < attnDim; j++ {
			queryWeights[i][j] = rand.NormFloat64() * scale
			keyWeights[i][j] = rand.NormFloat64() * scale
			valueWeights[i][j] = rand.NormFloat64() * scale
		}
	}

	// Initialize output weights
	outputWeights := make(attention.Vector, attnDim)
	for i := range outputWeights {
		outputWeights[i] = rand.NormFloat64() * scale
	}

	return &Model{
		WordEmbedding: wordEmbedding,
		QueryWeights:  queryWeights,
		KeyWeights:    keyWeights,
		ValueWeights:  valueWeights,
		OutputWeights: outputWeights,
		VocabSize:     vocabSize,
		EmbedDim:      embedDim,
		AttnDim:       attnDim,
	}
}

// hashWord creates a simple hash for word to vocabulary mapping
func hashWord(word string) int {
	hash := 0
	for _, c := range word {
		hash = (hash*31 + int(c)) % 10000
	}
	return hash
}

type Batch struct {
	Embeddings []attention.Matrix
	Targets    []float64
}

func (m *Model) createBatch(examples []TrainingExample, batchSize int) []Batch {
	var batches []Batch
	for i := 0; i < len(examples); i += batchSize {
		end := i + batchSize
		if end > len(examples) {
			end = len(examples)
		}

		batch := Batch{
			Embeddings: make([]attention.Matrix, end-i),
			Targets:    make([]float64, end-i),
		}

		for j, example := range examples[i:end] {
			words := strings.Fields(strings.ToLower(example.Text))
			if len(words) > 100 {
				words = words[:100]
			}

			embeddings := make(attention.Matrix, len(words))
			for k, word := range words {
				wordIdx := hashWord(word) % m.VocabSize
				embeddings[k] = m.WordEmbedding[wordIdx]
			}
			batch.Embeddings[j] = embeddings
			batch.Targets[j] = example.Sentiment
		}
		batches = append(batches, batch)
	}
	return batches
}

func clipGradients(grads attention.Vector, maxNorm float64) {
	var norm float64
	for _, g := range grads {
		norm += g * g
	}
	norm = math.Sqrt(norm)
	if norm > maxNorm {
		scale := maxNorm / norm
		for i := range grads {
			grads[i] *= scale
		}
	}
}

func (m *Model) Train(examples []TrainingExample, epochs int, learningRate float64) error {
	rand.Seed(time.Now().UnixNano())

	// Training hyperparameters
	batchSize := 32
	maxGradNorm := 5.0
	l2Reg := 0.01
	lrDecay := 0.95

	// Split into training and validation sets (80-20 split)
	rand.Shuffle(len(examples), func(i, j int) {
		examples[i], examples[j] = examples[j], examples[i]
	})
	splitIdx := int(float64(len(examples)) * 0.8)
	trainExamples := examples[:splitIdx]
	valExamples := examples[splitIdx:]

	bestValAcc := 0.0
	noImprovementCount := 0
	currentLR := learningRate

	for epoch := 0; epoch < epochs; epoch++ {
		// Create batches
		batches := m.createBatch(trainExamples, batchSize)

		// Training phase
		totalLoss := 0.0
		correct := 0
		totalExamples := 0

		for batchIdx, batch := range batches {
			batchLoss := 0.0
			batchCorrect := 0

			// Accumulate gradients for the batch
			queryGrads := make(attention.Matrix, m.EmbedDim)
			keyGrads := make(attention.Matrix, m.EmbedDim)
			valueGrads := make(attention.Matrix, m.EmbedDim)
			outputGrads := make(attention.Vector, m.AttnDim)

			// Initialize gradient matrices
			for i := 0; i < m.EmbedDim; i++ {
				queryGrads[i] = make(attention.Vector, m.AttnDim)
				keyGrads[i] = make(attention.Vector, m.AttnDim)
				valueGrads[i] = make(attention.Vector, m.AttnDim)
			}

			for i := range batch.Embeddings {
				// Forward pass
				queries := m.project(batch.Embeddings[i], m.QueryWeights)
				keys := m.project(batch.Embeddings[i], m.KeyWeights)
				values := m.project(batch.Embeddings[i], m.ValueWeights)

				globalContext, scores, _ := attention.DotProductAttention(queries[0], keys, values)
				//fmt.Println("len: ", len(scores))

				// Final prediction
				logit := dotProduct(globalContext, m.OutputWeights)
				prediction := sigmoid(logit)

				// Compute loss with L2 regularization
				target := batch.Targets[i]
				loss := -(target*math.Log(prediction+1e-10) + (1-target)*math.Log(1-prediction+1e-10))

				// Add L2 regularization
				l2Loss := 0.0
				for _, w := range m.OutputWeights {
					l2Loss += w * w
				}
				loss += l2Reg * l2Loss * 0.5
				batchLoss += loss

				// Backward pass
				backError := prediction - target

				// Output weight gradients
				for j := range outputGrads {
					outputGrads[j] += backError*globalContext[j] + l2Reg*m.OutputWeights[j]
				}

				// Attention gradients
				contextGrad := make(attention.Vector, m.AttnDim)
				for j := range contextGrad {
					contextGrad[j] = backError * m.OutputWeights[j]
				}

				// Update attention weight gradients
				for j := range batch.Embeddings[i] {
					if j >= m.AttnDim {
						continue
					}
					for k := 0; k < m.AttnDim; k++ {
						queryGrad := 0.0
						keyGrad := 0.0
						valueGrad := 0.0

						for d := range scores {
							queryGrad += contextGrad[j] * scores[d] * keys[j][k] / math.Sqrt(float64(m.AttnDim))
							keyGrad += contextGrad[j] * scores[d] * queries[j][k] / math.Sqrt(float64(m.AttnDim))
							valueGrad += contextGrad[j] * scores[d]
						}

						queryGrads[j][k] += queryGrad
						keyGrads[j][k] += keyGrad
						valueGrads[j][k] += valueGrad
					}
				}

				// Calculate accuracy
				predictedClass := 0.0
				if prediction > 0.5 {
					predictedClass = 1.0
				}
				if predictedClass == target {
					batchCorrect++
				}
			}

			// Clip and apply gradients
			clipGradients(outputGrads, maxGradNorm)
			for i := range m.OutputWeights {
				m.OutputWeights[i] -= currentLR * outputGrads[i]
			}

			for i := 0; i < m.EmbedDim; i++ {
				clipGradients(queryGrads[i], maxGradNorm)
				clipGradients(keyGrads[i], maxGradNorm)
				clipGradients(valueGrads[i], maxGradNorm)
				for j := 0; j < m.AttnDim; j++ {
					m.QueryWeights[i][j] -= currentLR * queryGrads[i][j]
					m.KeyWeights[i][j] -= currentLR * keyGrads[i][j]
					m.ValueWeights[i][j] -= currentLR * valueGrads[i][j]
				}
			}

			totalLoss += batchLoss
			correct += batchCorrect
			totalExamples += len(batch.Embeddings)

			// Print batch progress
			if (batchIdx+1)%10 == 0 {
				fmt.Printf("Epoch %d, Batch %d/%d, Loss: %.4f, Accuracy: %.2f%%\n",
					epoch+1, batchIdx+1, len(batches),
					batchLoss/float64(len(batch.Embeddings)),
					float64(batchCorrect)*100.0/float64(len(batch.Embeddings)))
			}
		}

		trainLoss := totalLoss / float64(totalExamples)
		trainAcc := float64(correct) * 100.0 / float64(totalExamples)

		// Validation phase
		valLoss, valAcc := m.evaluate(valExamples)

		fmt.Printf("Epoch %d complete:\n", epoch+1)
		fmt.Printf("  Training - Loss: %.4f, Accuracy: %.2f%%\n", trainLoss, trainAcc)
		fmt.Printf("  Validation - Loss: %.4f, Accuracy: %.2f%%\n", valLoss, valAcc)

		// Learning rate decay and early stopping
		if valAcc > bestValAcc {
			bestValAcc = valAcc
			noImprovementCount = 0
		} else {
			noImprovementCount++
			if noImprovementCount >= 2 {
				currentLR *= lrDecay
				fmt.Printf("  Reducing learning rate to %.6f\n", currentLR)
				noImprovementCount = 0
			}
		}

		if currentLR < learningRate*0.01 {
			fmt.Println("Learning rate too small, stopping training")
			break
		}
	}

	return nil
}

// sigmoid computes the sigmoid function
func sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

// dotProduct computes the dot product of two vectors
func dotProduct(a, b attention.Vector) float64 {
	sum := 0.0
	for i := range a {
		sum += a[i] * b[i]
	}
	return sum
}

// project applies weight matrix to input
func (m *Model) project(input attention.Matrix, weights attention.Matrix) attention.Matrix {
	output := make(attention.Matrix, len(input))
	for i := range output {
		output[i] = make(attention.Vector, m.AttnDim)
		for j := 0; j < m.AttnDim; j++ {
			for k := 0; k < m.EmbedDim && k < len(input[i]); k++ {
				output[i][j] += input[i][k] * weights[k][j]
			}
		}
	}
	return output
}

func loadIMDBData(filename string) ([]TrainingExample, error) {
	data, err := os.ReadFile(filename)
	if err != nil {
		return nil, fmt.Errorf("failed to read file: %v", err)
	}

	// Split into lines and parse each line as a separate JSON object
	lines := strings.Split(string(data), "\n")
	examples := make([]TrainingExample, 0, len(lines))

	for _, line := range lines {
		if len(strings.TrimSpace(line)) == 0 {
			continue
		}

		var review IMDBReview
		if err := json.Unmarshal([]byte(line), &review); err != nil {
			return nil, fmt.Errorf("failed to parse JSON line: %v", err)
		}

		// Clean the text by removing HTML tags
		text := strings.ReplaceAll(review.Text, "<br />", " ")
		text = strings.ReplaceAll(text, "<br/>", " ")
		text = strings.ReplaceAll(text, "\\/", "/")

		examples = append(examples, TrainingExample{
			Text:      text,
			Sentiment: review.Label,
		})
	}

	return examples, nil
}

// evaluate computes loss and accuracy on a dataset
func (m *Model) evaluate(examples []TrainingExample) (float64, float64) {
	totalLoss := 0.0
	correct := 0

	for _, example := range examples {
		prediction := m.predict(example.Text)

		loss := -(example.Sentiment*math.Log(prediction+1e-10) +
			(1-example.Sentiment)*math.Log(1-prediction+1e-10))
		totalLoss += loss

		predictedClass := 0.0
		if prediction > 0.5 {
			predictedClass = 1.0
		}
		if predictedClass == example.Sentiment {
			correct++
		}
	}

	return totalLoss / float64(len(examples)),
		float64(correct) * 100.0 / float64(len(examples))
}

// Predict makes a prediction on new text
func (m *Model) Predict(tests []TrainingExample) float64 {
	right, all := 0, len(tests)
	for _, test := range tests {
		ans := m.predict(test.Text)
		if test.Sentiment == 1 && ans >= 0.5 {
			right++
		}
		if test.Sentiment == 0 && ans < 0.5 {
			right++
		}
	}
	return float64(right) / float64(all)
}

func (m *Model) predict(text string) float64 {
	// Tokenize
	words := strings.Fields(strings.ToLower(text))
	if len(words) > 100 {
		words = words[:100]
	}
	if len(words) == 0 {
		return 0.5 // Default prediction for empty text
	}

	// Get embeddings
	embeddings := make(attention.Matrix, len(words))
	for i, word := range words {
		wordIdx := hashWord(word) % m.VocabSize
		embeddings[i] = make(attention.Vector, m.EmbedDim)
		copy(embeddings[i], m.WordEmbedding[wordIdx])
	}

	// Apply attention
	queries := m.project(embeddings, m.QueryWeights)
	keys := m.project(embeddings, m.KeyWeights)
	values := m.project(embeddings, m.ValueWeights)

	context, _, _ := attention.DotProductAttention(queries[0], keys, values)

	// Final prediction
	logit := dotProduct(context, m.OutputWeights)
	return sigmoid(logit)
}

// SaveModel saves the model to a file
func (m *Model) SaveModel(filename string) error {
	file, err := os.Create(filename)
	if err != nil {
		return fmt.Errorf("failed to create file: %v", err)
	}
	defer file.Close()

	encoder := json.NewEncoder(file)
	if err := encoder.Encode(m); err != nil {
		return fmt.Errorf("failed to encode model: %v", err)
	}
	return nil
}

// LoadModel loads a model from a file
func LoadModel(filename string) (*Model, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, fmt.Errorf("failed to open file: %v", err)
	}
	defer file.Close()

	var model Model
	decoder := json.NewDecoder(file)
	if err := decoder.Decode(&model); err != nil {
		return nil, fmt.Errorf("failed to decode model: %v", err)
	}

	return &model, nil
}

func main() {
	// Load IMDB dataset
	examples, err := loadIMDBData("examples/data/train.json")
	if err != nil {
		log.Fatalf("Failed to load IMDB data: %v", err)
	}
	fmt.Printf("Loaded %d examples from IMDB dataset\n", len(examples))

	// Create and train model
	model := NewModel(10000, 64, 32) // vocab size: 10000, embed dim: 64, attention dim: 32

	fmt.Println("Training model...")
	if err := model.Train(examples, 10, 0.01); err != nil { // More epochs, higher learning rate
		log.Fatalf("Training failed: %v", err)
	}

	// Save the model
	if err := model.SaveModel("sentiment_model.json"); err != nil {
		log.Fatalf("Failed to save model: %v", err)
	}
	fmt.Println("Model saved to sentiment_model.json")

	tests, err := loadIMDBData("examples/data/test.json")
	ans := model.Predict(tests)
	fmt.Println("ans: ", ans)
}
