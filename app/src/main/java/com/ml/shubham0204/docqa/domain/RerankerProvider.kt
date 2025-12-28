package com.ml.shubham0204.docqa.domain

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import com.ml.shubham0204.docqa.data.Chunk
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.runBlocking
import org.json.JSONObject
import org.koin.core.annotation.Single
import java.io.File
import java.nio.LongBuffer

/**
 * Provides cross-encoder reranking functionality for improving retrieval quality.
 * Uses the ms-marco-MiniLM-L-6-v2 ONNX model to score query-passage pairs.
 * 
 * The reranker takes (query, passage) pairs and outputs relevance scores,
 * allowing re-sorting of retrieved chunks by semantic relevance rather than
 * just embedding similarity.
 */
@Single
class RerankerProvider(
    private val context: Context,
) {
    private val ortEnvironment: OrtEnvironment = OrtEnvironment.getEnvironment()
    private lateinit var ortSession: OrtSession
    private lateinit var tokenizer: RerankerTokenizer
    
    private var isInitialized = false
    
    companion object {
        private const val RERANKER_MODEL_FILE = "reranker.onnx"
        private const val RERANKER_TOKENIZER_FILE = "reranker_tokenizer.json"
        private const val MAX_SEQUENCE_LENGTH = 512
        private const val PAD_TOKEN_ID = 0L
        private const val CLS_TOKEN_ID = 101L
        private const val SEP_TOKEN_ID = 102L
    }

    init {
        runBlocking(Dispatchers.IO) {
            initializeModel()
        }
    }
    
    private fun initializeModel() {
        try {
            val modelFile = copyToLocalStorage(RERANKER_MODEL_FILE)
            val tokenizerFile = copyToLocalStorage(RERANKER_TOKENIZER_FILE)
            
            if (modelFile.exists() && tokenizerFile.exists()) {
                val sessionOptions = OrtSession.SessionOptions().apply {
                    setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
                    setIntraOpNumThreads(4)
                }
                ortSession = ortEnvironment.createSession(modelFile.absolutePath, sessionOptions)
                tokenizer = RerankerTokenizer(tokenizerFile.readText())
                isInitialized = true
            }
        } catch (e: Exception) {
            // Model files not present - reranking will be unavailable
            isInitialized = false
        }
    }
    
    /**
     * Check if the reranker model is available and loaded.
     */
    fun isAvailable(): Boolean = isInitialized

    /**
     * Rerank the given chunks based on their relevance to the query.
     * Uses batched inference for efficiency.
     *
     * @param query The user's search query
     * @param chunks List of chunks to rerank (with their original similarity scores)
     * @return List of (score, chunk) pairs sorted by relevance (highest first)
     */
    fun rerank(
        query: String,
        chunks: List<Pair<Float, Chunk>>,
    ): List<Pair<Float, Chunk>> = runBlocking(Dispatchers.Default) {
        if (!isInitialized || chunks.isEmpty()) {
            return@runBlocking chunks
        }
        
        try {
            val scores = batchInference(query, chunks.map { it.second.chunkData })
            
            // Pair scores with chunks and sort by score (descending - higher is more relevant)
            chunks.zip(scores)
                .map { (chunkPair, score) -> Pair(score, chunkPair.second) }
                .sortedByDescending { it.first }
        } catch (e: Exception) {
            // On error, return original order
            chunks
        }
    }

    /**
     * Perform batched inference on all query-passage pairs.
     * Tokenizes all pairs, runs them through the model in a single call.
     */
    private fun batchInference(
        query: String,
        passages: List<String>,
    ): List<Float> {
        val batchSize = passages.size
        
        // Tokenize all query-passage pairs
        val tokenizedPairs = passages.map { passage ->
            tokenizer.tokenizePair(query, passage, MAX_SEQUENCE_LENGTH)
        }
        
        // Prepare batched tensors
        val inputIds = Array(batchSize) { i -> 
            tokenizedPairs[i].inputIds.toLongArray()
        }
        val attentionMask = Array(batchSize) { i -> 
            tokenizedPairs[i].attentionMask.toLongArray()
        }
        val tokenTypeIds = Array(batchSize) { i -> 
            tokenizedPairs[i].tokenTypeIds.toLongArray()
        }
        
        // Create ONNX tensors
        val inputIdsTensor = OnnxTensor.createTensor(ortEnvironment, inputIds)
        val attentionMaskTensor = OnnxTensor.createTensor(ortEnvironment, attentionMask)
        val tokenTypeIdsTensor = OnnxTensor.createTensor(ortEnvironment, tokenTypeIds)
        
        val inputs = mapOf(
            "input_ids" to inputIdsTensor,
            "attention_mask" to attentionMaskTensor,
            "token_type_ids" to tokenTypeIdsTensor,
        )
        
        // Run inference
        val results = ortSession.run(inputs)
        
        // Extract scores - cross-encoders typically output logits for relevance
        @Suppress("UNCHECKED_CAST")
        val outputTensor = results[0].value as Array<FloatArray>
        
        // Clean up tensors
        inputIdsTensor.close()
        attentionMaskTensor.close()
        tokenTypeIdsTensor.close()
        results.close()
        
        // For binary cross-encoders, we take the logit directly as the score
        // Some models output [batch, 1] or [batch, 2] - handle both cases
        return outputTensor.map { logits ->
            if (logits.size == 1) {
                logits[0]
            } else {
                // For 2-class output, use the positive class logit
                logits[1]
            }
        }
    }

    private fun copyToLocalStorage(filename: String): File {
        val storageFile = File(context.filesDir, filename)
        if (!storageFile.exists()) {
            try {
                val bytes = context.assets.open(filename).readBytes()
                storageFile.writeBytes(bytes)
            } catch (e: Exception) {
                // Asset not found
            }
        }
        return storageFile
    }
}

/**
 * Simple tokenizer for the reranker model.
 * Handles BERT-style tokenization with special tokens for query-passage pairs.
 */
class RerankerTokenizer(tokenizerJson: String) {
    private val vocab: Map<String, Long>
    private val unkTokenId: Long
    
    companion object {
        private const val CLS_TOKEN = "[CLS]"
        private const val SEP_TOKEN = "[SEP]"
        private const val PAD_TOKEN = "[PAD]"
        private const val UNK_TOKEN = "[UNK]"
    }
    
    init {
        val json = JSONObject(tokenizerJson)
        val model = json.getJSONObject("model")
        val vocabJson = model.getJSONObject("vocab")
        
        vocab = buildMap {
            vocabJson.keys().forEach { key ->
                put(key, vocabJson.getLong(key))
            }
        }
        unkTokenId = vocab[UNK_TOKEN] ?: 100L
    }
    
    /**
     * Tokenize a query-passage pair for cross-encoder input.
     * Format: [CLS] query tokens [SEP] passage tokens [SEP] [PAD]...
     */
    fun tokenizePair(
        query: String,
        passage: String,
        maxLength: Int,
    ): TokenizedPair {
        val queryTokens = tokenize(query)
        val passageTokens = tokenize(passage)
        
        // Reserve space for special tokens: [CLS] query [SEP] passage [SEP]
        val maxQueryLen = (maxLength - 3) / 2
        val maxPassageLen = maxLength - 3 - minOf(queryTokens.size, maxQueryLen)
        
        val truncatedQuery = queryTokens.take(maxQueryLen)
        val truncatedPassage = passageTokens.take(maxPassageLen)
        
        val inputIds = mutableListOf<Long>()
        val attentionMask = mutableListOf<Long>()
        val tokenTypeIds = mutableListOf<Long>()
        
        // [CLS] token
        inputIds.add(vocab[CLS_TOKEN] ?: 101L)
        attentionMask.add(1L)
        tokenTypeIds.add(0L)
        
        // Query tokens (segment 0)
        truncatedQuery.forEach { token ->
            inputIds.add(vocab[token] ?: unkTokenId)
            attentionMask.add(1L)
            tokenTypeIds.add(0L)
        }
        
        // [SEP] token after query
        inputIds.add(vocab[SEP_TOKEN] ?: 102L)
        attentionMask.add(1L)
        tokenTypeIds.add(0L)
        
        // Passage tokens (segment 1)
        truncatedPassage.forEach { token ->
            inputIds.add(vocab[token] ?: unkTokenId)
            attentionMask.add(1L)
            tokenTypeIds.add(1L)
        }
        
        // [SEP] token after passage
        inputIds.add(vocab[SEP_TOKEN] ?: 102L)
        attentionMask.add(1L)
        tokenTypeIds.add(1L)
        
        // Pad to maxLength
        val padTokenId = vocab[PAD_TOKEN] ?: 0L
        while (inputIds.size < maxLength) {
            inputIds.add(padTokenId)
            attentionMask.add(0L)
            tokenTypeIds.add(0L)
        }
        
        return TokenizedPair(inputIds, attentionMask, tokenTypeIds)
    }
    
    /**
     * Basic wordpiece-style tokenization.
     * Lowercases and splits on whitespace and punctuation.
     */
    private fun tokenize(text: String): List<String> {
        val tokens = mutableListOf<String>()
        val words = text.lowercase().split(Regex("\\s+"))
        
        for (word in words) {
            if (word.isEmpty()) continue
            
            // Try to find the word in vocab
            if (vocab.containsKey(word)) {
                tokens.add(word)
            } else {
                // Simple character-level fallback with ## prefix for continuation
                var remaining = word
                var isFirst = true
                
                while (remaining.isNotEmpty()) {
                    var found = false
                    
                    // Try to find the longest matching prefix
                    for (end in remaining.length downTo 1) {
                        val subword = if (isFirst) {
                            remaining.substring(0, end)
                        } else {
                            "##" + remaining.substring(0, end)
                        }
                        
                        if (vocab.containsKey(subword)) {
                            tokens.add(subword)
                            remaining = remaining.substring(end)
                            isFirst = false
                            found = true
                            break
                        }
                    }
                    
                    if (!found) {
                        // Use [UNK] for unknown characters
                        tokens.add(UNK_TOKEN)
                        remaining = remaining.drop(1)
                        isFirst = false
                    }
                }
            }
        }
        
        return tokens
    }
}

data class TokenizedPair(
    val inputIds: List<Long>,
    val attentionMask: List<Long>,
    val tokenTypeIds: List<Long>,
)
