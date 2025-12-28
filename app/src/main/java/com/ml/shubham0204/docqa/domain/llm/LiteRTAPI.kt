package com.ml.shubham0204.docqa.domain.llm

import android.content.Context
import android.util.Log
import com.google.mediapipe.tasks.genai.llminference.LlmInference
import com.google.mediapipe.tasks.genai.llminference.ProgressListener
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.channels.awaitClose
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.callbackFlow
import kotlinx.coroutines.flow.flowOn
import kotlinx.coroutines.withContext
import org.koin.core.annotation.Single

@Single
class LiteRTAPI : LLMInferenceAPI() {
    private lateinit var llmInference: LlmInference
    var isLoaded = false
    var loadedModelPath: String? = null

    class PartialProgressListener(
        private val onPartialResponseGenerated: (String) -> Unit,
        private val onSuccess: (String) -> Unit,
    ) : ProgressListener<String> {
        private var result = ""

        override fun run(
            partialResult: String?,
            done: Boolean,
        ) {
            if (done) {
                onSuccess(result)
                result = ""
            } else {
                result += partialResult ?: ""
                onPartialResponseGenerated(result)
            }
        }
    }

    fun load(
        context: Context,
        modelPath: String,
        onSuccess: () -> Unit,
        onError: (Exception) -> Unit,
    ) {
        try {
            val taskOptions =
                LlmInference.LlmInferenceOptions
                    .builder()
                    .setModelPath(modelPath)
                    .setMaxTopK(64)
                    .setMaxTokens(2048)
                    .build()
            llmInference = LlmInference.createFromOptions(context, taskOptions)
            isLoaded = true
            loadedModelPath = modelPath
            onSuccess()
        } catch (e: Exception) {
            isLoaded = false
            loadedModelPath = null
            Log.e("LiteRTAPI", "Failed to load model: ${e.message}", e)
            onError(e)
        }
    }

    override suspend fun getResponse(prompt: String): String? =
        withContext(Dispatchers.Default) {
            Log.e("APP", "Prompt given: $prompt")
            llmInference.generateResponse(prompt)
        }

    override fun getResponseStream(prompt: String): Flow<String> = callbackFlow {
        Log.e("APP", "Streaming prompt: $prompt")
        val listener = PartialProgressListener(
            onPartialResponseGenerated = { partialResponse ->
                trySend(partialResponse)
            },
            onSuccess = { finalResponse ->
                trySend(finalResponse)
                close()
            }
        )
        llmInference.generateResponseAsync(prompt, listener)
        awaitClose { /* Cleanup if needed */ }
    }.flowOn(Dispatchers.Default)

    fun unload() {
        llmInference.close()
        isLoaded = false
        loadedModelPath = null
    }
}
