package com.ml.shubham0204.docqa.data

import org.koin.core.annotation.Single

@Single
class ChunksDB {
    private val chunksBox = ObjectBoxStore.store.boxFor(Chunk::class.java)

    fun addChunk(chunk: Chunk) {
        chunksBox.put(chunk)
    }

    fun getSimilarChunks(
        queryEmbedding: FloatArray,
        n: Int = 5,
        candidateCount: Int = 25,
    ): List<Pair<Float, Chunk>> {
        /*
        Use maxResultCount to set the maximum number of objects to return by the ANN condition.
        Hint: it can also be used as the "ef" HNSW parameter to increase the search quality in combination
        with a query limit. For example, use maxResultCount of 100 with a Query limit of 10 to have 10 results
        that are of potentially better quality than just passing in 10 for maxResultCount
        (quality/performance tradeoff).
        
        When reranking is enabled, we fetch more candidates (e.g., 50) and let the reranker
        select the most relevant ones.
         */
        val maxCandidates = maxOf(candidateCount, n)
        return chunksBox
            .query(Chunk_.chunkEmbedding.nearestNeighbors(queryEmbedding, maxCandidates))
            .build()
            .findWithScores()
            .map { Pair(it.score.toFloat(), it.get()) }
            .take(minOf(candidateCount, chunksBox.count().toInt()))
    }

    fun removeChunks(docId: Long) {
        chunksBox.removeByIds(
            chunksBox
                .query(Chunk_.docId.equal(docId))
                .build()
                .findIds()
                .toList(),
        )
    }
}
